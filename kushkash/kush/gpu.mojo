# kushkash GPU kernels (NVIDIA sm_75+; tensor-core decode needs sm_80+).
# Cache format and page-record layout: kush/quant.mojo. All kernels work on one layer.
#
# Buffers (the host owns page allocation; vLLM's block tables plug in directly):
#   cache       bytes: record of (block b, kv head h) at cache + b*block_stride + h*head_stride
#   tail_k      IN [slots][Hkv][P][D]   K of the page being filled, exact; sequence s uses slot
#               tail_idx[s]. Its V rows are already final in the page's record (per-token codes).
#   block_table int32 [S][max_pages]    seq_lens int32 [S]
#   q, out      IN [S][Hkv*G][D]   (q head = kv_head*G + g)        IN = fp16 | bf16 | fp32
#
# Kernels
#   kush_append     one new token per sequence (decode step); quantizes a page when it fills
#   kush_write      ragged multi-token writes (prefill / chunked prefill / mixed batches):
#                   one job per (sequence, page); mode 0 quantizes completed pages, mode 1
#                   stashes the unfinished page in the tail (two launches: no tail races)
#   kush_decode_tc  fused paged decode attention on tensor cores (D % 128 == 0, G <= 8, sm_80+)
#   kush_decode     same contract on CUDA cores (any D in {64,128,256}, any G, sm_75+)
#   kush_combine    merges split-K partials (when n_splits > 1)
#
# Dequantization is never materialized in global memory: K scales are folded into Q per
# page, K biases cost one dot per page, V scales fold into the softmax weights, V biases
# into one scalar per head. Codes become floats with prmt (byte under an exponent) + one
# add, instead of int->float conversion instructions.

from std.math import exp
from std.memory import stack_allocation, bitcast
from std.sys import inlined_assembly, size_of
from max.gpu import thread_idx, block_idx, barrier, syncwarp
from max.gpu.memory import AddressSpace
from max.gpu.primitives import warp
from max.gpu.compute.mma import mma, ld_matrix
from kush.quant import PAGE, row_bytes, qparams, qcode, pack4, off_vq, off_ks, off_kb, off_vs, off_vb, fwht_scale

comptime F32P = Pointer[Float32, MutAnyOrigin]
comptime I8P = Pointer[Int8, MutAnyOrigin]
comptime I32P = Pointer[Int32, MutAnyOrigin]
comptime SF32P = Pointer[Float32, MutUntrackedOrigin, address_space=AddressSpace.SHARED]
comptime LF32P = Pointer[Float32, MutUntrackedOrigin]
comptime WARP = 32
comptime P = PAGE

comptime XOR8 = UInt32(0x80808080)     # signed int8 code -> unsigned (c + 128)
comptime XOR4 = UInt32(0x08080808)     # masked nibble (two's complement) -> unsigned (c + 8)
comptime NIB = UInt32(0x0F0F0F0F)
comptime H64 = UInt32(0x64646464)


# ------------------------------------------------------------------ rotation
@always_inline
def _mul_rn(a: Float32, b: Float32) -> Float32:
    """a * b rounded once, as its own instruction: inline PTX is opaque to LLVM and `.rn` keeps ptxas
    from contracting it into an FMA with the caller's next add (kush/quant.mojo: rotation)."""
    return inlined_assembly["mul.rn.f32 $0, $1, $2;", Float32, constraints="=f,f,f"](a, b)


@always_inline
def _fwht_warp[D: Int](x: SIMD[DType.float32, D // WARP], lane: Int) -> SIMD[DType.float32, D // WARP]:
    """Walsh-Hadamard transform of one row held by a warp: lane holds x[i] = row[lane + i*32]
    (the layout of _quantize_v_row). Canonical order (kush/quant.mojo), bit-identical to the CPU:
    strides 1..16 cross lanes (shuffle_xor), strides 32..D/2 stay in a lane's registers."""
    comptime R = D // WARP
    var y = x
    comptime for k in range(5):                               # h = 1, 2, 4, 8, 16: partner lane ^ h
        comptime h = 1 << k
        var upper = (lane & h) != 0
        comptime for i in range(R):
            var other = warp.shuffle_xor(y[i], UInt32(h))
            y[i] = other - y[i] if upper else y[i] + other    # upper: a - b with a = partner
    comptime if R >= 2:                                       # h = 32: register bit 0
        comptime for i in range(0, R, 2):
            var a = y[i]
            var b = y[i + 1]
            y[i] = a + b
            y[i + 1] = a - b
    comptime if R >= 4:                                       # h = 64: register bit 1
        comptime for i in range(R):
            comptime if i & 2 == 0:
                var a = y[i]
                var b = y[i + 2]
                y[i] = a + b
                y[i + 2] = a - b
    comptime if R >= 8:                                       # h = 128: register bit 2
        comptime for i in range(R):
            comptime if i & 4 == 0:
                var a = y[i]
                var b = y[i + 4]
                y[i] = a + b
                y[i + 4] = a - b
    comptime sc = fwht_scale[D]()
    comptime for i in range(R):
        y[i] = _mul_rn(y[i], sc)
    return y


def kush_fwht[D: Int, IN: DType](
    x: Pointer[Scalar[IN], MutAnyOrigin], y: Pointer[Scalar[IN], MutAnyOrigin], x_stride: Int32,
    y_stride: Int32, rows: Int32,
):
    """grid (ceil(rows / 4)), block 128: one warp per row. y[r] = rotated x[r] (D values per row,
    row strides in elements; y may alias x). Tests and a stand-alone rotation for rows the
    attention kernels do not rotate themselves."""
    var lane = Int(thread_idx.x) % WARP
    var r = Int(block_idx.x) * 4 + Int(thread_idx.x) // WARP
    if r >= Int(rows):
        return
    comptime R = D // WARP
    var v = SIMD[DType.float32, R]()
    comptime for i in range(R):
        v[i] = Float32(x[unsafe_offset=r * Int(x_stride) + lane + i * WARP])
    v = _fwht_warp[D](v, lane)
    comptime for i in range(R):
        y[unsafe_offset=r * Int(y_stride) + lane + i * WARP] = v[i].cast[IN]()


# ------------------------------------------------------------------ write path
@always_inline
def _rec(cache: I8P, blk: Int, h: Int, block_stride: Int32, head_stride: Int32) -> I8P:
    return cache.unsafe_offset(blk * Int(block_stride) + h * Int(head_stride))


@always_inline
def _quantize_k_page[D: Int, BITS: Int, IN: DType](
    ka: Pointer[Scalar[IN], MutAnyOrigin], stride_a: Int,
    kb_src: Pointer[Scalar[IN], MutAnyOrigin], skb: Int, split: Int, rec: I8P,
):
    """K of one page by a block of D threads (thread d = channel d, per-channel range over the
    page). Token row t < split comes from ka + t*stride_a (the tail), the rest from
    kb_src + (t - split)*skb (new tokens)."""
    comptime DC = row_bytes[D, BITS]()
    var kc = rec
    var ksp = rec.unsafe_offset(off_ks[D, BITS]()).unsafe_bitcast[Float16]()
    var kbp = rec.unsafe_offset(off_kb[D, BITS]()).unsafe_bitcast[Float16]()
    var stage_k = stack_allocation[P * D, Int8, address_space=AddressSpace.SHARED]()
    var d = Int(thread_idx.x)

    @always_inline
    def kx(t: Int) {var ka, var kb_src, var split, var stride_a, var skb, var d} -> Float32:
        return Float32(ka[unsafe_offset=t * stride_a + d]) if t < split else Float32(kb_src[unsafe_offset=(t - split) * skb + d])

    var mn = Float32.MAX
    var mx = Float32.MIN
    for t in range(P):
        var x = kx(t)
        mn = min(mn, x)
        mx = max(mx, x)
    var kp = qparams[BITS](mn, mx)
    ksp[unsafe_offset=d] = kp[0].cast[DType.float16]()
    kbp[unsafe_offset=d] = kp[1].cast[DType.float16]()
    for t in range(P):
        var c = qcode[BITS](kx(t), kp[1], kp[2])
        comptime if BITS == 8:
            kc[unsafe_offset=t * DC + d] = c
        else:
            stage_k[unsafe_offset=t * D + d] = c
    comptime if BITS == 4:
        barrier()
        if d < D // 2:
            for t in range(P):
                kc[unsafe_offset=t * DC + d] = pack4(stage_k[unsafe_offset=t * D + d], stage_k[unsafe_offset=t * D + d + D // 2])
    barrier()                                     # sources / staging are reused by the next call


@always_inline
def _quantize_v_row[D: Int, BITS: Int, IN: DType](src: Pointer[Scalar[IN], MutAnyOrigin], rec: I8P, t: Int,
                                                  lane: Int):
    """One warp: V token row src[0:D] -> row t of a record. V ranges are per token, so this is
    the row's final encoding (bit-identical to quantizing the whole page at once)."""
    comptime DC = row_bytes[D, BITS]()
    comptime R = D // WARP
    var x = stack_allocation[R, Float32]()            # R = D/32 values per lane (registers)
    var mn = Float32.MAX
    var mx = Float32.MIN
    comptime for i in range(R):
        x[unsafe_offset=i] = Float32(src[unsafe_offset=lane + i * WARP])
        mn = min(mn, x[unsafe_offset=i])
        mx = max(mx, x[unsafe_offset=i])
    var vp = qparams[BITS](warp.min(mn), warp.max(mx))
    if lane == 0:
        rec.unsafe_offset(off_vs[D, BITS]()).unsafe_bitcast[Float16]()[unsafe_offset=t] = vp[0].cast[DType.float16]()
        rec.unsafe_offset(off_vb[D, BITS]()).unsafe_bitcast[Float16]()[unsafe_offset=t] = vp[1].cast[DType.float16]()
    var vc = rec.unsafe_offset(off_vq[D, BITS]() + t * DC)
    comptime if BITS == 8:
        comptime for i in range(R):
            vc[unsafe_offset=lane + i * WARP] = qcode[BITS](x[unsafe_offset=i], vp[1], vp[2])
    else:
        comptime for i in range(R // 2):          # byte j = dim j | dim j + D/2 (= x[i + R/2])
            vc[unsafe_offset=lane + i * WARP] = pack4(qcode[BITS](x[unsafe_offset=i], vp[1], vp[2]),
                                                      qcode[BITS](x[unsafe_offset=i + R // 2], vp[1], vp[2]))


@always_inline
def _vdeq[D: Int, BITS: Int](rec: I8P, t: Int, c: Int) -> Float32:
    """Dequantized V[t, c] of a record (reads one code + the token's scale / bias)."""
    comptime DC = row_bytes[D, BITS]()
    var row = rec.unsafe_offset(off_vq[D, BITS]() + t * DC)
    var code: Int8
    comptime if BITS == 8:
        code = row[unsafe_offset=c]
    else:
        var byte = row[unsafe_offset=c % (D // 2)]
        code = (byte << 4) >> 4 if c < D // 2 else byte >> 4
    var vs = Float32(rec.unsafe_offset(off_vs[D, BITS]()).unsafe_bitcast[Float16]()[unsafe_offset=t])
    var vb = Float32(rec.unsafe_offset(off_vb[D, BITS]()).unsafe_bitcast[Float16]()[unsafe_offset=t])
    return Float32(code) * vs + vb


def kush_append[D: Int, BITS: Int, IN: DType](
    k_new: Pointer[Scalar[IN], MutAnyOrigin], v_new: Pointer[Scalar[IN], MutAnyOrigin], k_stride: Int32,
    v_stride: Int32, seq_lens: I32P, pos_offset: Int32, advance: Int32, tail_idx: I32P,
    cache: I8P, block_stride: Int32, head_stride: Int32, tail_k: Pointer[Scalar[IN], MutAnyOrigin],
    block_table: I32P, max_pages: Int32, hkv: Int32,
):
    """grid (S), block D. One token per sequence s (row s of k_new/v_new; row strides may
    differ, head h at +h*D) at position seq_lens[s] + pos_offset (HF: 0, then seq_lens += 1
    if advance; vLLM: -1, seq_lens already counts the token). Negative positions (padding
    rows) are skipped. K goes to the exact tail and is quantized when the page fills; V goes
    straight into its record row. The token's page must be in block_table."""
    var s = Int(block_idx.x)
    var H = Int(hkv)
    var d = Int(thread_idx.x)
    var pos = Int(seq_lens[unsafe_offset=s]) + Int(pos_offset)
    if pos < 0:
        return
    var slot = pos % P
    var tb = Int(tail_idx[unsafe_offset=s]) * H * P * D
    var blk = Int(block_table[unsafe_offset=s * Int(max_pages) + pos // P])
    for h in range(H):
        tail_k[unsafe_offset=tb + (h * P + slot) * D + d] = k_new[unsafe_offset=s * Int(k_stride) + h * D + d]
    for h in range(d // WARP, H, D // WARP):
        _quantize_v_row[D, BITS, IN](v_new.unsafe_offset(s * Int(v_stride) + h * D),
                                     _rec(cache, blk, h, block_stride, head_stride), slot, d % WARP)
    barrier()
    if slot == P - 1:
        for h in range(H):
            var t0 = tail_k.unsafe_offset(tb + h * P * D)
            _quantize_k_page[D, BITS, IN](t0, D, t0, D, P, _rec(cache, blk, h, block_stride, head_stride))
    if advance != 0 and d == 0:
        seq_lens[unsafe_offset=s] = Int32(pos + 1)


def kush_write[D: Int, BITS: Int, IN: DType](
    k_new: Pointer[Scalar[IN], MutAnyOrigin], v_new: Pointer[Scalar[IN], MutAnyOrigin], k_stride: Int32,
    v_stride: Int32, query_start_loc: I32P, seq_lens: I32P, tail_idx: I32P, job_seq: I32P, job_page: I32P,
    mode: Int32, cache: I8P, block_stride: Int32, head_stride: Int32, tail_k: Pointer[Scalar[IN], MutAnyOrigin],
    block_table: I32P, max_pages: Int32, hkv: Int32,
):
    """grid (num_jobs, Hkv), block D. Sequence s owns new-token rows [qsl[s], qsl[s+1]) at
    positions [ctx, seq_lens[s]) with ctx = seq_lens[s] - its row count. Job j covers logical
    page job_page[j] of job_seq[j]. mode 0: write the page's new V rows; if the page is
    complete, quantize its K (older tokens from the tail, newer from the new rows). mode 1:
    if it is not, copy its new K rows into the tail. Launch mode 0 for all jobs, then mode 1."""
    var j = Int(block_idx.x)
    var h = Int(block_idx.y)
    var H = Int(hkv)
    var d = Int(thread_idx.x)
    var s = Int(job_seq[unsafe_offset=j])
    var ps = Int(job_page[unsafe_offset=j]) * P
    var q0 = Int(query_start_loc[unsafe_offset=s])
    var total = Int(seq_lens[unsafe_offset=s])
    var ctx = total - (Int(query_start_loc[unsafe_offset=s + 1]) - q0)
    var tb = (Int(tail_idx[unsafe_offset=s]) * H + h) * P * D
    var ks = Int(k_stride)
    var vs = Int(v_stride)
    if mode == 0:
        var rec = _rec(cache, Int(block_table[unsafe_offset=s * Int(max_pages) + ps // P]), h, block_stride, head_stride)
        for t in range(max(ps, ctx) + d // WARP, min(ps + P, total), D // WARP):
            _quantize_v_row[D, BITS, IN](v_new.unsafe_offset((q0 + t - ctx) * vs + h * D), rec, t - ps, d % WARP)
        if ps + P <= total:
            var split = max(0, min(P, ctx - ps))
            _quantize_k_page[D, BITS, IN](tail_k.unsafe_offset(tb), D,
                                          k_new.unsafe_offset((q0 + ps + split - ctx) * ks + h * D), ks, split, rec)
    else:
        if ps + P <= total:
            return
        for t in range(max(ps, ctx), total):
            tail_k[unsafe_offset=tb + (t - ps) * D + d] = k_new[unsafe_offset=(q0 + t - ctx) * ks + h * D + d]


# ------------------------------------------------------------------ unpack (prefill with history)
@always_inline
def _codes_row[D: Int, BITS: Int](row: I8P, lane: Int) -> SIMD[DType.float32, D // WARP]:
    """Codes of one page row for the lane's dims lane + 32*i, as fp32 (exact). int4: byte j holds dim j
    (low nibble) and dim j + D/2 (high), so a lane's first R/2 bytes give all R codes."""
    comptime R = D // WARP
    var c = SIMD[DType.float32, R]()
    comptime if BITS == 8:
        comptime for i in range(R):
            c[i] = Float32(row[unsafe_offset=lane + i * WARP])
    else:
        comptime for i in range(R // 2):
            var byte = row[unsafe_offset=lane + i * WARP]
            c[i] = Float32((byte << 4) >> 4)
            c[i + R // 2] = Float32(byte >> 4)
    return c


@always_inline
def _store_row[D: Int, IN: DType](dst: Pointer[Scalar[IN], MutAnyOrigin], x: SIMD[DType.float32, D // WARP], lane: Int):
    comptime for i in range(D // WARP):
        dst[unsafe_offset=lane + i * WARP] = x[i].cast[IN]()


def kush_unpack[D: Int, BITS: Int, IN: DType](
    cache: I8P, block_stride: Int32, head_stride: Int32, hkv: Int32,
    page_blk: I32P, page_dst: I32P, n_pages: Int32,
    tail_blk: I32P, tail_info: I32P, n_tail: Int32, tail_k: Pointer[Scalar[IN], MutAnyOrigin],
    k_new: Pointer[Scalar[IN], MutAnyOrigin], v_new: Pointer[Scalar[IN], MutAnyOrigin], k_stride: Int32,
    v_stride: Int32, new_dst: I32P, n_new: Int32,
    kp: Pointer[Scalar[IN], MutAnyOrigin], vp: Pointer[Scalar[IN], MutAnyOrigin], rot: Int32,
):
    """Builds a packed-prefill group's keys and values in kp / vp ([rows][Hkv][D], IN) in one launch.
    grid (n_pages + n_tail + ceil(n_new / 32), Hkv), block 128 (4 warps; a warp does one token row at a time,
    lane holding dims lane + 32*i). Block x:
      < n_pages            full history page: record of block page_blk[x] -> rows page_dst[x] + t, t < 32
      < n_pages + n_tail   partial page i: tail_info[3i..3i+2] = (first row, tokens n, tail slot); K rows from
                           the exact tail (slot), V rows from the record of block tail_blk[i]
      else                 32 rows of the new chunk: row j of k_new / v_new (row strides) -> row new_dst[j]
    Dequantization is fma(code, scale, bias) in fp32 (scales and biases are fp16, exact in fp32), rounded once
    to IN. rot != 0: history (pages and tail) is rotated back with the canonical Walsh-Hadamard transform
    before rounding; new rows are copied as they are. Same arithmetic as the CPU _unpack: bit-identical."""
    comptime R = D // WARP
    comptime DC = row_bytes[D, BITS]()
    var b = Int(block_idx.x)
    var h = Int(block_idx.y)
    var H = Int(hkv)
    var w = Int(thread_idx.x) // WARP
    var lane = Int(thread_idx.x) % WARP
    var np = Int(n_pages)
    var nt = Int(n_tail)
    var rs = H * D                                         # row stride of kp / vp
    if b < np + nt:
        var page = b < np
        var i = b - np
        var rec = _rec(cache, Int(page_blk[unsafe_offset=b]) if page else Int(tail_blk[unsafe_offset=i]), h,
                       block_stride, head_stride)
        var dst0 = Int(page_dst[unsafe_offset=b]) if page else Int(tail_info[unsafe_offset=3 * i])
        var n = P if page else Int(tail_info[unsafe_offset=3 * i + 1])
        var vsp = rec.unsafe_offset(off_vs[D, BITS]()).unsafe_bitcast[Float16]()
        var vbp = rec.unsafe_offset(off_vb[D, BITS]()).unsafe_bitcast[Float16]()
        var ks = SIMD[DType.float32, R]()
        var kb = SIMD[DType.float32, R]()
        if page:                                           # per-channel K scale / bias: once per block
            var ksp = rec.unsafe_offset(off_ks[D, BITS]()).unsafe_bitcast[Float16]()
            var kbp = rec.unsafe_offset(off_kb[D, BITS]()).unsafe_bitcast[Float16]()
            comptime for j in range(R):
                ks[j] = Float32(ksp[unsafe_offset=lane + j * WARP])
                kb[j] = Float32(kbp[unsafe_offset=lane + j * WARP])
        var tk = 0
        if not page:
            tk = (Int(tail_info[unsafe_offset=3 * i + 2]) * H + h) * P * D
        for t in range(w, n, 4):
            var kx: SIMD[DType.float32, R]
            if page:
                kx = _codes_row[D, BITS](rec.unsafe_offset(t * DC), lane).fma(ks, kb)
            else:
                kx = SIMD[DType.float32, R]()
                comptime for j in range(R):
                    kx[j] = Float32(tail_k[unsafe_offset=tk + t * D + lane + j * WARP])
            var vx = _codes_row[D, BITS](rec.unsafe_offset(off_vq[D, BITS]() + t * DC), lane).fma(
                SIMD[DType.float32, R](Float32(vsp[unsafe_offset=t])), SIMD[DType.float32, R](Float32(vbp[unsafe_offset=t])))
            if rot != 0:
                kx = _fwht_warp[D](kx, lane)
                vx = _fwht_warp[D](vx, lane)
            var o = (dst0 + t) * rs + h * D
            _store_row[D, IN](kp.unsafe_offset(o), kx, lane)
            _store_row[D, IN](vp.unsafe_offset(o), vx, lane)
    else:
        var j0 = (b - np - nt) * P
        for t in range(w, P, 4):
            var j = j0 + t
            if j < Int(n_new):
                var o = Int(new_dst[unsafe_offset=j]) * rs + h * D
                comptime for c in range(R):
                    kp[unsafe_offset=o + lane + c * WARP] = k_new[unsafe_offset=j * Int(k_stride) + h * D + lane + c * WARP]
                    vp[unsafe_offset=o + lane + c * WARP] = v_new[unsafe_offset=j * Int(v_stride) + h * D + lane + c * WARP]


# ------------------------------------------------------------------ shared decode helpers
@always_inline
def _byte_f32[k: Int](w: UInt32) -> Float32:
    """Unsigned byte k of w -> fp32 exactly: prmt puts it under exponent 0x4B (2^23 + u).
    A float with bits 0x4B000000 | u is exactly 2^23 + u (exponent 23, and u < 2^23 is exact in the
    mantissa), so one subtract of 2^23 turns the byte into the integer as a float without an
    int->float conversion. 4 bytes per word, one prmt each."""
    comptime sel = UInt32(k | (4 << 4) | (5 << 8) | (7 << 12))
    var bits = inlined_assembly["prmt.b32 $0, $1, $2, $3;", UInt32, constraints="=r,r,r,r"](
        w, UInt32(0x4B000000), sel)
    return bitcast[DType.float32](bits) - Float32(8388608.0)


@always_inline
def _w2f(w: UInt32) -> SIMD[DType.float32, 4]:
    var r = SIMD[DType.float32, 4]()
    comptime for k in range(4):
        r[k] = _byte_f32[k](w)
    return r


@always_inline
def _w_to_h4[HALF: Int](w: UInt32) -> SIMD[DType.float16, 4]:
    """4 unsigned code bytes (u = c + HALF) -> signed codes c as fp16, exact (2 values / op).
    The same trick as _byte_f32 in fp16: prmt moves a byte under fp16's exponent for 1024 (0x64),
    then subtracting 1024 + HALF undoes both the exponent bias and the code offset in one add."""
    var lo = inlined_assembly["prmt.b32 $0, $1, $2, $3;", UInt32, constraints="=r,r,r,r"](w, H64, UInt32(0x4140))
    var hi = inlined_assembly["prmt.b32 $0, $1, $2, $3;", UInt32, constraints="=r,r,r,r"](w, H64, UInt32(0x4342))
    var off = SIMD[DType.float16, 2](Float16(1024 + HALF))
    return (bitcast[DType.float16, 2](lo) - off).join(bitcast[DType.float16, 2](hi) - off)


@always_inline
def _nib[SHIFT: Int](w: UInt32) -> UInt32:
    """Nibbles at bit SHIFT of each byte, two's complement -> unsigned (c + 8): ((w >> SHIFT) & 0x0F0F0F0F) ^
    0x08080808 as one LOP3 (lut 0x6A = (a & b) ^ c); written out, the compiler emits an AND and an XOR."""
    var x = w >> UInt32(SHIFT) if SHIFT else w
    return inlined_assembly["lop3.b32 $0, $1, $2, $3, 0x6A;", UInt32, constraints="=r,r,r,r"](x, NIB, XOR4)


@always_inline
def _split_range(full: Int, n_splits: Int, z: Int) -> Tuple[Int, Int]:
    """[start, end) in pages for split z: ceil(full / n_splits) pages each. A split can come out
    empty (more splits than pages), which the callers handle as a no-op."""
    var pps = (full + n_splits - 1) // n_splits
    var start = min(full, z * pps)
    return (start, min(full, start + pps))


@always_inline
def _load_q[D: Int, G: Int, NW: Int, IN: DType](
    q: Pointer[Scalar[IN], MutAnyOrigin], qbase: Int, sm: Float32, qsm: SF32P,
):
    """q * softmax scale into shared memory, once per block; the per-page K scale is then folded into
    this shared copy instead of re-scaling q from global memory for every page."""
    for i in range(Int(thread_idx.x), G * D, NW * WARP):
        qsm[unsafe_offset=i] = Float32(q[unsafe_offset=qbase + i]) * sm
    barrier()


@always_inline
def _merge_store[D: Int, G: Int, NW: Int, IN: DType](
    c_acc: SF32P, c_m: SF32P, c_l: SF32P,
    o_ptr: Pointer[Scalar[IN], MutAnyOrigin], part_ml: F32P, part_o: F32P,
    qbase: Int, pbase: Int, n_splits: Int,
):
    """Merge the NW warps' (m, l, unnormalized O) of this block (after a barrier). One split:
    write the normalized output. Else write (M, L, O) of this split for kush_combine."""
    for idx in range(Int(thread_idx.x), G * D, NW * WARP):
        var g = idx // D
        var M = Float32.MIN
        comptime for w in range(NW):
            M = max(M, c_m[unsafe_offset=w * G + g])
        var L = Float32(0)
        var o = Float32(0)
        if M > Float32.MIN:                      # else: no tokens in this split
            comptime for w in range(NW):
                var mw = c_m[unsafe_offset=w * G + g]
                if mw > Float32.MIN:
                    var f = exp(mw - M)
                    L += c_l[unsafe_offset=w * G + g] * f
                    o += f * c_acc[unsafe_offset=(w * G + g) * D + idx % D]
        if n_splits == 1:
            o_ptr[unsafe_offset=qbase + idx] = (o / L if L > 0 else Float32(0)).cast[IN]()
        else:
            part_o[unsafe_offset=(pbase + g) * D + idx % D] = o
            if idx % D == 0:
                part_ml[unsafe_offset=(pbase + g) * 2] = M
                part_ml[unsafe_offset=(pbase + g) * 2 + 1] = L


def kush_combine[D: Int, G: Int, IN: DType](
    o_ptr: Pointer[Scalar[IN], MutAnyOrigin], part_ml: F32P, part_o: F32P, n_splits: Int32, hkv: Int32,
):
    """grid (S, Hkv), block 256: merge split-K partials into the normalized output."""
    var s = Int(block_idx.x)
    var h = Int(block_idx.y)
    var H = Int(hkv)
    var NS = Int(n_splits)
    for idx in range(Int(thread_idx.x), G * D, 256):
        var g = idx // D
        var M = Float32.MIN
        for z in range(NS):
            M = max(M, part_ml[unsafe_offset=(((s * H + h) * NS + z) * G + g) * 2])
        var L = Float32(0)
        var o = Float32(0)
        for z in range(NS):
            var pb = ((s * H + h) * NS + z) * G + g
            var mz = part_ml[unsafe_offset=pb * 2]
            if mz > Float32.MIN:
                var f = exp(mz - M)
                L += part_ml[unsafe_offset=pb * 2 + 1] * f
                o += f * part_o[unsafe_offset=pb * D + idx % D]
        o_ptr[unsafe_offset=(s * H + h) * G * D + idx] = (o / L if L > 0 else Float32(0)).cast[IN]()


# ------------------------------------------------------------------ tensor-core decode
#   QK:  S[16 x 8 tok] += Q'[16 x 16] . K^T[16 x 8 tok]  (rows = q heads; rows >= G are zero)
#        mma.m16n8k16 f16 x f16 -> f32, Q' = fp16(q * sm * k_scale) per page. The contraction
#        over D is permuted so each lane's B fragment is one 32-bit word of a K row: lane
#        (g, t) owns physical dims c*64 + t*16 + j*4 + {0..3} (int4: low/high nibble tiles).
#   PV:  O^T[16 dims x 8 heads] += V^T[16 x 16 tok] . P^T[16 tok x 8 heads]; P^T comes straight
#        from the S accumulators, V^T via ldmatrix.trans from a padded fp16 page in shared memory.

@always_inline
def _kphys[D: Int, BITS: Int, kt: Int](tid: Int) -> Int:
    """Physical dim lane (g, tid) covers for k-tile kt: dims c*64 + tid*16 + j*4 + {0..3}. Choosing
    this permutation is what lets each lane's B fragment be one aligned 32-bit word of a K row;
    int4 needs a separate permutation for the low-nibble and the high-nibble tiles."""
    comptime HT = D // 32
    comptime if BITS == 8 or kt < HT:
        return (kt // 4) * 64 + tid * 16 + (kt % 4) * 4
    else:
        return D // 2 + ((kt - HT) // 4) * 64 + tid * 16 + ((kt - HT) % 4) * 4


@always_inline
def _load_page_tc[D: Int, BITS: Int](
    rec: I8P, lane: Int,
    kw: Pointer[UInt32, MutUntrackedOrigin], vw: Pointer[UInt32, MutUntrackedOrigin],
):
    """K rows nt*8 + g (nt = 0..3): 16 bytes per chunk at c*64 + t*16 -> kw[(nt*NC + c)*4 + j].
    V: 512-byte slabs, lane L takes bytes 16L.. of slab j -> vw[j*4 ..] (token row j*RPI + L // CH, 16-byte chunk
    L % CH, CH = DC // 16 chunks per row): every load instruction reads 512 contiguous bytes. (One row per lane, the
    earlier layout, read 16 bytes per 64-byte row per instruction: 2x the sectors, ncu A100.) All loads 16-byte,
    issued together."""
    comptime DC = row_bytes[D, BITS]()
    comptime NC = DC // 64
    var g = lane // 4
    var tid = lane % 4
    comptime for nt in range(4):
        comptime for c in range(NC):
            var w4 = rec.unsafe_offset((nt * 8 + g) * DC + c * 64 + tid * 16).unsafe_bitcast[UInt32]().unsafe_load[width=4, alignment=16]()
            comptime for j in range(4):
                kw[unsafe_offset=(nt * NC + c) * 4 + j] = w4[j]
    var vp = rec.unsafe_offset(off_vq[D, BITS]() + lane * 16).unsafe_bitcast[UInt32]()
    comptime for j in range(DC // 16):
        var v4 = vp.unsafe_load[width=4, alignment=16](j * 128)
        comptime for k in range(4):
            vw[unsafe_offset=j * 4 + k] = v4[k]


@always_inline
def _quad_sum(x: Float32) -> Float32:
    """Sum over the 4 lanes of a quad (lane // 4 = q head g, lane % 4 = tid). Those 4 lanes split
    the page's tokens and the contraction dims between them, so a quad reduction is a full reduce
    for one q head: 2 shuffle_xor steps, no shared memory."""
    var y = x + warp.shuffle_xor(x, UInt32(1))
    return y + warp.shuffle_xor(y, UInt32(2))


@always_inline
def _quad_max(x: Float32) -> Float32:
    var y = max(x, warp.shuffle_xor(x, UInt32(1)))
    return max(y, warp.shuffle_xor(y, UInt32(2)))


@always_inline
def _rescale_ot[MT: Int](o: LF32P, ca: Float32, cb: Float32):
    """Rescale the O^T accumulators after a softmax correction. Each lane holds two q heads (columns
    0, 2 and 1, 3 of the 4-wide fragment), so the two corrections are applied separately."""
    comptime for mt in range(MT):
        o[unsafe_offset=mt * 4 + 0] *= ca
        o[unsafe_offset=mt * 4 + 1] *= cb
        o[unsafe_offset=mt * 4 + 2] *= ca
        o[unsafe_offset=mt * 4 + 3] *= cb


def kush_decode_tc[D: Int, BITS: Int, G: Int, NW: Int, IN: DType](
    q: Pointer[Scalar[IN], MutAnyOrigin], o_ptr: Pointer[Scalar[IN], MutAnyOrigin],
    cache: I8P, block_stride: Int32, head_stride: Int32,
    tail_k: Pointer[Scalar[IN], MutAnyOrigin], tail_idx: I32P,
    block_table: I32P, seq_lens: I32P, max_pages: Int32, hkv: Int32, sm: Float32,
    n_splits: Int32, part_ml: F32P, part_o: F32P,
):
    """grid (S, Hkv, n_splits), block NW warps. Split z owns a contiguous page range; warps
    take pages w, w+NW, ... (int4 prefetches the next page into registers)."""
    comptime assert D % 128 == 0, "tensor-core path needs D % 128 == 0 (use kush_decode)"
    comptime assert G <= 8, "tensor-core path supports up to 8 q heads per kv head"
    comptime DC = row_bytes[D, BITS]()
    comptime NC = DC // 64
    comptime NKW = 4 * NC * 4
    comptime NVW = DC // 4
    comptime KT = D // 16
    comptime MT = D // 16
    comptime HT = D // 32
    comptime VS = D + 8                           # padded fp16 row stride: conflict-free ldmatrix
    comptime HALF = 1 << (BITS - 1)
    comptime PREFETCH = BITS == 4
    comptime STAGE = P * VS
    comptime TA = 4 * size_of[Scalar[IN]]()          # alignment of a 4-wide tail load
    comptime CH = DC // 16                            # 16-byte V chunks per token row
    comptime RPI = WARP // CH                         # V rows per 512-byte load slab
    comptime assert NW * STAGE * 2 >= NW * G * D * 4, "merge scratch must fit in the V staging buffer"
    var s = Int(block_idx.x)
    var h = Int(block_idx.y)
    var z = Int(block_idx.z)
    var NS = Int(n_splits)
    var H = Int(hkv)
    var w_id = Int(thread_idx.x) // WARP
    var lane = Int(thread_idx.x) % WARP
    var g = lane // 4
    var tid = lane % 4
    var hv = g < G
    var gq = min(g, G - 1)
    var vrow = lane // CH
    var vch = lane % CH
    var n = Int(seq_lens[unsafe_offset=s])
    var tail_n = n % P
    var rng = _split_range(n // P, NS, z)
    var bt = block_table.unsafe_offset(s * Int(max_pages))

    var qsm = stack_allocation[G * D, Float32, address_space=AddressSpace.SHARED, alignment=16]()
    var qh = stack_allocation[G * D, Float16, address_space=AddressSpace.SHARED, alignment=16]()
    var stage_all = stack_allocation[NW * STAGE, Float16, address_space=AddressSpace.SHARED, alignment=16]()
    var ewt = stack_allocation[G * P, Float32, address_space=AddressSpace.SHARED, alignment=16]()
    var c_m = stack_allocation[NW * G, Float32, address_space=AddressSpace.SHARED]()
    var c_l = stack_allocation[NW * G, Float32, address_space=AddressSpace.SHARED]()
    var qbase = (s * H + h) * G * D
    _load_q[D, G, NW, IN](q, qbase, sm, qsm)
    # fp16 copy of q * sm for the A fragments (step 1 multiplies by k_scale in fp16: 2 HMUL2 per k-tile instead of
    # 4 fp16->fp32 conversions, 4 FMUL and 2 packs). The K-bias dot stays in fp32 on qsm.
    for i in range(Int(thread_idx.x), G * D, NW * WARP):
        qh[unsafe_offset=i] = qsm[unsafe_offset=i].cast[DType.float16]()
    barrier()
    var vst = stage_all.unsafe_offset(w_id * STAGE)

    var o = stack_allocation[MT * 4, Float32]()        # O^T accumulators (dim, head) per lane
    comptime for i in range(MT * 4):
        o[unsafe_offset=i] = 0
    var m = Float32.MIN                                 # softmax state of head g (quad-replicated)
    var l = Float32(0)
    var vbacc = Float32(0)
    var afr = stack_allocation[KT * 8, Float16]()
    var sacc = stack_allocation[8, Float32]()
    var pfr = stack_allocation[8, Float16]()
    var kw = stack_allocation[NKW, UInt32]()
    var vw = stack_allocation[NVW, UInt32]()
    var kn = stack_allocation[NKW, UInt32]()
    var vn = stack_allocation[NVW, UInt32]()

    var lp = rng[0] + w_id
    comptime if PREFETCH:
        if lp < rng[1]:
            _load_page_tc[D, BITS](_rec(cache, Int(bt[unsafe_offset=lp]), h, block_stride, head_stride), lane, kn, vn)
    while lp < rng[1]:
        var nxt = lp + NW
        var rec = _rec(cache, Int(bt[unsafe_offset=lp]), h, block_stride, head_stride)
        var ksp = rec.unsafe_offset(off_ks[D, BITS]()).unsafe_bitcast[Float16]()
        var kbp = rec.unsafe_offset(off_kb[D, BITS]()).unsafe_bitcast[Float16]()
        var vsp = rec.unsafe_offset(off_vs[D, BITS]()).unsafe_bitcast[Float16]()
        var vbp = rec.unsafe_offset(off_vb[D, BITS]()).unsafe_bitcast[Float16]()
        comptime if PREFETCH:
            comptime for i in range(NKW):
                kw[unsafe_offset=i] = kn[unsafe_offset=i]
            comptime for i in range(NVW):
                vw[unsafe_offset=i] = vn[unsafe_offset=i]
            if nxt < rng[1]:
                _load_page_tc[D, BITS](_rec(cache, Int(bt[unsafe_offset=nxt]), h, block_stride, head_stride), lane, kn, vn)
        else:
            _load_page_tc[D, BITS](rec, lane, kw, vw)

        # 1. A fragments Q' = fp16(q * sm) * k_scale in fp16 at this lane's dims; K bias dot in fp32. The 4 k-tiles of a
        #    group cover 16 contiguous dims of this lane: one 32-byte load each for k_scale and k_bias
        var kbs4 = SIMD[DType.float32, 4](0)
        comptime for kg in range(KT // 4):
            var base0 = _kphys[D, BITS, kg * 4](tid)
            var ks16 = ksp.unsafe_load[width=16, alignment=16](base0)
            var kb16 = kbp.unsafe_load[width=16, alignment=16](base0)
            comptime for j in range(4):
                comptime kt = kg * 4 + j
                comptime assert _kphys[D, BITS, kt](0) == _kphys[D, BITS, kg * 4](0) + 4 * j, "k-tile group not contiguous"
                var kb4 = kb16.slice[4, offset=4 * j]().cast[DType.float32]()
                var q4 = qsm.unsafe_load[width=4, alignment=16](gq * D + base0 + 4 * j) if hv else SIMD[DType.float32, 4](0)
                var qh4 = qh.unsafe_load[width=4, alignment=8](gq * D + base0 + 4 * j) if hv else SIMD[DType.float16, 4](0)
                var a4 = qh4 * ks16.slice[4, offset=4 * j]()
                afr.unsafe_store(kt * 8, SIMD[DType.float16, 8](a4[0], a4[1], 0, 0, a4[2], a4[3], 0, 0))
                kbs4 = q4.fma(kb4, kbs4)
        var kbs = _quad_sum(kbs4.reduce_add())

        # 2. stage the V chunks this lane loaded (_load_page_tc) as exact fp16 codes: slab j, row j*RPI + vrow,
        #    words vch*4 .. vch*4+3 of that row
        comptime for j in range(DC // 16):
            var row = (j * RPI + vrow) * VS
            comptime for k in range(0, 4, 2):
                var w0 = vw[unsafe_offset=j * 4 + k]
                var w1 = vw[unsafe_offset=j * 4 + k + 1]
                var col = 4 * (vch * 4 + k)
                comptime if BITS == 8:
                    vst.unsafe_store[alignment=16](row + col, _w_to_h4[HALF](w0 ^ XOR8).join(_w_to_h4[HALF](w1 ^ XOR8)))
                else:
                    vst.unsafe_store[alignment=16](row + col, _w_to_h4[HALF](_nib[0](w0)).join(_w_to_h4[HALF](_nib[0](w1))))
                    vst.unsafe_store[alignment=16](row + D // 2 + col,
                        _w_to_h4[HALF](_nib[4](w0)).join(_w_to_h4[HALF](_nib[4](w1))))

        # 3. QK^T: 4 token tiles x KT k-tiles
        comptime for nt in range(4):
            var acc = SIMD[DType.float32, 4](0)
            comptime for kt in range(KT):
                var bfr: SIMD[DType.float16, 4]
                comptime if BITS == 8:
                    bfr = _w_to_h4[HALF](kw[unsafe_offset=(nt * NC + kt // 4) * 4 + kt % 4] ^ XOR8)
                else:
                    comptime kk = kt if kt < HT else kt - HT
                    var wv = kw[unsafe_offset=(nt * NC + kk // 4) * 4 + kk % 4]
                    comptime if kt < HT:
                        bfr = _w_to_h4[HALF](_nib[0](wv))
                    else:
                        bfr = _w_to_h4[HALF](_nib[4](wv))
                mma(acc, afr.unsafe_load[width=8](kt * 8), bfr, acc)
            sacc[unsafe_offset=nt * 2 + 0] = acc[0] + kbs if hv else Float32(0)
            sacc[unsafe_offset=nt * 2 + 1] = acc[1] + kbs if hv else Float32(0)

        # 4. page-granular online softmax, head g, tokens nt*8 + tid*2 + e
        var rmax = Float32.MIN
        comptime for i in range(8):
            rmax = max(rmax, sacc[unsafe_offset=i])
        var mn = max(m, _quad_max(rmax))
        var corr = exp(m - mn)
        var rsum = Float32(0)
        var rvb = Float32(0)
        comptime for nt in range(4):
            var vs2 = vsp.unsafe_load[width=2, alignment=4](nt * 8 + tid * 2).cast[DType.float32]()
            var vb2 = vbp.unsafe_load[width=2, alignment=4](nt * 8 + tid * 2).cast[DType.float32]()
            comptime for e in range(2):
                var ev = exp(sacc[unsafe_offset=nt * 2 + e] - mn)
                rsum += ev
                rvb += ev * vb2[e]
                pfr[unsafe_offset=nt * 2 + e] = (ev * vs2[e]).cast[DType.float16]() if hv else Float16(0)
        l = l * corr + _quad_sum(rsum)
        vbacc = vbacc * corr + _quad_sum(rvb)
        m = mn
        var ca = warp.shuffle_idx(corr, UInt32(tid * 8))
        var cb = warp.shuffle_idx(corr, UInt32(tid * 8 + 4))
        if ca != 1 or cb != 1:                  # no running max moved (most pages of a long context): exact skip
            _rescale_ot[MT](o, ca, cb)
        syncwarp()

        # 5. PV: O^T[dims x heads] += V^T . P^T
        var r8 = lane % 8
        var q8 = lane // 8
        comptime for kt2 in range(2):
            var b4 = pfr.unsafe_load[width=4](kt2 * 4)
            var tok = kt2 * 16 + r8 + (8 if q8 >= 2 else 0)
            comptime for mt in range(MT):
                var a8 = ld_matrix[8, transpose=True](vst.unsafe_offset(tok * VS + mt * 16 + (8 if q8 % 2 == 1 else 0)))
                var c4 = o.unsafe_load[width=4](mt * 4)
                mma(c4, a8, b4, c4)
                o.unsafe_store(mt * 4, c4)
        syncwarp()
        lp = nxt

    # tail page (exact K, V rows from its record): last split, warp that would take the next page
    if tail_n > 0 and z == NS - 1 and w_id == max(0, rng[1] - rng[0]) % NW:
        var tk = tail_k.unsafe_offset((Int(tail_idx[unsafe_offset=s]) * H + h) * P * D)
        var trec = _rec(cache, Int(bt[unsafe_offset=n // P]), h, block_stride, head_stride)   # tail V rows
        var rmax = Float32.MIN
        comptime for j in range(8):
            var t = tid + 4 * j
            var x = Float32.MIN
            if hv and t < tail_n:
                var acc4 = SIMD[DType.float32, 4](0)
                for c in range(0, D, 4):
                    acc4 += qsm.unsafe_load[width=4, alignment=16](g * D + c) * tk.unsafe_load[width=4, alignment=TA](t * D + c).cast[DType.float32]()
                x = acc4.reduce_add()
            sacc[unsafe_offset=j] = x
            rmax = max(rmax, x)
        var mn = max(m, _quad_max(rmax))
        var corr = exp(m - mn) if mn > Float32.MIN else Float32(1)
        var rsum = Float32(0)
        comptime for j in range(8):
            var t = tid + 4 * j
            var ev = exp(sacc[unsafe_offset=j] - mn) if (hv and t < tail_n) else Float32(0)
            rsum += ev
            if hv:
                ewt[unsafe_offset=g * P + t] = ev
        l = l * corr + _quad_sum(rsum)
        vbacc = vbacc * corr
        m = mn
        _rescale_ot[MT](o, warp.shuffle_idx(corr, UInt32(tid * 8)), warp.shuffle_idx(corr, UInt32(tid * 8 + 4)))
        syncwarp()
        for t in range(tail_n):
            var wa = ewt[unsafe_offset=(tid * 2) * P + t] if tid * 2 < G else Float32(0)
            var wb = ewt[unsafe_offset=(tid * 2 + 1) * P + t] if tid * 2 + 1 < G else Float32(0)
            comptime for mt in range(MT):
                var v0 = _vdeq[D, BITS](trec, t, mt * 16 + g)
                var v1 = _vdeq[D, BITS](trec, t, mt * 16 + g + 8)
                o[unsafe_offset=mt * 4 + 0] += wa * v0
                o[unsafe_offset=mt * 4 + 1] += wb * v0
                o[unsafe_offset=mt * 4 + 2] += wa * v1
                o[unsafe_offset=mt * 4 + 3] += wb * v1

    # merge warps; scratch aliases the V staging buffers
    barrier()
    var c_acc = stage_all.unsafe_bitcast[Float32]()
    var vba = warp.shuffle_idx(vbacc, UInt32(tid * 8))
    var vbb = warp.shuffle_idx(vbacc, UInt32(tid * 8 + 4))
    comptime for mt in range(MT):
        var d0 = mt * 16 + g
        if tid * 2 < G:
            c_acc[unsafe_offset=(w_id * G + tid * 2) * D + d0] = o[unsafe_offset=mt * 4 + 0] + vba
            c_acc[unsafe_offset=(w_id * G + tid * 2) * D + d0 + 8] = o[unsafe_offset=mt * 4 + 2] + vba
        if tid * 2 + 1 < G:
            c_acc[unsafe_offset=(w_id * G + tid * 2 + 1) * D + d0] = o[unsafe_offset=mt * 4 + 1] + vbb
            c_acc[unsafe_offset=(w_id * G + tid * 2 + 1) * D + d0 + 8] = o[unsafe_offset=mt * 4 + 3] + vbb
    if tid == 0 and hv:
        c_m[unsafe_offset=w_id * G + g] = m
        c_l[unsafe_offset=w_id * G + g] = l
    barrier()
    _merge_store[D, G, NW, IN](c_acc, c_m, c_l, o_ptr, part_ml, part_o, qbase, ((s * H + h) * NS + z) * G, NS)


# ------------------------------------------------------------------ CUDA-core decode
# Any D in {64, 128, 256}, any G, sm_75+. Lane t loads K and V row t of a page with
# 16-byte loads (int4 also prefetches the next page). K is consumed from registers
# (lane = token); V rows go through a padded per-warp shared buffer read column-wise
# (lane = dims); softmax weights are broadcast from shared memory.

@always_inline
def _lane_dim[D: Int, BITS: Int](lane: Int, i: Int) -> Int:
    """Output dim owned by `lane` in slot i (int4: matching low/high-nibble dims)."""
    comptime R = D // WARP
    comptime if BITS == 8:
        return lane * R + i
    else:
        comptime HR = R // 2
        return lane * HR + i if i < HR else D // 2 + lane * HR + (i - HR)


@always_inline
def _load_rows[NWD: Int](krow: I8P, vrow: I8P,
                         dk: Pointer[UInt32, MutUntrackedOrigin], dv: Pointer[UInt32, MutUntrackedOrigin]):
    var kp = krow.unsafe_bitcast[UInt32]()
    var vp = vrow.unsafe_bitcast[UInt32]()
    comptime for c4 in range(0, NWD, 4):
        var a4 = kp.unsafe_load[width=4, alignment=16](c4)
        var b4 = vp.unsafe_load[width=4, alignment=16](c4)
        comptime for j in range(4):
            dk[unsafe_offset=c4 + j] = a4[j]
            dv[unsafe_offset=c4 + j] = b4[j]


def kush_decode[D: Int, BITS: Int, G: Int, NW: Int, IN: DType](
    q: Pointer[Scalar[IN], MutAnyOrigin], o_ptr: Pointer[Scalar[IN], MutAnyOrigin],
    cache: I8P, block_stride: Int32, head_stride: Int32,
    tail_k: Pointer[Scalar[IN], MutAnyOrigin], tail_idx: I32P,
    block_table: I32P, seq_lens: I32P, max_pages: Int32, hkv: Int32, sm: Float32,
    n_splits: Int32, part_ml: F32P, part_o: F32P,
):
    comptime assert D == 64 or D == 128 or D == 256, "D must be 64, 128 or 256"
    comptime R = D // WARP
    comptime HR = R // 2
    comptime DC = row_bytes[D, BITS]()
    comptime NWD = DC // 4
    comptime SROW = NWD + 4                   # padded shared row stride (words)
    comptime STAGE = P * SROW
    comptime PREFETCH = BITS == 4
    comptime HALF = Float32(1 << (BITS - 1))
    comptime SCR = max(NW * STAGE, NW * G * D)
    comptime TA = 4 * size_of[Scalar[IN]]()
    comptime VQ = off_vq[D, BITS]()
    var s = Int(block_idx.x)
    var h = Int(block_idx.y)
    var z = Int(block_idx.z)
    var NS = Int(n_splits)
    var H = Int(hkv)
    var w_id = Int(thread_idx.x) // WARP
    var lane = Int(thread_idx.x) % WARP
    var n = Int(seq_lens[unsafe_offset=s])
    var tail_n = n % P
    var rng = _split_range(n // P, NS, z)
    var bt = block_table.unsafe_offset(s * Int(max_pages))

    var qsm = stack_allocation[G * D, Float32, address_space=AddressSpace.SHARED, alignment=16]()
    var qs_all = stack_allocation[NW * G * D, Float32, address_space=AddressSpace.SHARED, alignment=16]()
    var scratch = stack_allocation[SCR, UInt32, address_space=AddressSpace.SHARED, alignment=16]()
    var wsh_all = stack_allocation[NW * P * G, Float32, address_space=AddressSpace.SHARED, alignment=16]()
    var c_m = stack_allocation[NW * G, Float32, address_space=AddressSpace.SHARED]()
    var c_l = stack_allocation[NW * G, Float32, address_space=AddressSpace.SHARED]()
    var qbase = (s * H + h) * G * D
    _load_q[D, G, NW, IN](q, qbase, sm, qsm)
    var qs = qs_all.unsafe_offset(w_id * G * D)
    var vst = scratch.unsafe_offset(w_id * STAGE)
    var wsh = wsh_all.unsafe_offset(w_id * P * G)

    var lane_dims = stack_allocation[R, Int]()
    comptime for i in range(R):
        lane_dims[unsafe_offset=i] = _lane_dim[D, BITS](lane, i)
    var m = stack_allocation[G, Float32]()
    var l = stack_allocation[G, Float32]()
    var vbacc = stack_allocation[G, Float32]()
    var acc = stack_allocation[G * R, Float32]()
    var kbs = stack_allocation[G, Float32]()
    var sc4 = stack_allocation[G * 4, Float32]()
    var kr = stack_allocation[NWD, UInt32]()
    var vr = stack_allocation[NWD, UInt32]()
    var kn = stack_allocation[NWD, UInt32]()
    var vn = stack_allocation[NWD, UInt32]()
    comptime for g in range(G):
        m[unsafe_offset=g] = Float32.MIN
        l[unsafe_offset=g] = 0
        vbacc[unsafe_offset=g] = 0
        comptime for i in range(R):
            acc[unsafe_offset=g * R + i] = 0

    var lp = rng[0] + w_id
    comptime if PREFETCH:
        if lp < rng[1]:
            var r0 = _rec(cache, Int(bt[unsafe_offset=lp]), h, block_stride, head_stride)
            _load_rows[NWD](r0.unsafe_offset(lane * DC), r0.unsafe_offset(VQ + lane * DC), kn, vn)
    while lp < rng[1]:
        var nxt = lp + NW
        var rec = _rec(cache, Int(bt[unsafe_offset=lp]), h, block_stride, head_stride)
        var ksp = rec.unsafe_offset(off_ks[D, BITS]()).unsafe_bitcast[Float16]()
        var kbp = rec.unsafe_offset(off_kb[D, BITS]()).unsafe_bitcast[Float16]()
        var vsp = rec.unsafe_offset(off_vs[D, BITS]()).unsafe_bitcast[Float16]()
        var vbp = rec.unsafe_offset(off_vb[D, BITS]()).unsafe_bitcast[Float16]()
        comptime if PREFETCH:
            comptime for i in range(NWD):
                kr[unsafe_offset=i] = kn[unsafe_offset=i]
                vr[unsafe_offset=i] = vn[unsafe_offset=i]
            if nxt < rng[1]:
                var rn = _rec(cache, Int(bt[unsafe_offset=nxt]), h, block_stride, head_stride)
                _load_rows[NWD](rn.unsafe_offset(lane * DC), rn.unsafe_offset(VQ + lane * DC), kn, vn)
        else:
            _load_rows[NWD](rec.unsafe_offset(lane * DC), rec.unsafe_offset(VQ + lane * DC), kr, vr)
        # 1. fold K scale into q; K bias (minus HALF*scale for unsigned codes) -> 1 const per head
        comptime for g in range(G):
            kbs[unsafe_offset=g] = 0
        comptime for i in range(R):
            var d = lane_dims[unsafe_offset=i]
            var ksd = Float32(ksp[unsafe_offset=d])
            var kbd = Float32(kbp[unsafe_offset=d]) - HALF * ksd
            comptime for g in range(G):
                var qv = qsm[unsafe_offset=g * D + d]
                qs[unsafe_offset=g * D + d] = qv * ksd
                kbs[unsafe_offset=g] += qv * kbd
        var vs_t = Float32(vsp[unsafe_offset=lane])
        var vb_t = Float32(vbp[unsafe_offset=lane]) - HALF * vs_t
        comptime for c4 in range(0, NWD, 4):
            vst.unsafe_store[alignment=16](lane * SROW + c4, SIMD[DType.uint32, 4](
                vr[unsafe_offset=c4], vr[unsafe_offset=c4 + 1], vr[unsafe_offset=c4 + 2], vr[unsafe_offset=c4 + 3]))
        comptime for g in range(G):
            kbs[unsafe_offset=g] = warp.sum(kbs[unsafe_offset=g])
        syncwarp()
        # 2. K phase: lane = token, 4-wide accumulators per head, one reduction per page
        comptime for i in range(G * 4):
            sc4[unsafe_offset=i] = 0
        comptime for wj in range(NWD):
            var wv = kr[unsafe_offset=wj]
            comptime if BITS == 8:
                var f = _w2f(wv ^ XOR8)
                comptime for g in range(G):
                    var q4 = qs.unsafe_load[width=4, alignment=16](g * D + wj * 4)
                    comptime for k in range(4):
                        sc4[unsafe_offset=g * 4 + k] += q4[k] * f[k]
            else:
                var flo = _w2f((wv & NIB) ^ XOR4)
                var fhi = _w2f(((wv >> 4) & NIB) ^ XOR4)
                comptime for g in range(G):
                    var qlo = qs.unsafe_load[width=4, alignment=16](g * D + wj * 4)
                    var qhi = qs.unsafe_load[width=4, alignment=16](g * D + D // 2 + wj * 4)
                    comptime for k in range(4):
                        sc4[unsafe_offset=g * 4 + k] += qlo[k] * flo[k] + qhi[k] * fhi[k]
        # 3. page-granular online softmax; weights to shared for broadcast
        comptime for g in range(G):
            var x = kbs[unsafe_offset=g] + (sc4[unsafe_offset=g * 4] + sc4[unsafe_offset=g * 4 + 1]) \
                + (sc4[unsafe_offset=g * 4 + 2] + sc4[unsafe_offset=g * 4 + 3])
            var mo = m[unsafe_offset=g]
            var mn = max(mo, warp.max(x))
            var corr = exp(mo - mn)
            var e = exp(x - mn)
            l[unsafe_offset=g] = l[unsafe_offset=g] * corr + warp.sum(e)
            vbacc[unsafe_offset=g] = vbacc[unsafe_offset=g] * corr + warp.sum(e * vb_t)
            m[unsafe_offset=g] = mn
            comptime for i in range(R):
                acc[unsafe_offset=g * R + i] *= corr
            wsh[unsafe_offset=lane * G + g] = e * vs_t
        syncwarp()
        # 4. V phase: lane = dims; codes converted once per token, shared by G heads
        comptime for t in range(P):
            var codes = SIMD[DType.float32, R]()
            comptime if BITS == 8:
                comptime for wi in range(0, R, 4):
                    var byte_off = lane * R + wi
                    var f = _w2f((vst[unsafe_offset=t * SROW + byte_off // 4] >> UInt32(8 * (byte_off % 4))) ^ XOR8)
                    comptime for k in range(min(4, R - wi)):
                        codes[wi + k] = f[k]
            else:
                var byte_off = lane * HR
                var wv = vst[unsafe_offset=t * SROW + byte_off // 4] >> UInt32(8 * (byte_off % 4))
                var flo = _w2f((wv & NIB) ^ XOR4)
                var fhi = _w2f(((wv >> 4) & NIB) ^ XOR4)
                comptime for k in range(HR):
                    codes[k] = flo[k]
                    codes[HR + k] = fhi[k]
            comptime for g in range(G):
                var wt = wsh[unsafe_offset=t * G + g]
                comptime for i in range(R):
                    acc[unsafe_offset=g * R + i] += wt * codes[i]
        syncwarp()
        lp = nxt

    # tail page (exact K, V rows from its record): last split, warp that would take the next page
    if tail_n > 0 and z == NS - 1 and w_id == max(0, rng[1] - rng[0]) % NW:
        var tk = tail_k.unsafe_offset((Int(tail_idx[unsafe_offset=s]) * H + h) * P * D)
        var trec = _rec(cache, Int(bt[unsafe_offset=n // P]), h, block_stride, head_stride)   # tail V rows
        var valid = lane < tail_n
        comptime for g in range(G):
            var x = Float32.MIN
            if valid:
                var a4 = SIMD[DType.float32, 4](0)
                comptime for c in range(0, D, 4):
                    a4 += qsm.unsafe_load[width=4, alignment=16](g * D + c) * tk.unsafe_load[width=4, alignment=TA](lane * D + c).cast[DType.float32]()
                x = a4.reduce_add()
            var mo = m[unsafe_offset=g]
            var mn = max(mo, warp.max(x))
            var corr = exp(mo - mn)
            var e = exp(x - mn) if valid else Float32(0)
            l[unsafe_offset=g] = l[unsafe_offset=g] * corr + warp.sum(e)
            vbacc[unsafe_offset=g] = vbacc[unsafe_offset=g] * corr
            m[unsafe_offset=g] = mn
            comptime for i in range(R):
                acc[unsafe_offset=g * R + i] *= corr
            wsh[unsafe_offset=lane * G + g] = e
        syncwarp()
        for t in range(tail_n):
            comptime for g in range(G):
                var wt = wsh[unsafe_offset=t * G + g]
                comptime for i in range(R):
                    acc[unsafe_offset=g * R + i] += wt * _vdeq[D, BITS](trec, t, lane_dims[unsafe_offset=i])

    # merge warps; scratch aliases the V staging buffers
    barrier()
    var c_acc = scratch.unsafe_bitcast[Float32]()
    comptime for g in range(G):
        comptime for i in range(R):
            c_acc[unsafe_offset=(w_id * G + g) * D + lane_dims[unsafe_offset=i]] = acc[unsafe_offset=g * R + i] + vbacc[unsafe_offset=g]
        if lane == 0:
            c_m[unsafe_offset=w_id * G + g] = m[unsafe_offset=g]
            c_l[unsafe_offset=w_id * G + g] = l[unsafe_offset=g]
    barrier()
    _merge_store[D, G, NW, IN](c_acc, c_m, c_l, o_ptr, part_ml, part_o, qbase, ((s * H + h) * NS + z) * G, NS)

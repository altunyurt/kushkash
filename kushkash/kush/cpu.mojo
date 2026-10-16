# kushkash CPU kernels — same cache format and page-record layout (kush/quant.mojo) as
# the GPU kernels (kush/gpu.mojo), so pages are bit-identical across backends. Exported
# with the C ABI; the Python package calls them via ctypes with raw tensor addresses.
# Inputs, outputs and K tails are fp32 (the Python side converts).
#
# Every entry point takes the runtime head size D and BITS, dispatches to a comptime-
# specialized implementation (D in {64, 128, 256}, BITS in {4, 8}) and returns 0, or -1
# for an unsupported shape.

from std.math import exp
from std.memory import alloc, Layout, stack_allocation
from std.sys import simd_width_of
from max.algorithm import sync_parallelize
from kush.quant import PAGE, row_bytes, qparams, qcode, pack4, off_vq, off_ks, off_kb, off_vs, off_vb

comptime P = PAGE
comptime W = simd_width_of[DType.float32]()
comptime F32P = Pointer[Float32, MutUntrackedOrigin]
comptime I8P = Pointer[Int8, MutUntrackedOrigin]
comptime I32P = Pointer[Int32, MutUntrackedOrigin]
comptime V32 = SIMD[DType.float32, W]


@always_inline
def _p[T: AnyType](addr: Int64) -> Pointer[T, MutUntrackedOrigin]:
    return Pointer[T, MutUntrackedOrigin](unsafe_from_address=Int(addr))


def _buf(n: Int) -> F32P:
    return alloc(Layout[Float32](count=max(n, 1))).unsafe_leak()


@fieldwise_init
struct _Cache(ImplicitlyCopyable):
    """One layer's page records, K tails and per-sequence tables (layouts: kush/gpu.mojo)."""
    var cache: I8P
    var block_stride: Int
    var head_stride: Int
    var tail_k: F32P
    var tail_idx: I32P
    var block_table: I32P
    var max_pages: Int
    var hkv: Int

    @always_inline
    def rec(self, s: Int, lp: Int, h: Int) -> I8P:
        """Record of logical page lp of sequence s, kv head h."""
        var blk = Int(self.block_table[unsafe_offset=s * self.max_pages + lp])
        return self.cache.unsafe_offset(blk * self.block_stride + h * self.head_stride)

    @always_inline
    def tail(self, s: Int, h: Int) -> Int:
        """Element offset of the tail page of sequence s, kv head h."""
        return (Int(self.tail_idx[unsafe_offset=s]) * self.hkv + h) * P


@always_inline
def _cache(cache: Int64, bs: Int64, hs: Int64, tail_k: Int64, tail_idx: Int64,
           block_table: Int64, max_pages: Int64, hkv: Int64) -> _Cache:
    return _Cache(_p[Int8](cache), Int(bs), Int(hs), _p[Float32](tail_k), _p[Int32](tail_idx),
                  _p[Int32](block_table), Int(max_pages), Int(hkv))


# ------------------------------------------------------------------ write path
def _quantize_k_page[D: Int, BITS: Int](ka: F32P, sa: Int, kb_src: F32P, skb: Int, split: Int, rec: I8P):
    """K of one page (per-channel ranges), W channels per SIMD op. Token row t < split comes
    from ka + t*sa (the tail), the rest from kb_src + (t - split)*skb.
    Same arithmetic (kush/quant.mojo) as the GPU _quantize_k_page (kush/gpu.mojo)."""
    comptime DC = row_bytes[D, BITS]()
    comptime NCH = D if BITS == 8 else D // 2
    var kc = rec
    var ksp = rec.unsafe_offset(off_ks[D, BITS]()).unsafe_bitcast[Float16]()
    var kbp = rec.unsafe_offset(off_kb[D, BITS]()).unsafe_bitcast[Float16]()

    @always_inline
    def krow(t: Int) {var ka, var kb_src, var sa, var skb, var split} -> F32P:
        return ka.unsafe_offset(t * sa) if t < split else kb_src.unsafe_offset((t - split) * skb)

    var kbias = stack_allocation[D, Float32]()
    var kinv = stack_allocation[D, Float32]()
    for d in range(0, D, W):
        var mn = V32(Float32.MAX)
        var mx = V32(Float32.MIN)
        for t in range(P):
            var x = krow(t).unsafe_load[width=W](d)
            mn = min(mn, x)
            mx = max(mx, x)
        var kp = qparams[BITS, W](mn, mx)
        ksp.unsafe_store(d, kp[0].cast[DType.float16]())
        kbp.unsafe_store(d, kp[1].cast[DType.float16]())
        kbias.unsafe_store(d, kp[1])
        kinv.unsafe_store(d, kp[2])
    for t in range(P):
        var row = krow(t)
        for j in range(0, NCH, W):
            var c = qcode[BITS, W](row.unsafe_load[width=W](j), kbias.unsafe_load[width=W](j), kinv.unsafe_load[width=W](j))
            comptime if BITS == 8:
                kc.unsafe_store(t * DC + j, c)
            else:
                comptime H2 = D // 2
                kc.unsafe_store(t * DC + j, pack4[W](c, qcode[BITS, W](row.unsafe_load[width=W](j + H2),
                    kbias.unsafe_load[width=W](j + H2), kinv.unsafe_load[width=W](j + H2))))


def _quantize_v_row[D: Int, BITS: Int](row: F32P, rec: I8P, t: Int):
    """V token row -> row t of a record (per-token range: this is its final encoding)."""
    comptime DC = row_bytes[D, BITS]()
    comptime NCH = D if BITS == 8 else D // 2
    var vc = rec.unsafe_offset(off_vq[D, BITS]())
    var mn = V32(Float32.MAX)
    var mx = V32(Float32.MIN)
    for j in range(0, D, W):
        mn = min(mn, row.unsafe_load[width=W](j))
        mx = max(mx, row.unsafe_load[width=W](j))
    var vp = qparams[BITS](mn.reduce_min(), mx.reduce_max())
    rec.unsafe_offset(off_vs[D, BITS]()).unsafe_bitcast[Float16]()[unsafe_offset=t] = vp[0].cast[DType.float16]()
    rec.unsafe_offset(off_vb[D, BITS]()).unsafe_bitcast[Float16]()[unsafe_offset=t] = vp[1].cast[DType.float16]()
    var b = V32(vp[1])
    var inv = V32(vp[2])
    for j in range(0, NCH, W):
        var c = qcode[BITS, W](row.unsafe_load[width=W](j), b, inv)
        comptime if BITS == 8:
            vc.unsafe_store(t * DC + j, c)
        else:
            vc.unsafe_store(t * DC + j, pack4[W](c, qcode[BITS, W](row.unsafe_load[width=W](j + D // 2), b, inv)))


def _append[D: Int, BITS: Int](k_new: F32P, v_new: F32P, ks: Int, vs: Int, seq_lens: I32P, pos_offset: Int,
                               advance: Bool, cc: _Cache, S: Int):
    var H = cc.hkv
    for s in range(S):
        var pos = Int(seq_lens[unsafe_offset=s]) + pos_offset
        if pos < 0:
            continue
        var slot = pos % P
        for h in range(H):
            var dst = (cc.tail(s, h) + slot) * D
            for d in range(0, D, W):
                cc.tail_k.unsafe_store(dst + d, k_new.unsafe_load[width=W](s * ks + h * D + d))
            _quantize_v_row[D, BITS](v_new.unsafe_offset(s * vs + h * D), cc.rec(s, pos // P, h), slot)
            if slot == P - 1:
                var tk = cc.tail_k.unsafe_offset(cc.tail(s, h) * D)
                _quantize_k_page[D, BITS](tk, D, tk, D, P, cc.rec(s, pos // P, h))
        if advance:
            seq_lens[unsafe_offset=s] = Int32(pos + 1)


def _write[D: Int, BITS: Int](k_new: F32P, v_new: F32P, ks: Int, vs: Int, qsl: I32P, seq_lens: I32P,
                              job_seq: I32P, job_page: I32P, n_jobs: Int, mode: Int, cc: _Cache):
    """Same contract as the GPU kush_write (one call per mode)."""
    var H = cc.hkv
    for j in range(n_jobs):
        var s = Int(job_seq[unsafe_offset=j])
        var ps = Int(job_page[unsafe_offset=j]) * P
        var q0 = Int(qsl[unsafe_offset=s])
        var total = Int(seq_lens[unsafe_offset=s])
        var ctx = total - (Int(qsl[unsafe_offset=s + 1]) - q0)
        var complete = ps + P <= total
        if mode != 0 and complete:
            continue
        for h in range(H):
            var tb = cc.tail(s, h) * D
            if mode == 0:
                var rec = cc.rec(s, ps // P, h)
                for t in range(max(ps, ctx), min(ps + P, total)):
                    _quantize_v_row[D, BITS](v_new.unsafe_offset((q0 + t - ctx) * vs + h * D), rec, t - ps)
                if complete:
                    var split = max(0, min(P, ctx - ps))
                    _quantize_k_page[D, BITS](cc.tail_k.unsafe_offset(tb), D,
                                              k_new.unsafe_offset((q0 + ps + split - ctx) * ks + h * D), ks, split, rec)
            else:
                for t in range(max(ps, ctx), total):
                    var r = q0 + t - ctx
                    for d in range(0, D, W):
                        cc.tail_k.unsafe_store(tb + (t - ps) * D + d, k_new.unsafe_load[width=W](r * ks + h * D + d))


# ------------------------------------------------------------------ decode
@always_inline
def _codes[D: Int, BITS: Int](row: I8P, d: Int) -> Tuple[V32, V32]:
    """Codes of dims [d, d+W) (int8) or ([d, d+W), [d+D/2, d+D/2+W)) (int4, d < D/2) as fp32."""
    var b = row.unsafe_load[width=W](d)
    comptime if BITS == 8:
        return (b.cast[DType.float32](), V32(0))
    else:
        return (((b << 4) >> 4).cast[DType.float32](), (b >> 4).cast[DType.float32]())


def _decode_task[D: Int, BITS: Int](q: F32P, o: F32P, cc: _Cache, seq_lens: I32P, G: Int,
                                    sm: Float32, s: Int, h: Int):
    """Attention of the G q heads of (sequence s, kv head h) over the whole cache."""
    comptime DC = row_bytes[D, BITS]()
    comptime NCH = D if BITS == 8 else D // 2
    var H = cc.hkv
    var n = Int(seq_lens[unsafe_offset=s])
    var qb = (s * H + h) * G * D
    var scr = _buf(G * D * 3 + D + G * (P + 4))
    var qsm = scr                                 # q * sm                       [G][D]
    var qs = scr.unsafe_offset(G * D)             # q * sm * k_scale (this page) [G][D]
    var acc = scr.unsafe_offset(2 * G * D)        # unnormalized output          [G][D]
    var crow = scr.unsafe_offset(3 * G * D)       # one row of codes as fp32     [D]
    var sc = crow.unsafe_offset(D)                # scores -> weights            [G][P]
    var m = sc.unsafe_offset(G * P)
    var l = m.unsafe_offset(G)
    var vbacc = l.unsafe_offset(G)
    var kbs = vbacc.unsafe_offset(G)              # q . k_bias (this page)       [G]
    for g in range(G):
        for d in range(0, D, W):
            qsm.unsafe_store(g * D + d, q.unsafe_load[width=W](qb + g * D + d) * sm)
            acc.unsafe_store(g * D + d, V32(0))
        m[unsafe_offset=g] = Float32.MIN
        l[unsafe_offset=g] = 0
        vbacc[unsafe_offset=g] = 0

    @always_inline
    def softmax_page(cnt: Int, g: Int) {var sc, var m, var l} -> Float32:
        var row = sc.unsafe_offset(g * P)
        var pm = Float32.MIN
        for t in range(cnt):
            pm = max(pm, row[unsafe_offset=t])
        var mo = m[unsafe_offset=g]
        var mn = max(mo, pm)
        var corr = exp(mo - mn)
        var tot = Float32(0)
        for t in range(cnt):
            var e = exp(row[unsafe_offset=t] - mn)
            row[unsafe_offset=t] = e
            tot += e
        m[unsafe_offset=g] = mn
        l[unsafe_offset=g] = l[unsafe_offset=g] * corr + tot
        return corr

    @always_inline
    def unpack_row(row: I8P) {var crow}:
        """Codes of one token row -> fp32 once; reused by all G heads."""
        for d in range(0, NCH, W):
            var c = _codes[D, BITS](row, d)
            crow.unsafe_store(d, c[0])
            comptime if BITS == 4:
                crow.unsafe_store(d + D // 2, c[1])

    for lp in range(n // P):
        var kc = cc.rec(s, lp, h)
        var vc = kc.unsafe_offset(off_vq[D, BITS]())
        var ksp = kc.unsafe_offset(off_ks[D, BITS]()).unsafe_bitcast[Float16]()
        var kbp = kc.unsafe_offset(off_kb[D, BITS]()).unsafe_bitcast[Float16]()
        var vsp = kc.unsafe_offset(off_vs[D, BITS]()).unsafe_bitcast[Float16]()
        var vbp = kc.unsafe_offset(off_vb[D, BITS]()).unsafe_bitcast[Float16]()
        # fold the K scale into q, the K bias into one scalar per head
        for g in range(G):
            var kbias = V32(0)
            for d in range(0, D, W):
                var qv = qsm.unsafe_load[width=W](g * D + d)
                qs.unsafe_store(g * D + d, qv * ksp.unsafe_load[width=W](d).cast[DType.float32]())
                kbias += qv * kbp.unsafe_load[width=W](d).cast[DType.float32]()
            kbs[unsafe_offset=g] = kbias.reduce_add()
        for t in range(P):
            unpack_row(kc.unsafe_offset(t * DC))
            for g in range(G):
                var dot = V32(0)
                for d in range(0, D, W):
                    dot += qs.unsafe_load[width=W](g * D + d) * crow.unsafe_load[width=W](d)
                sc[unsafe_offset=g * P + t] = dot.reduce_add() + kbs[unsafe_offset=g]
        for g in range(G):
            var corr = softmax_page(P, g)
            vbacc[unsafe_offset=g] = vbacc[unsafe_offset=g] * corr
            for d in range(0, D, W):
                acc.unsafe_store(g * D + d, acc.unsafe_load[width=W](g * D + d) * corr)
        # V scale folds into the weights, V bias into one scalar per head
        for t in range(P):
            unpack_row(vc.unsafe_offset(t * DC))
            var vst = Float32(vsp[unsafe_offset=t])
            var vbt = Float32(vbp[unsafe_offset=t])
            for g in range(G):
                var e = sc[unsafe_offset=g * P + t]
                vbacc[unsafe_offset=g] += e * vbt
                var wv = V32(e * vst)
                for d in range(0, D, W):
                    acc.unsafe_store(g * D + d, acc.unsafe_load[width=W](g * D + d) + wv * crow.unsafe_load[width=W](d))

    var tail_n = n % P
    if tail_n > 0:                                # exact K from the tail, V rows from the record
        var tk = cc.tail_k.unsafe_offset(cc.tail(s, h) * D)
        var vrec = cc.rec(s, n // P, h)
        var vc = vrec.unsafe_offset(off_vq[D, BITS]())
        var vsp = vrec.unsafe_offset(off_vs[D, BITS]()).unsafe_bitcast[Float16]()
        var vbp = vrec.unsafe_offset(off_vb[D, BITS]()).unsafe_bitcast[Float16]()
        for g in range(G):
            for t in range(tail_n):
                var dot = V32(0)
                for d in range(0, D, W):
                    dot += qsm.unsafe_load[width=W](g * D + d) * tk.unsafe_load[width=W](t * D + d)
                sc[unsafe_offset=g * P + t] = dot.reduce_add()
            var corr = softmax_page(tail_n, g)
            vbacc[unsafe_offset=g] = vbacc[unsafe_offset=g] * corr
            for d in range(0, D, W):
                acc.unsafe_store(g * D + d, acc.unsafe_load[width=W](g * D + d) * corr)
        for t in range(tail_n):
            unpack_row(vc.unsafe_offset(t * DC))
            var vst = Float32(vsp[unsafe_offset=t])
            var vbt = Float32(vbp[unsafe_offset=t])
            for g in range(G):
                var e = sc[unsafe_offset=g * P + t]
                vbacc[unsafe_offset=g] += e * vbt
                var wv = V32(e * vst)
                for d in range(0, D, W):
                    acc.unsafe_store(g * D + d, acc.unsafe_load[width=W](g * D + d) + wv * crow.unsafe_load[width=W](d))

    for g in range(G):
        var inv_l = Float32(1) / l[unsafe_offset=g] if l[unsafe_offset=g] > 0 else Float32(0)
        var vb_g = vbacc[unsafe_offset=g]
        for d in range(0, D, W):
            o.unsafe_store(qb + g * D + d, (acc.unsafe_load[width=W](g * D + d) + vb_g) * inv_l)
    scr.unsafe_free()


def _decode[D: Int, BITS: Int](q: F32P, o: F32P, cc: _Cache, seq_lens: I32P, G: Int, sm: Float32, S: Int):
    var tasks = S * cc.hkv
    var pages = 0
    for s in range(S):
        pages += Int(seq_lens[unsafe_offset=s]) // P + 1
    if pages * cc.hkv < 256:                      # small: thread-pool start-up costs more than it saves
        for i in range(tasks):
            _decode_task[D, BITS](q, o, cc, seq_lens, G, sm, i // cc.hkv, i % cc.hkv)
        return

    def work(i: Int) raises {var q, var o, var cc, var seq_lens, var G, var sm}:
        _decode_task[D, BITS](q, o, cc, seq_lens, G, sm, i // cc.hkv, i % cc.hkv)
    sync_parallelize(work, tasks)


# ------------------------------------------------------------------ C ABI entry points
comptime _DIMS = [64, 128, 256]


@export
def kush_cpu_append(d: Int64, bits: Int64, k_new: Int64, v_new: Int64, k_stride: Int64, v_stride: Int64,
                    seq_lens: Int64,
                    pos_offset: Int64, advance: Int64, cache: Int64, bs: Int64, hs: Int64, tail_k: Int64,
                    tail_idx: Int64, block_table: Int64, max_pages: Int64, hkv: Int64,
                    n_seqs: Int64) abi("C") -> Int32:
    var cc = _cache(cache, bs, hs, tail_k, tail_idx, block_table, max_pages, hkv)
    comptime for di in range(len(_DIMS)):
        comptime DD = _DIMS[di]
        if Int(d) == DD:
            if bits == 4:
                _append[DD, 4](_p[Float32](k_new), _p[Float32](v_new), Int(k_stride), Int(v_stride), _p[Int32](seq_lens),
                               Int(pos_offset), advance != 0, cc, Int(n_seqs))
                return 0
            if bits == 8:
                _append[DD, 8](_p[Float32](k_new), _p[Float32](v_new), Int(k_stride), Int(v_stride), _p[Int32](seq_lens),
                               Int(pos_offset), advance != 0, cc, Int(n_seqs))
                return 0
    return -1


@export
def kush_cpu_write(d: Int64, bits: Int64, k_new: Int64, v_new: Int64, k_stride: Int64, v_stride: Int64, qsl: Int64,
                   seq_lens: Int64, job_seq: Int64, job_page: Int64, n_jobs: Int64, mode: Int64,
                   cache: Int64, bs: Int64, hs: Int64, tail_k: Int64, tail_idx: Int64,
                   block_table: Int64, max_pages: Int64, hkv: Int64) abi("C") -> Int32:
    var cc = _cache(cache, bs, hs, tail_k, tail_idx, block_table, max_pages, hkv)
    comptime for di in range(len(_DIMS)):
        comptime DD = _DIMS[di]
        if Int(d) == DD:
            if bits == 4:
                _write[DD, 4](_p[Float32](k_new), _p[Float32](v_new), Int(k_stride), Int(v_stride), _p[Int32](qsl),
                              _p[Int32](seq_lens), _p[Int32](job_seq), _p[Int32](job_page), Int(n_jobs), Int(mode), cc)
                return 0
            if bits == 8:
                _write[DD, 8](_p[Float32](k_new), _p[Float32](v_new), Int(k_stride), Int(v_stride), _p[Int32](qsl),
                              _p[Int32](seq_lens), _p[Int32](job_seq), _p[Int32](job_page), Int(n_jobs), Int(mode), cc)
                return 0
    return -1


@export
def kush_cpu_decode(d: Int64, bits: Int64, q: Int64, o: Int64, cache: Int64, bs: Int64, hs: Int64,
                    tail_k: Int64, tail_idx: Int64, block_table: Int64, seq_lens: Int64,
                    max_pages: Int64, hkv: Int64, group: Int64, sm: Float32, n_seqs: Int64) abi("C") -> Int32:
    var cc = _cache(cache, bs, hs, tail_k, tail_idx, block_table, max_pages, hkv)
    comptime for di in range(len(_DIMS)):
        comptime DD = _DIMS[di]
        if Int(d) == DD:
            if bits == 4:
                _decode[DD, 4](_p[Float32](q), _p[Float32](o), cc, _p[Int32](seq_lens), Int(group), sm, Int(n_seqs))
                return 0
            if bits == 8:
                _decode[DD, 8](_p[Float32](q), _p[Float32](o), cc, _p[Int32](seq_lens), Int(group), sm, Int(n_seqs))
                return 0
    return -1

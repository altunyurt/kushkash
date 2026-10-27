# kushkash cache format — the single source of truth shared by the CPU and GPU kernels,
# so both backends produce bit-identical pages. Width-generic: the GPU quantizes one
# channel per thread (W = 1), the CPU W channels per SIMD op.
#
# A page holds P = 32 tokens of one (layer, kv head). Every value is stored as an
# asymmetric code:  x ~= s * c + b,  c in [-HALF, HALF - 1],  HALF = 2^(BITS-1).
#   K  : s, b per channel (over the page's 32 tokens)  -> absorbs per-channel offsets/outliers
#   V  : s, b per token   (over the D channels)
#   s, b are stored as fp16; codes are computed against the fp16-rounded values.
# Byte layout of one page row (DC bytes):
#   BITS = 8 : byte j = code of dim j
#   BITS = 4 : byte j = code of dim j (low nibble) | code of dim j + D/2 (high nibble)

from std.memory import bitcast

comptime PAGE = 32


@always_inline
def row_bytes[D: Int, BITS: Int]() -> Int:
    """Bytes per token row of one page (DC)."""
    return D if BITS == 8 else D // 2


@always_inline
def qparams[BITS: Int, W: Int = 1](
    mn: SIMD[DType.float32, W], mx: SIMD[DType.float32, W]
) -> Tuple[SIMD[DType.float32, W], SIMD[DType.float32, W], SIMD[DType.float32, W]]:
    """(scale, bias, 1/scale) for groups with range [mn, mx]; scale and bias fp16-rounded."""
    comptime LEVELS = Float32((1 << BITS) - 1)
    comptime HALF = Float32(1 << (BITS - 1))
    var s = ((mx - mn) / LEVELS).cast[DType.float16]().cast[DType.float32]()
    var b = (mn + s * HALF).cast[DType.float16]().cast[DType.float32]()
    var inv = s.gt(0).select(Float32(1) / s, SIMD[DType.float32, W](0))
    return (s, b, inv)


@always_inline
def qcode[BITS: Int, W: Int = 1](
    x: SIMD[DType.float32, W], b: SIMD[DType.float32, W], inv: SIMD[DType.float32, W]
) -> SIMD[DType.int8, W]:
    """Signed codes of x for groups with bias b and inverse scale inv."""
    comptime HALF = Float32(1 << (BITS - 1))
    return round((x - b) * inv).clamp(-HALF, HALF - 1).cast[DType.int8]()


@always_inline
def pack4[W: Int = 1](lo: SIMD[DType.int8, W], hi: SIMD[DType.int8, W]) -> SIMD[DType.int8, W]:
    """Two signed 4-bit codes -> one byte (lo in bits 0-3, hi in bits 4-7)."""
    return (lo & 0x0F) | (hi << 4)


# ------------------------------------------------------------------ page record
# One page of one kv head is a single contiguous record (vLLM's [block][head] cell
# and the HF pool use the same bytes):
#   [ kq: P*DC int8 | vq: P*DC int8 | ks: D fp16 | kb: D fp16 | vs: P fp16 | vb: P fp16 ]
# Record r of (block b, head h) starts at cache + b*block_stride + h*head_stride.
# Size is a multiple of 16 bytes for D % 64 == 0 (16-byte loads stay aligned).

@always_inline
def rec_bytes[D: Int, BITS: Int]() -> Int:
    return 2 * PAGE * row_bytes[D, BITS]() + 4 * D + 4 * PAGE


@always_inline
def off_vq[D: Int, BITS: Int]() -> Int:
    return PAGE * row_bytes[D, BITS]()


@always_inline
def off_ks[D: Int, BITS: Int]() -> Int:
    return 2 * PAGE * row_bytes[D, BITS]()


@always_inline
def off_kb[D: Int, BITS: Int]() -> Int:
    return off_ks[D, BITS]() + 2 * D


@always_inline
def off_vs[D: Int, BITS: Int]() -> Int:
    return off_ks[D, BITS]() + 4 * D


@always_inline
def off_vb[D: Int, BITS: Int]() -> Int:
    return off_ks[D, BITS]() + 4 * D + 2 * PAGE


# ------------------------------------------------------------------ rotation (Walsh-Hadamard)
# At 4 bits the cache holds K and V rotated by the orthonormal Walsh-Hadamard matrix H (Sylvester
# order: H_1 = [1], H_2n = [[H_n, H_n], [H_n, -H_n]], scaled by 1/sqrt(D); symmetric, H @ H = I, so
# the same transform rotates back). Every backend evaluates it in one canonical way, so rotated
# values are bit-identical across CPU and GPU:
#   1. butterflies in fp32 over strides h = 1, 2, 4, ..., D/2, in that order:
#      (x[i], x[i + h]) <- (x[i] + x[i + h], x[i] - x[i + h]) for every i with bit h clear;
#   2. one multiply by fwht_scale[D](), never fused into a later operation (an FMA there rounds
#      differently, and only where a backend's compiler chooses to contract).
# Butterflies over different strides commute in exact arithmetic, not in fp32: the order is fixed.

@always_inline
def fwht_scale[D: Int]() -> Float32:
    """1/sqrt(D) rounded to fp32, as exact bits (no decimal-literal rounding in between)."""
    comptime assert D == 64 or D == 128 or D == 256, "head_dim must be 64, 128 or 256"
    comptime if D == 64:
        return bitcast[DType.float32](UInt32(0x3E000000))      # 0.125
    elif D == 128:
        return bitcast[DType.float32](UInt32(0x3DB504F3))      # 0.088388346
    else:
        return bitcast[DType.float32](UInt32(0x3D800000))      # 0.0625


@always_inline
def fwht_stages[D: Int]() -> Int:
    """log2(D): butterfly stages of the transform (strides 1 << k for k < fwht_stages)."""
    comptime assert D == 64 or D == 128 or D == 256, "head_dim must be 64, 128 or 256"
    comptime if D == 64:
        return 6
    elif D == 128:
        return 7
    else:
        return 8

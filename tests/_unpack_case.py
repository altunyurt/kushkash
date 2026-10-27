"""Shared inputs for the unpack kernel tests (CPU and GPU)."""
import torch

from kushkash._ops import PAGE, record_bytes


def unpack_case(D: int, bits: int, dtype: torch.dtype, seed: int = 0):
    """Random records with sane fp16 scales / biases, 5 full pages, 3 partial pages (1, 17, 31 tokens),
    45 new rows (a partial block of 32), packed into rows with a never-written gap row between groups."""
    g = torch.Generator().manual_seed(seed)
    H, P, n_blk, slots = 2, PAGE, 12, 4
    rb = record_bytes(D, bits)
    dc = D if bits == 8 else D // 2
    rec = torch.randint(0, 256, (n_blk, H, rb), dtype=torch.uint8, generator=g)
    m = rec[:, :, 2 * P * dc:].view(torch.float16)
    m.copy_((torch.rand(m.shape, generator=g) * 0.1).half())                          # scales
    m[:, :, D:2 * D] = ((torch.rand(n_blk, H, D, generator=g) - 0.5) * 0.4).half()    # K biases, signed
    m[:, :, 2 * D + P:] = ((torch.rand(n_blk, H, P, generator=g) - 0.5) * 0.4).half()  # V biases
    tail_k = (torch.randn(slots, H, P, D, generator=g) * 0.5).to(dtype)
    page_blk = torch.tensor([7, 2, 9, 0, 11], dtype=torch.int32)
    page_dst = torch.tensor([0, 32, 64, 97, 129], dtype=torch.int32)                  # row 96: gap
    tails = [(161, 1, 3, 5), (163, 17, 0, 8), (180, 31, 2, 1)]                       # (first row, n, slot, blk)
    tail_blk = torch.tensor([t[3] for t in tails], dtype=torch.int32)
    tail_info = torch.tensor([x for t in tails for x in t[:3]], dtype=torch.int32)    # row 162: gap
    n_new = 45
    new_dst = torch.cat([torch.arange(212, 212 + 20), torch.arange(233, 233 + 25)]).to(torch.int32)  # 232: gap
    rows = 260
    k_new = (torch.randn(n_new, H, D, generator=g)).to(dtype)
    v_new = (torch.randn(n_new, H, D, generator=g)).to(dtype)
    return dict(records=rec, page_blk=page_blk, page_dst=page_dst, tail_blk=tail_blk, tail_info=tail_info,
                tail_k=tail_k, k_new=k_new, v_new=v_new, new_dst=new_dst, rows=rows, H=H)

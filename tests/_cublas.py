"""Shared by test_cpu.py and test_gpu.py: the cuBLAS reduction flags rotate() may switch off for one GEMM."""
import torch

NAMES = ("fp16_reduced_precision_reduction", "bf16_reduced_precision_reduction", "fp16_accumulation")


def cublas_flags():
    """Every flag _cublas_exact_sums may touch, as torch._C reports it (flags this torch lacks are skipped)."""
    return {n: getattr(torch._C, "_get_cublas_allow_" + n)() for n in NAMES
            if hasattr(torch._C, "_get_cublas_allow_" + n)}

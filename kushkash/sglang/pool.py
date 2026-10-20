"""KV pool holding kushkash page records: per layer a uint8 tensor [pages, Hkv, record_bytes].

SGLang addresses tokens by slot (page = slot // 32); page 0 holds slot 0, SGLang's padding slot. Only the
kushkash backend reads or writes the records, so the K/V-row accessors of the base pool are disabled.
The exact K tail of each request's unfinished page lives in `kush_tail` (per layer [req slots, Hkv, 32, D]),
allocated by the backend once the request-pool size is known.
"""
from __future__ import annotations

from contextlib import nullcontext

import torch
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

from kushkash._ops import PAGE, record_bytes

from . import bits as _bits

try:
    from sglang.srt.mem_cache.memory_pool import GPU_MEMORY_TYPE_KV_CACHE
except ImportError:                                    # older/newer layouts: no memory-saver tag
    GPU_MEMORY_TYPE_KV_CACHE = None


class KushPool(MHATokenToKVPool):

    def _create_buffers(self):
        if self.page_size != PAGE:
            raise ValueError(f"kushkash needs page_size {PAGE}, got {self.page_size}")
        if self.head_dim != self.v_head_dim:
            raise ValueError("kushkash needs equal K and V head sizes")
        self.kush_bits = _bits()
        self.kush_record_bytes = record_bytes(self.head_dim, self.kush_bits)
        self.kush_num_pages = -(-(self.size + self.page_size) // self.page_size)
        for name in ("k_scale_buffer", "v_scale_buffer", "native_k_scale_buffer", "native_v_scale_buffer",
                     "dq_k_buffer", "dq_v_buffer"):
            setattr(self, name, None)
        region = (self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE)
                  if GPU_MEMORY_TYPE_KV_CACHE is not None else nullcontext())
        with region:
            self.k_buffer = [torch.zeros((self.kush_num_pages, self.head_num, self.kush_record_bytes),
                                         dtype=torch.uint8, device=self.device) for _ in range(self.layer_num)]
        self.v_buffer = [torch.empty((self.kush_num_pages, self.head_num, 0), dtype=torch.uint8, device=self.device)
                         for _ in range(self.layer_num)]
        self.kush_tail = None
        self._kv_buffer_descs = []
        self._init_data_ptrs_and_strides()

    def records(self, layer_id: int) -> torch.Tensor:
        return self.k_buffer[layer_id - self.start_layer]

    def tail(self, layer_id: int) -> torch.Tensor:
        return self.kush_tail[layer_id - self.start_layer]

    # ---------------------------------------------------------------- not supported in the spike
    def _unsupported(self, what: str):
        raise NotImplementedError(f"kushkash pool: {what} is not supported (pages are kushkash records)")

    def get_key_buffer(self, *args, **kwargs):
        self._unsupported("get_key_buffer (K/V rows; records are read by the kushkash backend)")

    def get_value_buffer(self, *args, **kwargs):
        self._unsupported("get_value_buffer (K/V rows; records are read by the kushkash backend)")

    def get_kv_buffer(self, *args, **kwargs):
        self._unsupported("get_kv_buffer (K/V rows; records are read by the kushkash backend)")

    def set_kv_buffer(self, *args, **kwargs):
        self._unsupported("set_kv_buffer (the kushkash backend writes the cache itself)")

    def move_kv_cache(self, *args, **kwargs):
        self._unsupported("move_kv_cache")

    def get_cpu_copy(self, *args, **kwargs):
        self._unsupported("get_cpu_copy")

    def load_cpu_copy(self, *args, **kwargs):
        self._unsupported("load_cpu_copy")

    def get_contiguous_buf_infos(self):
        self._unsupported("PD disaggregation")

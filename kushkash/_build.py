"""Compile the Mojo kernels on first use and cache the results.

CPU: kush/cpu.mojo -> libkush_cpu.so (C ABI, loaded with ctypes).
GPU: one PTX module per kernel instantiation, emitted by a generated Mojo program via
     `compile_info`, specialized for (head_dim, bits, q-heads-per-kv-head, dtype, arch).

Artifacts live in $KUSHKASH_CACHE (default ~/.cache/kushkash)/<sha256 of sources + mojo
version + config>, so a source change or a Mojo upgrade rebuilds automatically.
"""
from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent          # contains the `kush` Mojo package
MOJO_DIR = PKG_DIR / "kush"
PAGE = 32
SMEM_LIMIT = 48 * 1024                              # static shared memory per block
_DTYPES = {"float16": "DType.float16", "bfloat16": "DType.bfloat16", "float32": "DType.float32"}


class BuildError(RuntimeError):
    pass


def cache_root() -> Path:
    return Path(os.environ.get("KUSHKASH_CACHE", Path.home() / ".cache" / "kushkash"))


@lru_cache(None)
def mojo_exe() -> str:
    """`mojo` from $KUSHKASH_MOJO, PATH, or next to the running Python."""
    cands = [os.environ.get("KUSHKASH_MOJO"), shutil.which("mojo"), str(Path(sys.executable).parent / "mojo")]
    for c in cands:
        if c and Path(c).exists():
            return c
    raise BuildError("`mojo` not found: `pip install mojo` (>= 1.1) or set KUSHKASH_MOJO")


@lru_cache(None)
def _mojo_version() -> str:
    return subprocess.run([mojo_exe(), "--version"], capture_output=True, text=True).stdout.strip()


def _key(*parts: str, files: tuple[str, ...]) -> str:
    h = hashlib.sha256(_mojo_version().encode())
    for f in files:
        h.update((MOJO_DIR / f).read_bytes())
    for p in parts:
        h.update(p.encode())
    return h.hexdigest()[:16]


def _run(cmd: list[str], cwd: Path) -> str:
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        raise BuildError(f"{' '.join(cmd)} failed:\n{r.stderr[-4000:]}")
    return r.stdout


def _publish(tmp: Path, out: Path) -> None:
    """Atomic directory publish (safe against concurrent first-use builds)."""
    try:
        tmp.rename(out)
    except OSError:                                  # another process won the race
        shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------------------------ CPU
@lru_cache(None)
def _cpu_id() -> str:
    """Host CPU identity: `mojo build` targets the native CPU, so a library built on another machine
    (shared or synced cache dir) can die with SIGILL here."""
    try:
        info = Path("/proc/cpuinfo").read_text().splitlines()
        pick = [ln.split(":", 1)[1].strip() for ln in info if ln.startswith(("model name", "flags", "Features"))]
        return "|".join(dict.fromkeys(pick))
    except OSError:
        import platform
        return f"{platform.machine()}|{platform.processor()}"


@lru_cache(None)
def cpu_library() -> Path:
    out = cache_root() / f"cpu-{_key(_cpu_id(), files=('quant.mojo', 'cpu.mojo'))}"
    lib = out / "libkush_cpu.so"
    if lib.exists():
        return lib
    cache_root().mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(dir=cache_root()))
    _run([mojo_exe(), "build", "--emit", "shared-lib", "-I", str(PKG_DIR), str(MOJO_DIR / "cpu.mojo"),
          "-o", str(tmp / "libkush_cpu.so")], cwd=tmp)
    _publish(tmp, out)
    return lib


# ------------------------------------------------------------------ GPU
def _sm(arch: str) -> int:
    """'sm_89' -> 89, 'sm_120' -> 120, 'sm_90a' -> 90."""
    return int(arch.split("_")[1].rstrip("af"))


@dataclass(frozen=True)
class GpuConfig:
    head_dim: int
    bits: int
    group: int            # q heads per kv head
    dtype: str            # q / out / new-token dtype: float16 | bfloat16 | float32
    arch: str             # sm_XY

    @property
    def tensor_cores(self) -> bool:
        return _sm(self.arch) >= 80 and self.head_dim % 128 == 0 and self.group <= 8

    @property
    def warps(self) -> int:
        """Warps per decode block: the most (<= 4) whose static shared memory fits."""
        D, G = self.head_dim, self.group
        for nw in (4, 2, 1):
            if self.tensor_cores:
                smem = G * D * 4 + nw * PAGE * (D + 8) * 2 + G * PAGE * 4 + 2 * nw * G * 4
            else:
                dc = D if self.bits == 8 else D // 2
                stage = nw * PAGE * (dc // 4 + 4)
                smem = (G * D + nw * G * D + max(stage, nw * G * D) + nw * PAGE * G + 2 * nw * G) * 4
            if smem <= SMEM_LIMIT:
                return nw
        raise BuildError(f"head_dim={D} with {G} q heads per kv head exceeds shared memory")

    def kernels(self) -> dict[str, str]:
        D, B, G, IN, NW = self.head_dim, self.bits, self.group, _DTYPES[self.dtype], self.warps
        dec = "kush_decode_tc" if self.tensor_cores else "kush_decode"
        return {
            "append": f"kush_append[{D}, {B}, {IN}]",
            "write": f"kush_write[{D}, {B}, {IN}]",
            "decode": f"{dec}[{D}, {B}, {G}, {NW}, {IN}]",
            "combine": f"kush_combine[{D}, {G}, {IN}]",
        }


_GEN = """from kush.gpu import kush_append, kush_write, kush_decode_tc, kush_decode, kush_combine
from max.gpu.host.compile import get_gpu_target
from std.compile import compile_info

def main():
    comptime T = get_gpu_target["{arch}"]()
{body}
"""
_ENTRY = re.compile(r"\.visible\s+\.entry\s+(\S+)\(([^)]*)\)", re.S)
_PARAM = re.compile(r"\.param\s+\.(\w+)")


def _emit_ptx(cfg: GpuConfig, arch: str, tmp: Path) -> dict[str, str]:
    body = "".join(
        f'    print("@@{name}")\n    print(String(compile_info[{inst}, emission_kind="asm", target=T]()))\n'
        for name, inst in cfg.kernels().items())
    src = tmp / "gen.mojo"
    src.write_text(_GEN.format(arch=arch, body=body))
    text = _run([mojo_exe(), "run", "-I", str(PKG_DIR), str(src)], cwd=tmp)
    return {m.group(1): m.group(2) for m in re.finditer(r"@@(\w+)\n(.*?)(?=\n@@|\Z)", text, re.S)}


@lru_cache(None)
def gpu_modules(cfg: GpuConfig) -> dict[str, tuple[str, str, tuple[str, ...]]]:
    """{kernel: (ptx path, entry name, PTX param types)}. Builds for cfg.arch; if this Mojo
    cannot target it, falls back to sm_80 PTX (JIT-compiled by the driver)."""
    out = cache_root() / f"gpu-{_key(repr(cfg), files=('quant.mojo', 'gpu.mojo'))}"
    if not (out / "ok").exists():
        cache_root().mkdir(parents=True, exist_ok=True)
        tmp = Path(tempfile.mkdtemp(dir=cache_root()))
        try:
            ptx = _emit_ptx(cfg, cfg.arch, tmp)
        except BuildError:
            if _sm(cfg.arch) < 80:
                raise
            ptx = _emit_ptx(cfg, "sm_80", tmp)
        for name, text in ptx.items():
            if "__local_depot" in text:
                warnings.warn(f"kushkash: {name} spills registers for {cfg} (slower, still correct)")
            (tmp / f"{name}.ptx").write_text(text)
        (tmp / "gen.mojo").unlink()
        (tmp / "ok").write_text(repr(cfg))
        _publish(tmp, out)
    mods = {}
    for name in cfg.kernels():
        path = out / f"{name}.ptx"
        m = _ENTRY.search(path.read_text())
        mods[name] = (str(path), m.group(1), tuple(_PARAM.findall(m.group(2))))
    return mods

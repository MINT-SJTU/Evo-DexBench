"""Cross-CPU numerics for exact replay (x86-64 Linux only).

Why this exists
---------------
The x86 approximate-reciprocal instructions ``RCPPS``/``RSQRTPS`` return
vendor-specific values: Intel and AMD CPUs implement different lookup tables
(both depend only on the leading 10-12 mantissa bits plus exact exponent
scaling). Two places in the simulation stack execute them:

* PhysX inside ``libsapien.so`` (e.g. contact generation), and
* Intel MKL's portable "EX" vector-math kernels inside ``libtorch_cpu.so``
  (``sqrt``/``acos``/``asin`` on small tensors). MKL always uses these kernels
  on AMD CPUs, and on Intel CPUs when ``MKL_CBWR=COMPATIBLE``.

Episodes recorded on an AMD CPU therefore diverge when replayed on an Intel CPU
(and vice versa) even with identical code, packages and inputs.

What it does
------------
``install_cpu_numerics("amd")`` makes an Intel host compute exactly what an AMD
(Zen 5) host computes in those code paths: every RCPPS/RSQRTPS site in the
patched functions is redirected, in memory only, to a bit-exact table emulation
(verified against native hardware for all 2^32 inputs on both vendors). On an
AMD host the call is a verified no-op. Nothing on disk is modified, and each
library and each patched byte window is checked against its expected SHA-256 /
original bytes before patching.

Requirements: x86-64 Linux, a C compiler (``cc``), and GNU binutils when the
installed ``libtorch_cpu.so``/``libsapien.so`` builds have no bundled spec.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from typing import Any

from dex_benchmark.sim.determinism.spec_gen import generate_patch_spec

PACKAGE_DIR = Path(__file__).resolve().parent
TABLE_PATHS: dict[str, Path] = {
    # Reference CPU whose RCPPS/RSQRTPS tables are emulated.
    "amd": PACKAGE_DIR / "tables" / "amd_ryzen9_9950x3d.bin",
}
TABLE_SHA256: dict[str, str] = {
    "amd": "5988ae33074dccf30ff6a7115981bf6973b8d065592590b8d92515b240f470b0",
}
# MKL single-precision "EX" kernels that contain RSQRTPS and are reached by the
# simulation (verified by tracing). Other EX kernels matched identically.
TORCH_MKL_FUNCTION_PATTERN = r"^mkl_vml_kernel_s(Sqrt|Acos|Asin)_EX"
SUPPORTED_TARGETS = ("amd",)
_MAX_OPS = 6


class _Window(ctypes.Structure):
    """Mirror of ``window_t`` in hotpatch.c (layout checked in ``_load_helper``)."""

    _fields_ = [
        ("off", ctypes.c_uint64),
        ("patch_len", ctypes.c_uint32),
        ("nops", ctypes.c_uint32),
        ("orig", ctypes.c_uint8 * 48),
        ("op_type", ctypes.c_uint32 * _MAX_OPS),
        ("op_kind", ctypes.c_uint32 * _MAX_OPS),
        ("op_dst", ctypes.c_uint32 * _MAX_OPS),
        ("op_src", ctypes.c_uint32 * _MAX_OPS),
        ("op_counter", ctypes.c_uint32 * _MAX_OPS),
        ("op_off", ctypes.c_uint64 * _MAX_OPS),
        ("op_len", ctypes.c_uint32 * _MAX_OPS),
        ("op_rip", ctypes.c_uint32 * _MAX_OPS),
        ("op_bytes", (ctypes.c_uint8 * 16) * _MAX_OPS),
    ]


@dataclass
class CpuNumericsReport:
    """What ``install_cpu_numerics`` did; safe to serialize with ``as_dict``."""

    target: str
    host_vendor: str
    patched: bool
    mkl_cbwr: str | None
    libraries: list[dict[str, Any]] = field(default_factory=list)
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "host_vendor": self.host_vendor,
            "patched": self.patched,
            "mkl_cbwr": self.mkl_cbwr,
            "libraries": self.libraries,
            "note": self.note,
        }


_helper: ctypes.CDLL | None = None
_installed: list[tuple[dict[str, Any], int]] = []
_next_counter = 0
_target_installed: str | None = None


def host_cpu_vendor() -> str:
    """Return the CPUID vendor string, e.g. ``GenuineIntel`` or ``AuthenticAMD``."""

    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("vendor_id"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "unknown"


def _check_platform() -> None:
    if sys.platform != "linux" or platform.machine() not in ("x86_64", "AMD64"):
        raise RuntimeError("Cross-CPU numerics emulation supports x86-64 Linux only.")


def prepare_cpu_numerics(target: str) -> None:
    """Set process settings that must precede the first MKL call.

    Call before importing torch. On Intel hosts MKL must run its portable
    ("COMPATIBLE") code path, which is the one AMD hosts always use.
    """

    if target not in SUPPORTED_TARGETS:
        raise ValueError(f"Unsupported CPU numerics target {target!r}; supported: {SUPPORTED_TARGETS}.")
    _check_platform()
    current = os.environ.get("MKL_CBWR")
    if current not in (None, "", "COMPATIBLE"):
        raise RuntimeError(f"MKL_CBWR={current!r} conflicts with AMD emulation; unset it or use COMPATIBLE.")
    if current != "COMPATIBLE" and "torch" in sys.modules and host_cpu_vendor() == "GenuineIntel":
        raise RuntimeError(
            "torch was imported before MKL_CBWR=COMPATIBLE was set. Export MKL_CBWR=COMPATIBLE "
            "before starting Python, or call prepare_cpu_numerics() before importing torch."
        )
    os.environ["MKL_CBWR"] = "COMPATIBLE"


def _load_helper() -> ctypes.CDLL:
    global _helper
    if _helper is not None:
        return _helper
    compiler = shutil.which("cc") or shutil.which("gcc")
    if compiler is None:
        raise RuntimeError("A C compiler (cc/gcc) is required to build the CPU numerics helper.")
    source = (PACKAGE_DIR / "hotpatch.c").read_bytes() + (PACKAGE_DIR / "rcp_emu.h").read_bytes()
    digest = hashlib.sha256(source).hexdigest()[:16]
    cache_root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "dex_benchmark" / "determinism" / digest
    cache_root.mkdir(parents=True, exist_ok=True)
    library = cache_root / "libdexbench_hotpatch.so"
    if not library.is_file():
        temporary = cache_root / f"libdexbench_hotpatch.{os.getpid()}.so"
        subprocess.run(
            [compiler, "-O2", "-Wall", "-fPIC", "-shared", "-mgeneral-regs-only", f"-I{PACKAGE_DIR}",
             "-o", str(temporary), str(PACKAGE_DIR / "hotpatch.c")],
            check=True,
        )
        temporary.replace(library)
    helper = ctypes.CDLL(str(library))
    helper.detnum_hits.restype = ctypes.c_uint64
    helper.detnum_hits.argtypes = [ctypes.c_int]
    _helper = helper
    return helper


def _mapped_library(suffix: str) -> Path | None:
    for line in Path("/proc/self/maps").read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 6 and parts[5].endswith(suffix) and int(parts[2], 16) == 0:
            return Path(parts[5])
    return None


def _bundled_spec(sha256: str) -> dict[str, Any] | None:
    for path in sorted((PACKAGE_DIR / "specs").glob("*.json")):
        spec = json.loads(path.read_text(encoding="utf-8"))
        if spec.get("sha256") == sha256:
            return spec
    return None


def _approx_sites(library: Path) -> list[int]:
    objdump = shutil.which("objdump")
    if objdump is None:
        raise RuntimeError("objdump (GNU binutils) is required to scan an unknown libsapien build.")
    listing = subprocess.run([objdump, "-d", "--no-show-raw-insn", str(library)], capture_output=True, text=True, check=True)
    sites: list[int] = []
    for line in listing.stdout.splitlines():
        stripped = line.strip()
        if ":\t" not in stripped:
            continue
        address_text, _, instruction = stripped.partition(":\t")
        mnemonic = instruction.split(" ")[0]
        if mnemonic in ("rcpps", "rsqrtps", "vrcpps", "vrsqrtps", "rcpss", "rsqrtss", "vrcpss", "vrsqrtss"):
            sites.append(int(address_text, 16))
    return sites


def _spec_for(library: Path, sha256: str, selection: str) -> dict[str, Any]:
    spec = _bundled_spec(sha256)
    if spec is not None:
        return spec
    if selection == "torch":
        spec = generate_patch_spec(library, function_pattern=TORCH_MKL_FUNCTION_PATTERN)
    else:
        spec = generate_patch_spec(library, sites=_approx_sites(library))
    return spec


def _install_library(suffix: str, selection: str, tables: bytes) -> dict[str, Any]:
    global _next_counter
    library = _mapped_library(suffix)
    if library is None:
        raise RuntimeError(f"{suffix} is not loaded in this process.")
    sha256 = hashlib.sha256(library.read_bytes()).hexdigest()
    spec = _spec_for(library, sha256, selection)
    if spec["sha256"] != sha256:
        raise RuntimeError(f"Patch spec does not match {library} ({sha256}).")
    if spec["problems"]:
        raise RuntimeError(f"Refusing to patch {library}: unsupported approximate-instruction sites {spec['problems'][:3]}.")
    if not spec["windows"]:
        raise RuntimeError(f"No approximate-instruction windows found in {library}; the build is not supported.")
    windows = (_Window * len(spec["windows"]))()
    for index, window in enumerate(spec["windows"]):
        entry = windows[index]
        entry.off = window["off"]
        entry.patch_len = window["patch_len"]
        entry.nops = len(window["ops"])
        original = bytes.fromhex(window["orig"])
        ctypes.memmove(entry.orig, original, len(original))
        for op_index, op in enumerate(window["ops"]):
            entry.op_type[op_index] = op["type"]
            if op["type"] == 1:
                entry.op_kind[op_index] = 1 if op["kind"] == "rsqrtps" else 0
                entry.op_dst[op_index] = op["dst"]
                entry.op_src[op_index] = op["src"]
                entry.op_counter[op_index] = op["counter"]
            else:
                encoding = bytes.fromhex(op["bytes"])
                entry.op_off[op_index] = op["off"]
                entry.op_len[op_index] = len(encoding)
                entry.op_rip[op_index] = op["rip"]
                ctypes.memmove(entry.op_bytes[op_index], encoding, len(encoding))
    helper = _load_helper()
    message = ctypes.create_string_buffer(256)
    counter_base = _next_counter
    status = helper.detnum_patch(suffix.encode(), tables, ctypes.c_long(len(tables)), windows, len(spec["windows"]),
                                 counter_base, message)
    if status != 0:
        raise RuntimeError(f"Patching {library} failed ({status}): {message.value.decode()}")
    _next_counter += int(spec["counters"])
    _installed.append((spec, counter_base))
    return {"library": str(library), "sha256": sha256, "windows": len(spec["windows"]),
            "emulated_instructions": int(spec["counters"]), "status": message.value.decode()}


def install_cpu_numerics(target: str = "amd") -> CpuNumericsReport:
    """Make this process reproduce ``target`` CPU numerics; call after ``prepare_cpu_numerics``.

    Must run before any simulation step. Imports torch and sapien so that both
    libraries are mapped. Raises instead of degrading silently.
    """

    global _target_installed
    prepare_cpu_numerics(target)
    vendor = host_cpu_vendor()
    if _target_installed is not None:
        if _target_installed != target:
            raise RuntimeError(f"CPU numerics already installed for {_target_installed!r}.")
        return CpuNumericsReport(target, vendor, bool(_installed), os.environ.get("MKL_CBWR"), note="already installed")
    import torch  # noqa: F401  (maps libtorch_cpu.so)
    import sapien  # noqa: F401  (maps libsapien.so)

    if vendor == "AuthenticAMD":
        _target_installed = target
        return CpuNumericsReport(target, vendor, False, os.environ.get("MKL_CBWR"),
                                 note="host already computes the AMD reference numerics; nothing patched")
    tables = TABLE_PATHS[target].read_bytes()
    if hashlib.sha256(tables).hexdigest() != TABLE_SHA256[target]:
        raise RuntimeError(f"Emulation table {TABLE_PATHS[target]} is corrupted.")
    report = CpuNumericsReport(target, vendor, True, os.environ.get("MKL_CBWR"))
    report.libraries.append(_install_library("libtorch_cpu.so", "torch", tables))
    report.libraries.append(_install_library("libsapien.so", "sapien", tables))
    _target_installed = target
    return report


def emulation_hit_counts() -> dict[str, int]:
    """Number of emulated executions per patched instruction (for diagnostics)."""

    if _helper is None:
        return {}
    counts: dict[str, int] = {}
    for spec, base in _installed:
        library_name = Path(spec["lib"]).name
        for window in spec["windows"]:
            for op in window["ops"]:
                if op["type"] == 1:
                    hits = int(_helper.detnum_hits(base + op["counter"]))
                    if hits:
                        counts[f"{library_name}:{hex(op['site'])}"] = hits
    return counts

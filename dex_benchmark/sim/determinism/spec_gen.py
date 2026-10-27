"""Find and describe patchable RCPPS/RSQRTPS sites in a shared library.

The output ("patch spec") lists non-overlapping patch windows. Each window starts
at a legacy-SSE ``rcpps``/``rsqrtps xmm, xmm`` instruction and extends over the
following instructions until at least 5 bytes (the size of the ``jmp rel32``)
are covered. Approximate instructions inside a window become emulation ops;
other instructions are relocated verbatim. RIP-relative operands are accepted
only when objdump's resolved target proves that the trailing 4 bytes are the
displacement. Windows containing branches or branch targets are rejected.

Requires GNU binutils (``objdump`` and ``nm``).
"""

from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

# Approximate-reciprocal instructions whose results differ between CPU vendors.
_APPROX_MNEMONIC = re.compile(r"v?(rcp|rsqrt)(14|28)?[ps]s")
_PATCHABLE = re.compile(r"(rcpps|rsqrtps)\s+xmm(\d+),xmm(\d+)$")
_INSTRUCTION_LINE = re.compile(r"\s+([0-9a-f]+):\t((?:[0-9a-f]{2} )+)\s*\t?(.*)")
_BRANCH = re.compile(r"(j\w+|call|loop\w*)\s+([0-9a-f]+)")
_MAX_OPS = 6
_MAX_WINDOW_BYTES = 48


def _require_tool(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise RuntimeError(f"{name!r} (GNU binutils) is required to generate a CPU-numerics patch spec.")
    return path


def _function_symbols(library: Path) -> list[tuple[int, int, str]]:
    nm = _require_tool("nm")
    symbols: list[tuple[int, int, str]] = []
    for extra in (["-D"], []):
        listing = subprocess.run(
            [nm, *extra, "--defined-only", "-S", str(library)], capture_output=True, text=True, check=False
        ).stdout
        for line in listing.splitlines():
            parts = line.split()
            if len(parts) == 4 and parts[2] in "tTwW":
                symbols.append((int(parts[0], 16), int(parts[1], 16), parts[3]))
    return symbols


def _disassemble(library: Path, start: int, stop: int) -> list[tuple[int, bytes, str]]:
    objdump = _require_tool("objdump")
    listing = subprocess.run(
        [objdump, "-d", "-M", "intel", f"--start-address={start}", f"--stop-address={stop}", str(library)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    merged: list[tuple[int, bytes, str]] = []
    for line in listing.splitlines():
        match = _INSTRUCTION_LINE.match(line)
        if match is None:
            continue
        address = int(match.group(1), 16)
        encoding = bytes.fromhex(match.group(2).replace(" ", ""))
        text = match.group(3).strip()
        # objdump wraps long encodings onto continuation lines without text.
        if merged and not text and address == merged[-1][0] + len(merged[-1][1]):
            previous = merged[-1]
            merged[-1] = (previous[0], previous[1] + encoding, previous[2])
        else:
            merged.append((address, encoding, text))
    return merged


def generate_patch_spec(library: str | Path, function_pattern: str | None = None, sites: list[int] | None = None) -> dict[str, Any]:
    """Return a patch spec for ``library``.

    Select functions either by a regular expression over symbol names or by the
    addresses of approximate instructions (``sites``). Unsupported or unsafe
    sites are listed under ``problems`` and never patched.
    """

    library = Path(library).resolve()
    if (function_pattern is None) == (sites is None):
        raise ValueError("Pass exactly one of function_pattern or sites.")
    file_bytes = library.read_bytes()
    symbols = _function_symbols(library)
    if sites is not None:
        selected: set[tuple[int, int, str]] = set()
        for site in sites:
            owners = [symbol for symbol in symbols if symbol[0] <= site < symbol[0] + symbol[1]]
            if not owners:
                raise ValueError(f"No function symbol contains 0x{site:x}.")
            selected.add(min(owners, key=lambda symbol: symbol[1]))
    else:
        pattern = re.compile(function_pattern or "")
        selected = {symbol for symbol in symbols if pattern.search(symbol[2])}

    windows: list[dict[str, Any]] = []
    problems: list[list[Any]] = []
    approx_sites: list[list[str]] = []
    covered: set[int] = set()
    counter = 0
    for start, size, name in sorted(selected):
        instructions = _disassemble(library, start, start + size + 32)
        branch_targets = {
            int(match.group(2), 16)
            for _, _, text in instructions
            if (match := _BRANCH.match(text)) is not None
        }
        for index, (address, encoding, text) in enumerate(instructions):
            mnemonic = text.split(" ")[0] if text else ""
            if not _APPROX_MNEMONIC.fullmatch(mnemonic) or not start <= address < start + size:
                continue
            approx_sites.append([hex(address), text, name])
            if address in covered:
                continue
            first = _PATCHABLE.match(text)
            if first is None:
                problems.append([hex(address), "unsupported form", text, name])
                continue
            ops: list[dict[str, Any]] = [
                {"type": 1, "kind": first.group(1), "dst": int(first.group(2)), "src": int(first.group(3)),
                 "counter": counter, "site": address}
            ]
            counter += 1
            members = [address]
            length = len(encoding)
            cursor = index + 1
            rejected = False
            while length < 5:
                next_address, next_encoding, next_text = instructions[cursor]
                nested = _PATCHABLE.match(next_text)
                if nested is not None:
                    ops.append({"type": 1, "kind": nested.group(1), "dst": int(nested.group(2)),
                                "src": int(nested.group(3)), "counter": counter, "site": next_address})
                    counter += 1
                    members.append(next_address)
                elif re.match(r"(j|call|ret|loop)", next_text):
                    problems.append([hex(address), "branch in window", next_text, name])
                    rejected = True
                    break
                else:
                    rip_relative = 0
                    if "rip" in next_text:
                        target = re.search(r"#\s*([0-9a-f]+)", next_text)
                        if target is None:
                            problems.append([hex(address), "rip operand without resolved target", next_text, name])
                            rejected = True
                            break
                        displacement = int(target.group(1), 16) - (next_address + len(next_encoding))
                        if next_encoding[-4:] != (displacement & 0xFFFFFFFF).to_bytes(4, "little"):
                            problems.append([hex(address), "rip displacement is not trailing", next_text, name])
                            rejected = True
                            break
                        rip_relative = 1
                    if len(next_encoding) > 16:
                        problems.append([hex(address), "instruction too long", next_text, name])
                        rejected = True
                        break
                    ops.append({"type": 0, "off": next_address, "bytes": next_encoding.hex(), "rip": rip_relative,
                                "text": next_text})
                length += len(next_encoding)
                cursor += 1
            if rejected:
                continue
            inner_targets = [target for target in branch_targets if address < target < address + length]
            if inner_targets:
                problems.append([hex(address), "branch target inside window", [hex(t) for t in inner_targets], name])
                continue
            if len(ops) > _MAX_OPS or length > _MAX_WINDOW_BYTES:
                problems.append([hex(address), "window too large", name])
                continue
            covered.update(members)
            windows.append({"off": address, "patch_len": length, "orig": file_bytes[address:address + length].hex(),
                            "ops": ops, "function": name})
    return {
        "lib": str(library),
        "sha256": hashlib.sha256(file_bytes).hexdigest(),
        "functions": sorted(symbol[2] for symbol in selected),
        "all_approx_sites": approx_sites,
        "windows": windows,
        "counters": counter,
        "problems": problems,
    }

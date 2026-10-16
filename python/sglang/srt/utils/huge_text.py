"""Run a process's machine code from transparent huge pages.

Without CONFIG_READ_ONLY_THP_FOR_FS, executable file mappings stay on 4 KiB page-cache pages. In a
confidential VM every iTLB miss then walks the guest page tables nested in the secure EPT, and the
scheduler's large code footprint (Python, libtorch, CUDA) runs about twice as slowly as on the host.
Copying the 2 MiB-aligned interior of each executable mapping into anonymous THP memory and moving
it back over the same address keeps the bytes and addresses while removing those misses.
"""

import ctypes
import logging

logger = logging.getLogger(__name__)

_HP = 2 << 20
_PROT_READ, _PROT_WRITE, _PROT_EXEC = 1, 2, 4
_MAP_PRIVATE, _MAP_ANONYMOUS = 0x02, 0x20
_MADV_HUGEPAGE = 14
_MREMAP_MAYMOVE, _MREMAP_FIXED = 1, 2
_MAP_FAILED = ctypes.c_void_p(-1).value


def _libc():
    libc = ctypes.CDLL(None, use_errno=True)
    vp, sz = ctypes.c_void_p, ctypes.c_size_t
    libc.mmap.restype = vp
    libc.mmap.argtypes = [
        vp,
        sz,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_long,
    ]
    libc.mremap.restype = vp
    libc.mremap.argtypes = [vp, sz, sz, ctypes.c_int, vp]
    libc.munmap.argtypes = [vp, sz]
    libc.mprotect.argtypes = [vp, sz, ctypes.c_int]
    libc.madvise.argtypes = [vp, sz, ctypes.c_int]
    return libc


def _executable_ranges():
    with open("/proc/self/maps") as f:
        for line in f:
            fields = line.split()
            if len(fields) < 6 or fields[1] != "r-xp" or not fields[5].startswith("/"):
                continue
            start, end = (int(x, 16) for x in fields[0].split("-"))
            lo, hi = -(-start // _HP) * _HP, end // _HP * _HP
            if hi > lo:
                yield lo, hi


def remap_text_to_huge_pages() -> int:
    """Returns the number of bytes moved onto huge pages."""
    libc = _libc()
    moved = 0
    for lo, hi in list(_executable_ranges()):
        size = hi - lo
        raw = libc.mmap(
            None,
            size + _HP,
            _PROT_READ | _PROT_WRITE,
            _MAP_PRIVATE | _MAP_ANONYMOUS,
            -1,
            0,
        )
        if raw == _MAP_FAILED:
            continue
        new = -(-raw // _HP) * _HP
        if new > raw:
            libc.munmap(raw, new - raw)
        if raw + _HP > new:
            libc.munmap(new + size, raw + _HP - new)
        libc.madvise(new, size, _MADV_HUGEPAGE)
        ctypes.memmove(new, lo, size)
        if (
            libc.mprotect(new, size, _PROT_READ | _PROT_EXEC) != 0
            or libc.mremap(new, size, size, _MREMAP_MAYMOVE | _MREMAP_FIXED, lo)
            == _MAP_FAILED
        ):
            libc.munmap(new, size)
            continue
        moved += size
    logger.info(f"Moved {moved >> 20} MiB of executable mappings onto huge pages")
    return moved

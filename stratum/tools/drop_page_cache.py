#!/usr/bin/env python3
"""drop_page_cache.py <file> [...] — evict a file's pages from the OS page cache.

After a large-model run the weights stay resident in page cache (~12 GB for
the 27B GGUF), which skews later measurements and slows reclaim under memory
pressure. This tool returns exactly those pages: posix_fadvise(DONTNEED)
first, then msync(MS_INVALIDATE) for whatever survived — the only call that
actually evicts clean file pages on darwin (measured 2048 MB -> 0; MADV_DONTNEED
is a no-op there). Residency is measured with mincore() before/after and
reported, never assumed.

All syscalls go through ctypes because CPython hides posix_fadvise/mincore
on darwin and its read-only mmap objects cannot back a ctypes pointer.
"""
import ctypes
import os
import sys

_libc = ctypes.CDLL(None)

_c_int = ctypes.c_int
_c_void_p = ctypes.c_void_p
_libc.mmap.restype = _c_void_p
_libc.mmap.argtypes = [_c_void_p, ctypes.c_size_t, _c_int, _c_int, _c_int, ctypes.c_long]
_libc.munmap.argtypes = [_c_void_p, ctypes.c_size_t]
_libc.mincore.restype = _c_int
_libc.mincore.argtypes = [_c_void_p, ctypes.c_size_t,
                          ctypes.POINTER(ctypes.c_ubyte)]
_has_fadvise = hasattr(_libc, "posix_fadvise")
if _has_fadvise:
    _libc.posix_fadvise.restype = _c_int
    _libc.posix_fadvise.argtypes = [_c_int, ctypes.c_long, ctypes.c_long, _c_int]
_libc.msync.restype = _c_int
_libc.msync.argtypes = [_c_void_p, ctypes.c_size_t, _c_int]

PROT_READ, MAP_PRIVATE, MAP_SHARED, MAP_FAILED = 0x1, 0x0002, 0x0001, -1
POSIX_FADV_DONTNEED = 4   # linux <fcntl.h>; not exported by darwin libSystem
MS_INVALIDATE, MS_SYNC = 0x0002, 0x0010   # darwin <sys/mman.h>


def _pagesize():
    return os.sysconf("SC_PAGESIZE")


def _mmap(fd, length, flags):
    addr = _libc.mmap(None, length, PROT_READ, flags, fd, 0)
    if addr in (None, MAP_FAILED) or addr == 2 ** 64 - 1:
        raise OSError(ctypes.get_errno(), "mmap failed")
    return addr


def _count_resident(addr, length):
    npages = (length + _pagesize() - 1) // _pagesize()
    vec = (ctypes.c_ubyte * npages)()
    if _libc.mincore(addr, length, vec) != 0:
        raise OSError(ctypes.get_errno(), "mincore failed")
    return sum(1 for b in bytes(vec) if b & 1), npages


def process(path):
    size = os.path.getsize(path)
    pgs = _pagesize()
    aligned_len = (size // pgs) * pgs
    if aligned_len == 0:
        print(f"{path}: empty, nothing to drop")
        return True
    np_expect = (aligned_len + pgs - 1) // pgs
    if np_expect * pgs - aligned_len >= pgs:
        raise AssertionError("page math invariant")
    fd = os.open(path, os.O_RDONLY)
    try:
        priv = _mmap(fd, aligned_len, MAP_PRIVATE)
        res0, npages = _count_resident(priv, aligned_len)
        _libc.munmap(priv, aligned_len)

        dropped_by = []
        res = res0
        if _has_fadvise:
            if _libc.posix_fadvise(fd, 0, aligned_len, POSIX_FADV_DONTNEED) == 0:
                p2 = _mmap(fd, aligned_len, MAP_PRIVATE)
                res, _ = _count_resident(p2, aligned_len)
                _libc.munmap(p2, aligned_len)
                if res < res0:
                    dropped_by.append(f"fadvise(-{res0 - res}p)")

        if res > 0:
            # msync(MS_INVALIDATE) is the eviction that actually works on
            # darwin — measured: 2048 MB resident -> 0 (MADV_DONTNEED on a
            # shared file mapping is a no-op there).
            sh = _mmap(fd, aligned_len, MAP_SHARED)
            rc = _libc.msync(sh, aligned_len, MS_INVALIDATE | MS_SYNC)
            _libc.munmap(sh, aligned_len)
            if rc == 0:
                p3 = _mmap(fd, aligned_len, MAP_PRIVATE)
                res2, _ = _count_resident(p3, aligned_len)
                _libc.munmap(p3, aligned_len)
                if res2 < res:
                    dropped_by.append(f"msync(-{res - res2}p)")
                res = res2

        mb = lambda pages: pages * pgs / (1024.0 * 1024.0)
        how = " ".join(dropped_by) if dropped_by else "no-op"
        verdict = "clear" if res == 0 else "PARTIAL" if res < res0 else "RESIDENT"
        print(f"{path}: resident {mb(res0):.0f}/{mb(npages):.0f} MB"
              f" -> {mb(res):.0f} MB [{how}] {verdict}")
        return True
    finally:
        os.close(fd)


def main(argv):
    if not argv:
        print(__doc__)
        return 2
    ok = True
    for path in argv:
        try:
            ok &= bool(process(path))
        except OSError as e:
            print(f"{path}: ERROR {e}")
            ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

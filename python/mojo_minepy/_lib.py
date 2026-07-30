"""ctypes bindings for the Mojo characteristic-matrix kernel."""

from __future__ import annotations

import ctypes
import os
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB = os.environ.get("MOJO_MINEPY_LIB") or os.path.join(
    ROOT, "dist", "libmojo-minepy.so"
)

I = ctypes.c_int64
F = ctypes.c_double

_lib = None


def build() -> str:
    source = os.path.join(ROOT, "src", "mine.mojo")
    stale = not os.path.exists(LIB)
    if not os.environ.get("MOJO_MINEPY_LIB") and not stale:
        stale = os.path.getmtime(source) > os.path.getmtime(LIB)
    if stale:
        subprocess.run(
            ["bash", os.path.join(ROOT, "build", "build.sh")],
            cwd=ROOT,
            check=True,
            timeout=1800,
        )
    return LIB


def lib() -> ctypes.CDLL:
    global _lib
    if _lib is None:
        _lib = ctypes.CDLL(build())
        fn = _lib.mine_compute_matrix
        fn.argtypes = [I] * 8 + [F, I] + [I] * 19
        fn.restype = I
    return _lib


def addr(array: np.ndarray) -> int:
    if not isinstance(array, np.ndarray):
        raise TypeError("FFI buffers must be NumPy arrays")
    if not array.flags.c_contiguous:
        raise ValueError("FFI buffers must be C-contiguous")
    address = int(array.ctypes.data)
    if address == 0:
        raise ValueError("FFI buffers must have non-null data pointers")
    return address

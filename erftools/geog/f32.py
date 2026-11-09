"""Single-precision arithmetic helpers.

The ``wps_equivalent`` code paths reproduce geogrid's REAL*4 arithmetic.  Basic
operations (+ - * / sqrt) on numpy float32 arrays are IEEE-754 correctly
rounded, exactly like the SSE instructions gfortran emits.  Transcendental
functions are NOT guaranteed to agree between numpy's SIMD kernels and glibc,
so here they are evaluated with the C library's float functions (sinf, cosf,
tanf, atanf, atan2f, powf, log10f) through ctypes -- the same functions a
gfortran REAL*4 intrinsic call resolves to at run time on this system.

If libm cannot be loaded (non-glibc platform) the module falls back to numpy
float32 ufuncs; results then agree to ~1 ulp instead of bit-for-bit.
"""
import ctypes
import ctypes.util

import numpy as np

F = np.float32

_libm = None
try:
    _libm = ctypes.CDLL(ctypes.util.find_library("m") or "libm.so.6")
    for _n in ("sinf", "cosf", "tanf", "atanf", "log10f"):
        getattr(_libm, _n).restype = ctypes.c_float
        getattr(_libm, _n).argtypes = [ctypes.c_float]
    for _n in ("atan2f", "powf"):
        getattr(_libm, _n).restype = ctypes.c_float
        getattr(_libm, _n).argtypes = [ctypes.c_float, ctypes.c_float]
    HAVE_LIBM = True
except (OSError, AttributeError):  # pragma: no cover
    HAVE_LIBM = False


def _wrap1(name, npfunc):
    if HAVE_LIBM:
        cfun = getattr(_libm, name)
        uf = np.frompyfunc(cfun, 1, 1)

        def f(x):
            x = np.asarray(x, dtype=F)
            if x.ndim == 0:
                return F(cfun(float(x)))
            return uf(x).astype(F)
    else:
        def f(x):
            return npfunc(np.asarray(x, dtype=F)).astype(F)
    f.__name__ = name
    return f


def _wrap2(name, npfunc):
    if HAVE_LIBM:
        cfun = getattr(_libm, name)
        uf = np.frompyfunc(cfun, 2, 1)

        def f(x, y):
            x = np.asarray(x, dtype=F)
            y = np.asarray(y, dtype=F)
            if x.ndim == 0 and y.ndim == 0:
                return F(cfun(float(x), float(y)))
            return uf(x, y).astype(F)
    else:
        def f(x, y):
            return npfunc(np.asarray(x, dtype=F), np.asarray(y, dtype=F)).astype(F)
    f.__name__ = name
    return f


sin = _wrap1("sinf", np.sin)
cos = _wrap1("cosf", np.cos)
tan = _wrap1("tanf", np.tan)
atan = _wrap1("atanf", np.arctan)
log10 = _wrap1("log10f", np.log10)
atan2 = _wrap2("atan2f", np.arctan2)
power = _wrap2("powf", np.power)

# constants_module.F (folded at compile time in REAL*4)
PI = F(3.141592653589793)
OMEGA_E = F(7.292e-5)
DEG_PER_RAD = F(180.0) / PI
RAD_PER_DEG = PI / F(180.0)
EARTH_RADIUS_M = F(6370000.0)


def nint(x):
    """Fortran NINT (round half away from zero), exact for float32 input."""
    x = np.asarray(x, dtype=np.float64)
    return (np.sign(x) * np.floor(np.abs(x) + 0.5)).astype(np.int64)


def f32(s):
    """Parse a decimal string (or number) to float32 like a Fortran list read."""
    return F(np.float64(float(s)))

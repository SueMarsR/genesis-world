#!/usr/bin/env python3
"""
Map EGL device IDs <-> CUDA ordinals, and pick the right EGL_DEVICE_ID so
Genesis' offscreen EGL renderer lands on the same physical GPU as your CUDA
physics backend.

Why this exists
---------------
`EGL_DEVICE_ID` indexes EGL's own device enumeration (eglQueryDevicesEXT),
which is NOT guaranteed to match CUDA ordinals — and CUDA ordinals are further
remapped by CUDA_VISIBLE_DEVICES. To bind rendering to a specific card you must
translate: (your CUDA ordinal) -> (physical GPU via UUID) -> (EGL_DEVICE_ID).

On this H20 box with CUDA_VISIBLE_DEVICES unset they happen to be identity
(EGL i == CUDA i), but this script stays correct when that env var is set.

Usage
-----
    python egl_cuda_map.py            # print the full mapping table
    python egl_cuda_map.py --cuda 3   # print EGL_DEVICE_ID for CUDA ordinal 3
    EGL_DEVICE_ID=$(python egl_cuda_map.py --cuda 3 --quiet)
"""
import argparse
import ctypes as C
import os
import sys

EGL_EXTENSIONS = 0x3055
EGL_CUDA_DEVICE_NV = 0x323A


def _egl():
    egl = C.CDLL("libEGL.so.1")
    egl.eglGetProcAddress.restype = C.c_void_p
    egl.eglGetProcAddress.argtypes = [C.c_char_p]
    return egl


def _proc(egl, name, restype, argtypes):
    a = egl.eglGetProcAddress(name.encode())
    return C.CFUNCTYPE(restype, *argtypes)(a) if a else None


def egl_to_cuda():
    """Return list where index = EGL_DEVICE_ID, value = CUDA ordinal.

    NOTE: EGL_CUDA_DEVICE_NV already honours CUDA_VISIBLE_DEVICES — it returns
    the ordinal *as your process sees it*, not the raw hardware ordinal. So the
    value here can be compared directly against the ordinal you pass to
    torch.cuda / gs.init, with no extra remapping. Devices hidden by
    CUDA_VISIBLE_DEVICES report None."""
    egl = _egl()
    QueryDevices = _proc(egl, "eglQueryDevicesEXT", C.c_uint,
                         [C.c_int, C.POINTER(C.c_void_p), C.POINTER(C.c_int)])
    QueryDeviceString = _proc(egl, "eglQueryDeviceStringEXT", C.c_char_p,
                              [C.c_void_p, C.c_int])
    QueryDeviceAttrib = _proc(egl, "eglQueryDeviceAttribEXT", C.c_uint,
                              [C.c_void_p, C.c_int, C.POINTER(C.c_ssize_t)])
    if not QueryDevices:
        sys.exit("EGL_EXT_device_base not available")
    n = C.c_int(0)
    QueryDevices(0, None, C.byref(n))
    devs = (C.c_void_p * n.value)()
    got = C.c_int(0)
    QueryDevices(n.value, devs, C.byref(got))
    out = []
    for i in range(n.value):
        ext = QueryDeviceString(devs[i], EGL_EXTENSIONS)
        ext = ext.decode() if ext else ""
        cuda = None
        if "EGL_NV_device_cuda" in ext and QueryDeviceAttrib:
            v = C.c_ssize_t(-1)
            if QueryDeviceAttrib(devs[i], EGL_CUDA_DEVICE_NV, C.byref(v)) and v.value >= 0:
                cuda = v.value
        out.append(cuda)
    return out


def egl_id_for_cuda(cuda_ordinal):
    """Given the CUDA ordinal your code uses (as seen after
    CUDA_VISIBLE_DEVICES), return the EGL_DEVICE_ID for the same physical GPU.
    EGL_CUDA_DEVICE_NV already accounts for CUDA_VISIBLE_DEVICES, so this is a
    direct lookup."""
    e2c = egl_to_cuda()
    for egl_id, c in enumerate(e2c):
        if c == cuda_ordinal:
            return egl_id
    raise ValueError(f"no EGL device maps to (visible) CUDA ordinal "
                     f"{cuda_ordinal}; EGL->CUDA table is {e2c}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cuda", type=int, help="CUDA ordinal to resolve to EGL_DEVICE_ID")
    ap.add_argument("--quiet", action="store_true", help="print only the number")
    args = ap.parse_args()

    if args.cuda is not None:
        egl_id = egl_id_for_cuda(args.cuda)
        print(egl_id if args.quiet else
              f"CUDA ordinal {args.cuda} -> EGL_DEVICE_ID={egl_id}")
        return

    e2c = egl_to_cuda()
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','<unset>')}")
    print(f"{'EGL_DEVICE_ID':>13} | {'CUDA ordinal':>12}")
    print("-" * 30)
    for egl_id, c in enumerate(e2c):
        print(f"{egl_id:>13} | {('hidden' if c is None else c):>12}")


if __name__ == "__main__":
    main()

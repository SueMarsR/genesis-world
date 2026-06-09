#!/usr/bin/env python3
"""
Patch quadrants' embedded graph_do_while condition-kernel fatbin so it loads
on this host's driver.

Root cause
----------
quadrants 1.0.0 ships `quadrants_python.cpython-312-x86_64-linux-gnu.so` with a
hard-embedded fatbin (symbol `_ZL22kConditionKernelFatbin`) for the
`_qd_graph_do_while_cond` kernel, built with a CUDA 13.1 toolchain -> ELF
ABI=8. The host driver (570.133.20 / CUDA 12.8) only loads ABI<=7, so
`cuModuleLoadFatBinary` returns CUDA error 200 and GPU rigid solving crashes
with "Failed to load graph_do_while condition kernel fatbin".

Fix
---
Rebuild the same kernel with the local CUDA 12.8 nvcc (-> ABI=7) and overwrite
the embedded fatbin in place. The slot is 24360 bytes; the rebuilt fatbin is
~3.9KB and is zero-padded to keep the symbol size (and thus the whole ELF
layout) unchanged. The kernel's exact semantics matter:

    extern "C" __global__ void _qd_graph_do_while_cond(
            cudaGraphConditionalHandle handle, int** pflag) {
        cudaGraphSetConditional(handle, (**pflag) != 0 ? 1u : 0u);
    }

NOTE the parameter is `int**` (double indirection) — matching the original
SASS (`LDG.E.64` then `LD.E`). A single-`int*` version compiles and loads but
reads a pointer value as the counter, which is always non-zero -> the solver
loop never terminates (100% GPU spin, no crash). cond2_new.fatbin in this dir
is the correct prebuilt artifact.

Re-run this after any `pip install/upgrade quadrants`, which restores the
ABI=8 .so.
"""
import os, struct, shutil, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
SLOT_VADDR = 0x4E0DF40          # == file offset (PT_LOAD offset==vaddr); re-derive if version changes
SLOT_SIZE = 24360               # size of _ZL22kConditionKernelFatbin
FATBIN_MAGIC = bytes.fromhex("50ed55ba")


def find_so():
    import importlib.util
    spec = importlib.util.find_spec("quadrants")
    base = os.path.dirname(spec.origin)
    so = os.path.join(base, "_lib", "core",
                      "quadrants_python.cpython-312-x86_64-linux-gnu.so")
    if not os.path.exists(so):
        sys.exit(f"quadrants .so not found at {so}")
    return so


def build_fatbin():
    """Build the ABI=7 fatbin locally; fall back to the bundled artifact."""
    cu = os.path.join(HERE, "cond2.cu")
    cubin = "/tmp/_cond2.cubin"
    out = "/tmp/_cond2_new.fatbin"
    try:
        subprocess.run(["nvcc", "-arch=sm_90", "-cubin", cu, "-o", cubin],
                       check=True)
        subprocess.run(["fatbinary", f"--create={out}",
                        f"--image=profile=sm_90,file={cubin}"], check=True)
        return out
    except Exception as e:
        prebuilt = os.path.join(HERE, "cond2_new.fatbin")
        print(f"local build failed ({e}); using prebuilt {prebuilt}")
        return prebuilt


def main():
    so = find_so()
    bak = so + ".bak"
    if not os.path.exists(bak):
        shutil.copy2(so, bak)
        print(f"backup -> {bak}")

    fatbin = build_fatbin()
    new = open(fatbin, "rb").read()
    if len(new) > SLOT_SIZE:
        sys.exit(f"rebuilt fatbin {len(new)} > slot {SLOT_SIZE}")
    payload = new + b"\x00" * (SLOT_SIZE - len(new))

    with open(so, "r+b") as f:
        f.seek(SLOT_VADDR)
        if f.read(4) != FATBIN_MAGIC:
            sys.exit("slot does not start with fatbin magic — wrong offset / "
                     "quadrants version changed; re-derive SLOT_VADDR from "
                     "`readelf -sW <so> | grep kConditionKernelFatbin`")
        f.seek(SLOT_VADDR)
        f.write(payload)
    print(f"patched {os.path.basename(so)}: wrote {len(new)} B + "
          f"{SLOT_SIZE - len(new)} B pad (ABI=7 sm_90)")
    print("verify: gs.init(backend=gs.gpu); scene.step() should no longer "
          "raise CUDA error 200.")


if __name__ == "__main__":
    main()

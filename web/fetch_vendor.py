#!/usr/bin/env python3
"""Fetches the ONNX Runtime Web files the demo needs into web/vendor/.

    python web/fetch_vendor.py

Not committed to the repository. The wasm binary alone is 12.9 MB, which is heavy
for git history when it is a pinned third-party artifact that any build can
re-download. CI fetches it and includes it in the Pages bundle, so the deployed
site is self-contained even though the repository is not.

That is the opposite of the rule for model weights, which *are* committed: Git LFS
serves pointer files on GitHub Pages, so a quantised network has to be an ordinary
file in the repo. The difference is that weights are ours and change with training,
while this is a versioned dependency.

Three files:

  ort.wasm.bundle.min.mjs        the wasm-only backend
  ort-wasm-simd-threaded.mjs     its loader, which the bundle imports dynamically
  ort-wasm-simd-threaded.wasm    the kernels

The loader is separate despite the "bundle" in the first name -- that variant bundles
the backend glue, not the wasm loader. Omitting it produces "no available backend
found" at session creation and nothing earlier, which is a confusing way to discover a
missing file on a deployed site.

The wasm-only build is deliberate. `ort.all` carries WebGPU and WebGL backends the
demo does not use, and every unused byte is subtracted from a phone's patience.

The `-threaded` name is not a mistake and not a violation of the single-threaded
rule: upstream ships one binary that adapts, and it runs single-threaded when
SharedArrayBuffer is absent. The demo pins `ort.env.wasm.numThreads = 1` so the
behaviour is chosen rather than inherited from whether a header happened to be set.
"""

from __future__ import annotations

import hashlib
import pathlib
import sys
import urllib.request

# Pinned rather than tracking latest, for the same reason DESIGN.md pins HUGO_VERSION.
# Note this trails the native runtime in tools/ (1.28.0): onnxruntime-web is published
# separately and 1.28.0 does not exist for it. That skew is exactly why the parity
# fixture compares both runtimes against PyTorch rather than against each other.
ORT_WEB_VERSION = "1.27.0"

BASE = f"https://cdn.jsdelivr.net/npm/onnxruntime-web@{ORT_WEB_VERSION}/dist"
FILES = (
    "ort.wasm.bundle.min.mjs",
    "ort-wasm-simd-threaded.mjs",
    "ort-wasm-simd-threaded.wasm",
)

VENDOR = pathlib.Path(__file__).resolve().parent / "vendor"


def fetch(name: str) -> pathlib.Path:
    target = VENDOR / name
    if target.exists():
        print(f"  {name}: already present ({target.stat().st_size / 1e6:.2f} MB)")
        return target

    url = f"{BASE}/{name}"
    print(f"  {name}: downloading")
    with urllib.request.urlopen(url, timeout=300) as response:
        payload = response.read()
    target.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()[:16]
    print(f"    {len(payload) / 1e6:.2f} MB  sha256:{digest}")
    return target


def main() -> int:
    VENDOR.mkdir(parents=True, exist_ok=True)
    print(f"onnxruntime-web {ORT_WEB_VERSION} -> {VENDOR}")
    for name in FILES:
        fetch(name)
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())

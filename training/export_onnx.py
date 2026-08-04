"""Export a trained HexNet to ONNX and verify the exported graph numerically.

The verification is the point. An export that silently changes behaviour — from
BatchNorm left in training mode, a fused operator that rounds differently, or a
shape inferred wrong — produces a model that loads fine in the browser and plays
badly, with nothing in the logs to suggest why. Comparing outputs on random
inputs before shipping costs a second and removes the whole failure class.

    python export_onnx.py --checkpoint runs/best.pt --output web/hex9.onnx
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np
import torch

from model import HexNet, build

# Torch emits opset 18 natively; requesting anything lower triggers a
# down-conversion fallback that does no useful work and is one more thing to
# break on a future release. ONNX Runtime Web handles 18 comfortably.
OPSET = 18


def export(model: HexNet, path: pathlib.Path, batch: int = 1) -> None:
    model.eval()
    dummy = torch.zeros(
        batch, model.input_planes, model.board_size, model.board_size
    )
    path.parent.mkdir(parents=True, exist_ok=True)

    # Self-play batches many leaves at once; the browser evaluates a single
    # position at a time. One file has to serve both, so the batch axis is
    # dynamic. The key names the forward() argument, not the ONNX input.
    batch = torch.export.Dim("batch", min=1, max=8192)

    torch.onnx.export(
        model,
        (dummy,),
        str(path),
        input_names=["board"],
        output_names=["policy", "value"],
        dynamic_shapes={"x": {0: batch}},
        opset_version=OPSET,
        dynamo=True,
        # Keep the weights inside the .onnx file. The exporter otherwise writes
        # a sidecar .onnx.data, which is an easy way to deploy a model that
        # loads locally and fails in the browser with only a fetch error to go
        # on. A quantised network is small enough that one file is fine.
        external_data=False,
    )


def verify(
    model: HexNet, path: pathlib.Path, trials: int = 8, tolerance: float = 1e-4
) -> tuple[float, float]:
    """Compare ONNX Runtime against PyTorch on random inputs.

    Returns the worst absolute difference seen for the policy and the value.
    Raises if either exceeds the tolerance.
    """
    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    model.eval()

    worst_policy = 0.0
    worst_value = 0.0
    generator = torch.Generator().manual_seed(20260804)

    for trial in range(trials):
        # Vary batch size so the dynamic axis is genuinely exercised.
        batch = 1 + trial
        planes = torch.randint(
            0,
            2,
            (batch, model.input_planes, model.board_size, model.board_size),
            generator=generator,
        ).float()

        with torch.no_grad():
            reference_policy, reference_value = model(planes)

        actual_policy, actual_value = session.run(
            ["policy", "value"], {"board": planes.numpy()}
        )

        worst_policy = max(
            worst_policy,
            float(np.abs(reference_policy.numpy() - actual_policy).max()),
        )
        worst_value = max(
            worst_value,
            float(np.abs(reference_value.numpy() - actual_value).max()),
        )

    if worst_policy > tolerance or worst_value > tolerance:
        raise AssertionError(
            f"ONNX output diverges from PyTorch: policy {worst_policy:.2e}, "
            f"value {worst_value:.2e} (tolerance {tolerance:.0e})"
        )
    return worst_policy, worst_value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--board-size", type=int, default=9)
    parser.add_argument("--preset", default="full", choices=["tiny", "light", "full"])
    parser.add_argument("--checkpoint", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path("hex9.onnx"))
    args = parser.parse_args()

    model = build(args.board_size, args.preset)
    if args.checkpoint is not None:
        state = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(state["model"] if "model" in state else state)

    export(model, args.output)
    worst_policy, worst_value = verify(model, args.output)

    sidecar = args.output.with_suffix(args.output.suffix + ".data")
    if sidecar.exists():
        raise AssertionError(
            f"weights were written externally to {sidecar}; the browser build "
            "expects a single self-contained file"
        )

    size_mb = args.output.stat().st_size / (1024 * 1024)
    print(model.describe())
    print(f"wrote {args.output} ({size_mb:.2f} MB)")
    print(
        f"onnx matches pytorch: policy max diff {worst_policy:.2e}, "
        f"value max diff {worst_value:.2e}"
    )


if __name__ == "__main__":
    main()

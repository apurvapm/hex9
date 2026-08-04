"""Tests for the network and its ONNX export."""

from __future__ import annotations

import pathlib
import tempfile

import numpy as np
import pytest
import torch

from model import NON_NEIGHBOUR_TAPS, HexConv2d, HexNet, build


def test_output_shapes():
    model = HexNet(board_size=9, channels=16, blocks=2)
    planes = torch.zeros(4, 3, 9, 9)
    policy, value = model(planes)

    assert policy.shape == (4, 82), "policy must cover every cell plus swap"
    assert value.shape == (4,), "value is one scalar per position"


def test_value_is_bounded():
    model = HexNet(board_size=7, channels=16, blocks=2)
    planes = torch.randint(0, 2, (32, 3, 7, 7)).float()
    _, value = model(planes)
    assert torch.all(value >= -1.0) and torch.all(value <= 1.0)


def test_masked_taps_stay_zero_and_receive_no_gradient():
    """The two non-adjacent kernel positions must be inert, before and after
    a backward pass. If they picked up gradient the mask would be cosmetic."""
    conv = HexConv2d(3, 4, masked=True)
    effective = conv.weight * conv.mask
    for row, col in NON_NEIGHBOUR_TAPS:
        assert torch.all(effective[:, :, row, col] == 0.0)

    out = conv(torch.randn(2, 3, 5, 5))
    out.sum().backward()
    for row, col in NON_NEIGHBOUR_TAPS:
        assert torch.all(conv.weight.grad[:, :, row, col] == 0.0), (
            "masked taps must not accumulate gradient"
        )


def test_unmasked_variant_uses_all_taps():
    conv = HexConv2d(3, 4, masked=False)
    assert torch.all(conv.mask == 1.0)


def test_mask_changes_the_function():
    """Sanity check that masking is not a no-op: the same weights with and
    without the mask must produce different outputs."""
    torch.manual_seed(0)
    masked = HexConv2d(1, 1, masked=True)
    unmasked = HexConv2d(1, 1, masked=False)
    unmasked.load_state_dict(
        {"weight": masked.weight.detach().clone(), "mask": unmasked.mask}
    )
    x = torch.randn(1, 1, 6, 6)
    assert not torch.allclose(masked(x), unmasked(x))


def test_deterministic_initialisation():
    torch.manual_seed(1234)
    a = HexNet(board_size=5, channels=8, blocks=1)
    torch.manual_seed(1234)
    b = HexNet(board_size=5, channels=8, blocks=1)

    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.equal(pa, pb)


def test_presets_are_small_enough_to_ship():
    tiny = build(5, "tiny")
    full = build(9, "full")

    assert tiny.policy_size == 26
    assert full.policy_size == 82
    # float32 megabytes; int8 quantisation cuts this by roughly four.
    full_mb = full.parameter_count() * 4 / (1024 * 1024)
    assert full_mb < 4.0, f"full preset is {full_mb:.1f} MB before quantisation"


def test_light_preset_is_cheaper():
    """The light preset exists for CPU-only training, so it has to be
    materially smaller rather than nominally different."""
    light = build(9, "light")
    full = build(9, "full")
    assert light.parameter_count() < full.parameter_count() / 2


def test_batch_independence():
    """Positions in a batch must not influence one another. In eval mode
    BatchNorm uses running statistics, so a single position and that same
    position inside a larger batch must score identically."""
    model = HexNet(board_size=7, channels=16, blocks=2)
    model.eval()

    batch = torch.randint(0, 2, (5, 3, 7, 7)).float()
    with torch.no_grad():
        batched_policy, batched_value = model(batch)
        single_policy, single_value = model(batch[2:3])

    assert torch.allclose(batched_policy[2], single_policy[0], atol=1e-6)
    assert torch.allclose(batched_value[2], single_value[0], atol=1e-6)


def test_onnx_export_matches_pytorch():
    ort = pytest.importorskip("onnxruntime")
    from export_onnx import export, verify

    model = build(5, "tiny")
    with tempfile.TemporaryDirectory() as directory:
        path = pathlib.Path(directory) / "tiny.onnx"
        export(model, path)
        assert path.exists() and path.stat().st_size > 0

        worst_policy, worst_value = verify(model, path)
        assert worst_policy < 1e-4
        assert worst_value < 1e-4

        sidecar = path.with_suffix(path.suffix + ".data")
        assert not sidecar.exists(), (
            "weights must stay inside the .onnx file; a sidecar deploys "
            "cleanly on disk and fails in the browser"
        )


def test_onnx_accepts_variable_batch():
    ort = pytest.importorskip("onnxruntime")
    from export_onnx import export

    model = build(5, "tiny")
    model.eval()
    with tempfile.TemporaryDirectory() as directory:
        path = pathlib.Path(directory) / "tiny.onnx"
        export(model, path)
        session = ort.InferenceSession(
            str(path), providers=["CPUExecutionProvider"]
        )
        # Self-play batches leaves; the browser evaluates one at a time.
        for batch in (1, 3, 64):
            planes = np.zeros((batch, 3, 5, 5), dtype=np.float32)
            policy, value = session.run(["policy", "value"], {"board": planes})
            assert policy.shape == (batch, 26)
            assert value.shape == (batch,)

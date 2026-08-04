"""Policy and value network for hex9.

Input is the three-plane canonical encoding from encoding.py: the mover's
stones, the opponent's stones, and a plane set when swap is legal. Output is a
policy over N*N + 1 actions (every cell plus the swap sentinel) and a scalar
value in [-1, 1] from the mover's perspective.

The one Hex-specific choice here is the convolution kernel. A cell on the
rhombus has six neighbours, at offsets (-1,0) (-1,+1) (0,-1) (0,+1) (+1,-1)
(+1,0). A standard 3x3 kernel spans eight, so two of its taps — (-1,-1) and
(+1,+1) — sit on cells that are not adjacent at all. Masking those two weights
to zero gives the network the true board topology instead of asking it to learn
that two of its inputs are meaningless. The mask is optional so the choice can
be ablated.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Kernel positions that are NOT hex neighbours, in (row, col) kernel space where
# the centre tap is (1, 1).
NON_NEIGHBOUR_TAPS = ((0, 0), (2, 2))


class HexConv2d(nn.Conv2d):
    """3x3 convolution with the two non-adjacent corner taps held at zero."""

    def __init__(self, in_channels: int, out_channels: int, masked: bool = True):
        super().__init__(
            in_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        mask = torch.ones(3, 3)
        if masked:
            for row, col in NON_NEIGHBOUR_TAPS:
                mask[row, col] = 0.0
        self.register_buffer("mask", mask.view(1, 1, 3, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(
            x, self.weight * self.mask, self.bias, self.stride, self.padding
        )


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, masked: bool = True):
        super().__init__()
        self.conv1 = HexConv2d(channels, channels, masked)
        self.norm1 = nn.BatchNorm2d(channels)
        self.conv2 = HexConv2d(channels, channels, masked)
        self.norm2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return F.relu(x + residual)


class HexNet(nn.Module):
    """Small residual tower with separate policy and value heads."""

    def __init__(
        self,
        board_size: int,
        channels: int = 64,
        blocks: int = 6,
        masked: bool = True,
        input_planes: int = 3,
    ):
        super().__init__()
        self.board_size = board_size
        self.channels = channels
        self.blocks = blocks
        self.masked = masked
        self.input_planes = input_planes

        cells = board_size * board_size
        self.policy_size = cells + 1  # every cell, plus the swap action

        self.stem = HexConv2d(input_planes, channels, masked)
        self.stem_norm = nn.BatchNorm2d(channels)
        self.tower = nn.ModuleList(
            ResidualBlock(channels, masked) for _ in range(blocks)
        )

        # Two 1x1 channels is the standard AlphaZero policy head width; it is
        # enough to carry per-cell logits without a large fully-connected layer.
        self.policy_conv = nn.Conv2d(channels, 2, kernel_size=1, bias=False)
        self.policy_norm = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * cells, self.policy_size)

        self.value_conv = nn.Conv2d(channels, 1, kernel_size=1, bias=False)
        self.value_norm = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(cells, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (policy_logits, value).

        Logits are unnormalised and unmasked: illegal actions are masked by the
        caller, because the legal set depends on the position rather than on the
        network. Value is squashed to [-1, 1] from the mover's perspective.
        """
        x = F.relu(self.stem_norm(self.stem(x)))
        for block in self.tower:
            x = block(x)

        policy = F.relu(self.policy_norm(self.policy_conv(x)))
        policy = self.policy_fc(policy.flatten(1))

        value = F.relu(self.value_norm(self.value_conv(x)))
        value = F.relu(self.value_fc1(value.flatten(1)))
        value = torch.tanh(self.value_fc2(value)).squeeze(-1)

        return policy, value

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def describe(self) -> str:
        return (
            f"HexNet(board={self.board_size}, channels={self.channels}, "
            f"blocks={self.blocks}, masked={self.masked}, "
            f"params={self.parameter_count():,})"
        )


def build(board_size: int, preset: str = "full") -> HexNet:
    """Presets used across the project.

    'tiny' is for the 5x5 validation gate, where the game is exhaustively
    solvable and the point is to confirm the training loop reaches perfect play.
    'light' roughly halves the parameter count and runs about 2.4x faster on
    CPU, which matters because training has no GPU: it is the preset to reach
    for when a full run would not finish overnight.

    'full' is the 9x9 target sized to stay small enough to ship to a browser.
    """
    if preset == "tiny":
        return HexNet(board_size, channels=32, blocks=4)
    if preset == "light":
        return HexNet(board_size, channels=48, blocks=4)
    if preset == "full":
        return HexNet(board_size, channels=64, blocks=6)
    raise ValueError(f"unknown preset: {preset!r}")

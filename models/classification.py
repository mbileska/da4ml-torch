from __future__ import annotations

import torch


class DenseOnlyLGN_9F_Wide_128__40(torch.nn.Module):
    """Compatibility class for the saved SmartPixels TorchLogix checkpoint."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bin(x)
        for layer_name in ('fc1', 'fc2', 'fc3', 'fc4', 'fc5', 'fc6', 'fc7'):
            x = getattr(self, layer_name)(x)
        return self.group_sum(x)

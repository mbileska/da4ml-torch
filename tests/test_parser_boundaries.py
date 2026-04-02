import pytest
import torch

from torchlogix.layers import LogicDense


class NonLinearForward(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.flatten = torch.nn.Flatten()
        self.logic = LogicDense(in_dim=8, out_dim=8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = self.logic(x)
        return torch.relu(x)


def test_trace_rejects_custom_non_linear_forward_ops(da4ml_modules):
    trace_model, FixedVariableArrayInput, _ = da4ml_modules
    model = NonLinearForward().eval()

    with pytest.raises(NotImplementedError, match='call_function is not supported'):
        trace_model(model, inputs=FixedVariableArrayInput((1, 1, 2, 4)).quantize(0, 1, 1))

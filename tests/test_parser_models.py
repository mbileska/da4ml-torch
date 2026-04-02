import numpy as np
import torch

from torchlogix.layers import GroupSum, LogicConv2d, LogicDense, OrPooling2d


def assert_comb_matches_model(
    da4ml_modules, model: torch.nn.Module, symbolic_shape: tuple[int, ...], data_in: np.ndarray
) -> None:
    trace_model, FixedVariableArrayInput, comb_trace = da4ml_modules
    model.eval()
    inp, out = trace_model(model, inputs=FixedVariableArrayInput(symbolic_shape).quantize(0, 1, 1))
    comb = comb_trace(inp, out)

    with torch.no_grad():
        torch_out = model(torch.from_numpy(data_in)).detach().cpu().numpy()

    comb_out = np.asarray(comb.predict(data_in))

    np.testing.assert_array_equal(torch_out.reshape(torch_out.shape[0], -1), comb_out.reshape(comb_out.shape[0], -1))


class DenseClassifier(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.flatten = torch.nn.Flatten()
        self.hidden1 = LogicDense(in_dim=8, out_dim=6)
        self.hidden2 = LogicDense(in_dim=6, out_dim=4)
        self.head = GroupSum(k=2, tau=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        return self.head(x)


class ConvClassifier(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = LogicConv2d(in_dim=6, channels=1, num_kernels=4, receptive_field_size=3, tree_depth=2)
        self.pool = OrPooling2d(kernel_size=2, stride=2)
        self.flatten = torch.nn.Flatten()
        self.classifier = LogicDense(in_dim=16, out_dim=8)
        self.head = GroupSum(k=2, tau=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return self.head(x)


def test_custom_dense_forward_matches_comb_trace(da4ml_modules):
    torch.manual_seed(1)
    np.random.seed(1)

    model = DenseClassifier()
    data_in = np.random.randint(0, 2, size=(24, 1, 2, 4)).astype(np.float32)

    assert_comb_matches_model(da4ml_modules, model, (1, 1, 2, 4), data_in)


def test_conv_pool_dense_forward_matches_comb_trace(da4ml_modules):
    torch.manual_seed(2)
    np.random.seed(2)

    model = ConvClassifier()
    data_in = np.random.randint(0, 2, size=(12, 1, 6, 6)).astype(np.float32)

    assert_comb_matches_model(da4ml_modules, model, (1, 1, 6, 6), data_in)

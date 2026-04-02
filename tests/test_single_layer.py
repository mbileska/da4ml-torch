import numpy as np
import torch

from torchlogix.layers import LogicConv2d, LogicDense


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


def test_logic_dense_layer_matches_comb_trace(da4ml_modules):
    torch.manual_seed(0)
    np.random.seed(0)

    layer = LogicDense(in_dim=16, out_dim=8)
    model = torch.nn.Sequential(layer)
    data_in = np.random.randint(0, 2, size=(32, layer.in_dim)).astype(np.float32)

    assert_comb_matches_model(da4ml_modules, model, (1, layer.in_dim), data_in)


def test_logic_conv2d_layer_matches_comb_trace(da4ml_modules):
    torch.manual_seed(0)
    np.random.seed(0)

    layer = LogicConv2d(in_dim=6, channels=1, num_kernels=4, receptive_field_size=3, tree_depth=2)
    model = torch.nn.Sequential(layer)
    data_in = np.random.randint(0, 2, size=(16, 1, 6, 6)).astype(np.float32)

    assert_comb_matches_model(da4ml_modules, model, (1, 1, 6, 6), data_in)

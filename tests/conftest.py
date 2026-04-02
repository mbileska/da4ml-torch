import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

for relative_path in ('src', 'torchlogix/src'):
    path = ROOT / relative_path
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from da4ml.converter.plugin import _flatten_arr
from da4ml.trace import HWConfig

from da4ml_torch.parser import TorchParser


@pytest.fixture
def da4ml_modules():
    pytest.importorskip('da4ml')

    from da4ml.trace import FixedVariableArrayInput, comb_trace

    def trace_model(
        model,
        hwconf: HWConfig | tuple[int, int, int] = HWConfig(1, -1, -1),
        solver_options=None,
        verbose: bool = False,
        inputs=None,
        inputs_kif=None,
        dump: bool = False,
        framework: str | None = None,
        **kwargs,
    ):
        del framework
        hwconf = HWConfig(*hwconf) if isinstance(hwconf, tuple) else hwconf
        tracer = TorchParser(model, hwconf, solver_options, **kwargs)
        result = tracer.trace(verbose=verbose, inputs=inputs, inputs_kif=inputs_kif, dump=dump)
        if dump:
            return result
        inp, out = result
        return _flatten_arr(inp), _flatten_arr(out)

    return trace_model, FixedVariableArrayInput, comb_trace

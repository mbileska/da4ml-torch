import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]

for relative_path in ('src', 'torchlogix/src'):
    path = ROOT / relative_path
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


@pytest.fixture
def da4ml_modules():
    pytest.importorskip('da4ml')

    from da4ml.converter import trace_model
    from da4ml.trace import FixedVariableArrayInput, comb_trace

    return trace_model, FixedVariableArrayInput, comb_trace

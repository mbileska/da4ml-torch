# da4ml-torch

`da4ml-torch` provides the torch and torchlogix parser integration for `da4ml`.

## Tests

The pytest suite in `tests/` is intended to validate the `da4ml-torch` tracing and replay path against `torchlogix`, especially after the torchlogix eval forward path moved to NumPy.

Current test coverage is split into three groups:

- `tests/test_single_layer.py` verifies single `LogicDense` and `LogicConv2d` layers by comparing traced combinational models with direct model execution.
- `tests/test_parser_models.py` verifies multi-layer models with custom `forward()` methods, including combinations of `Flatten`, `LogicConv2d`, `OrPooling2d`, `LogicDense`, and `GroupSum`.
- `tests/test_parser_boundaries.py` verifies current parser limits by asserting that unsupported custom nonlinear operations fail with a clear `NotImplementedError`.

The test suite uses local source imports from both `src/` and `torchlogix/src/` through `tests/conftest.py`.

## Running tests

Install the project dependencies together with `da4ml`, then run:

```bash
pytest -q tests
```

If `da4ml` is not installed, the parser integration tests are skipped.

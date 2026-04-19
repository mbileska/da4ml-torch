# da4ml-torch

`da4ml-torch` converts supported PyTorch and TorchLogix models into DA4ML RTL projects. It traces a model, emits Verilog or VHDL through DA4ML, can optionally run RTL validation, and can launch vendor synthesis to collect rough timing and resource estimates.

This project is intended for logic-style models that DA4ML can replay, especially TorchLogix networks. It is not a general-purpose compiler for arbitrary PyTorch models.

## What It Does

- Loads a model by name or checkpoint path.
- Traces supported PyTorch/TorchLogix operations into DA4ML combinational logic.
- Writes a DA4ML RTL project with generated HDL, constraints, build scripts, and metadata.
- Produces JSON and Markdown conversion reports.
- Can run Vivado or Quartus synthesis when those tools are available.
- Can parse generated project reports later to print rough timing/resource estimates.

## Quick Start

Convert a model checkpoint under `models/`:

```bash
scripts/convert_torch_to_da4ml.py \
  --model best_model \
  --models-root models \
  --overwrite
```

The default output directory is:

```text
build/da4ml/<model-name>
```

Print estimates from an existing generated project:

```bash
scripts/estimate_da4ml_torch.py build/da4ml/best_model
```

## Estimate An LGN Without Vivado

Vivado or Quartus is only needed for real synthesis reports. For a quick local
estimate, leave synthesis disabled. This is the default, but it is fine to pass
`--synthesis-tool none` explicitly:

```bash
scripts/convert_torch_to_da4ml.py \
  --model /path/to/model.pth \
  --model-name my_lgn \
  --input-shape 1,NUM_INPUTS \
  --synthesis-tool none \
  --overwrite
```

Then print the estimate:

```bash
scripts/estimate_da4ml_torch.py build/da4ml/my_lgn
```

This reports DA4ML-side metadata: rough LUT estimate from `cost`, rough ASIC
logic estimate in DA4ML cost units, pipeline register bits when available,
target Fmax from `clock_period`, and target latency for pipelined designs. If
Vivado or Quartus reports exist in the project directory, the same command also
includes synthesis timing and resource numbers.

The ASIC estimate is not a cell-library area or gate-equivalent synthesis
report. It is a technology-independent proxy for comparing model sizes without
requiring ASIC tools.

For a state-dict checkpoint or a custom model constructor, expose a small
factory and point the converter at it:

```bash
scripts/convert_torch_to_da4ml.py \
  --model-factory my_model_file:make_model \
  --model-name my_lgn \
  --input-shape 1,NUM_INPUTS \
  --synthesis-tool none \
  --overwrite
```

If the package is installed, the same tools are available as console commands:

```bash
da4ml-torch-convert --model best_model --models-root models --overwrite
da4ml-torch-estimate build/da4ml/best_model
```

The old TorchLogix command name still exists for compatibility:

```bash
da4ml-torchlogix-convert --model best_model --models-root models --overwrite
```

## Running Synthesis

If Vivado is available on the cluster:

```bash
scripts/convert_torch_to_da4ml.py \
  --model best_model \
  --models-root models \
  --synthesis-tool vivado \
  --overwrite
```

If Quartus is available:

```bash
scripts/convert_torch_to_da4ml.py \
  --model best_model \
  --models-root models \
  --synthesis-tool quartus \
  --overwrite
```

The converter writes reports under:

```text
build/da4ml/<model-name>/analysis/
```

Important generated files include:

```text
analysis/conversion_summary.json
analysis/effective_config.json
analysis/da4ml_report.json
analysis/da4ml_report.md
metadata.json
build_vivado_prj.tcl
build_quartus_prj.tcl
src/
model/
sim/
```

## Vivado 2021.1 Compatibility

Some DA4ML-generated Vivado scripts include:

```tcl
-global_retiming on
```

Vivado 2021.1 rejects that option. The torch converter now sanitizes generated Vivado scripts before running synthesis, so the emitted command becomes:

```tcl
synth_design -top $top_module -mode out_of_context \
    -flatten_hierarchy full -resource_sharing auto -directive AreaOptimized_High
```

For an already-generated project, the manual workaround is:

```bash
python - <<'PY'
from pathlib import Path
p = Path("build/da4ml/best_model/build_vivado_prj.tcl")
p.write_text(p.read_text().replace(" -global_retiming on", ""))
PY
```

Then rerun Vivado from that project directory:

```bash
cd build/da4ml/best_model
vivado -mode batch -source build_vivado_prj.tcl
```

## Model Loading

The converter can load:

- a full `torch.nn.Module` checkpoint,
- a raw `state_dict`,
- a checkpoint dict containing `state_dict`, `model_state_dict`, or `model`,
- a model returned by a factory function.

### Full Module Checkpoints

For a full saved module:

```bash
scripts/convert_torch_to_da4ml.py \
  --checkpoint path/to/model.pth \
  --input-shape 1,9 \
  --overwrite
```

If the top-level model class is missing but the model is a simple single-input chain of child modules, the converter can auto-load it with a sequential fallback. This avoids per-checkpoint compatibility classes.

That fallback is valid for simple models where forward execution is equivalent to:

```python
x = input
for child in model.children():
    x = child(x)
return x
```

For models with branches, skip connections, multiple inputs, or custom forward logic, use a model factory.

### State Dict Checkpoints

A raw `state_dict` does not contain the architecture or forward method. For those checkpoints, provide one of:

- `--architecture <TorchLogixModelName>`
- `--training-config path/to/training_config.json`
- `--model-factory package.module:make_model`

Example:

```bash
scripts/convert_torch_to_da4ml.py \
  --checkpoint path/to/state_dict.pth \
  --model-factory my_package.my_models:make_model \
  --model-kwargs '{"hidden": 128, "classes": 10}' \
  --input-shape 1,784 \
  --overwrite
```

### Model Names

`--model` is a convenience selector. It can be:

- a checkpoint path,
- a directory containing a checkpoint,
- a model name under `--models-root`.

For example:

```bash
scripts/convert_torch_to_da4ml.py \
  --model mnist_logic \
  --models-root models/torchlogix \
  --overwrite
```

The converter looks for common checkpoint names such as:

```text
best_model.pt
best_model.pth
bestmodel.pt
bestmodel.pth
bestmodelpth
*best*.pt
*best*.pth
```

If multiple candidates are found, pass `--checkpoint` explicitly.

## Input Shape And Precision

The converter can infer input shape for many TorchLogix models by inspecting the first logic layer and binarization layer.

If inference fails, pass `--input-shape`:

```bash
--input-shape 1,9
--input-shape 1,1,28,28
```

Multiple inputs are separated by semicolons:

```bash
--input-shape "1,16;1,8"
```

Input precision is controlled by `--input-kif K I F`:

```bash
--input-kif 0 1 0
--input-kif 0 1 8
```

When omitted:

- thresholded/binarized models default to unsigned `1.8`,
- binary logic models default to unsigned `1.0`.

## Validation

Validate PyTorch output against DA4ML combinational simulation:

```bash
scripts/convert_torch_to_da4ml.py \
  --model best_model \
  --models-root models \
  --validate-comb-samples 32 \
  --overwrite
```

Compile and validate generated RTL against DA4ML combinational simulation:

```bash
scripts/convert_torch_to_da4ml.py \
  --model best_model \
  --models-root models \
  --validate-rtl-samples 32 \
  --overwrite
```

RTL validation requires the local Verilog/VHDL simulation toolchain used by DA4ML, such as Verilator for Verilog.

## Timing And Resource Estimates

The estimate command works on any generated DA4ML project:

```bash
scripts/estimate_da4ml_torch.py build/da4ml/best_model
```

Without vendor synthesis, reports contain DA4ML-side estimates and target-clock
metadata:

- rough LUT estimate from DA4ML cost,
- rough ASIC logic estimate in DA4ML cost units,
- rough ASIC register estimate from pipeline register bits,
- target Fmax from `clock_period`,
- target latency in nanoseconds for pipelined designs.

With Vivado or Quartus synthesis, reports can also include parsed implementation data:

- actual period,
- Fmax,
- latency in nanoseconds,
- LUT/FF/DSP/BRAM-style resource usage,
- power when available.

Write estimates as JSON:

```bash
scripts/estimate_da4ml_torch.py build/da4ml/best_model --format json -o estimates.json
```

## Supported Model Surface

The converter supports the subset of PyTorch/TorchLogix that `da4ml_torch.parser` and `da4ml_torch.layers` can replay.

Currently supported model components include:

- `torchlogix.layers.LogicDense`
- `torchlogix.layers.LogicConv2d`
- `torchlogix.layers.GroupSum`
- `torchlogix.layers.OrPooling2d`
- TorchLogix binarization layers
- `torch.nn.Flatten`
- supported `reshape`, `flatten`, `transpose`, `matmul`, and basic arithmetic patterns

Unsupported or likely unsupported examples include:

- arbitrary `torch.nn.Linear` networks,
- standard `torch.nn.Conv2d` networks,
- BatchNorm,
- attention layers,
- arbitrary Python control flow,
- custom operators without a DA4ML replay implementation,
- custom multi-input forward methods unless the parser can trace and replay them.

When a model uses unsupported operations, conversion usually fails during `torch.fx` tracing or DA4ML replay with a specific unsupported layer/operator message.

## Common Commands

List known TorchLogix architecture names:

```bash
scripts/convert_torch_to_da4ml.py --list-architectures
```

Force a top-level RTL module name:

```bash
scripts/convert_torch_to_da4ml.py \
  --model best_model \
  --project-name my_top_module \
  --overwrite
```

Generate VHDL instead of Verilog:

```bash
scripts/convert_torch_to_da4ml.py \
  --model best_model \
  --flavor vhdl \
  --overwrite
```

Use a specific FPGA part:

```bash
scripts/convert_torch_to_da4ml.py \
  --model best_model \
  --part-name xcvu13p-flga2577-2-e \
  --overwrite
```

Adjust pipeline target:

```bash
scripts/convert_torch_to_da4ml.py \
  --model best_model \
  --latency-cutoff 5.0 \
  --clock-period 5.0 \
  --overwrite
```

Use `--latency-cutoff <= 0` for combinational RTL:

```bash
scripts/convert_torch_to_da4ml.py \
  --model best_model \
  --latency-cutoff -1 \
  --overwrite
```

## Configuration Files

You can place conversion settings in JSON:

```json
{
  "model": "best_model",
  "models_root": "models",
  "input_shape": [1, 9],
  "flavor": "verilog",
  "latency_cutoff": 5.0,
  "clock_period": 5.0,
  "part_name": "xcvu13p-flga2577-2-e",
  "validate_comb_samples": 32,
  "synthesis_tool": "none",
  "report_formats": ["json", "md"]
}
```

Run with:

```bash
scripts/convert_torch_to_da4ml.py --config path/to/config.json --overwrite
```

Command-line flags override JSON config values.

## Repository Layout

```text
src/da4ml_torch/
  parser.py                         Torch FX to DA4ML replay parser
  layers/                           Replay implementations for supported layers
  tools/torchlogix_to_da4ml.py      Main conversion and estimate CLI implementation

scripts/
  convert_torch_to_da4ml.py         Generic local conversion wrapper
  estimate_da4ml_torch.py           Local estimate/report wrapper
  convert_torchlogix_to_da4ml.py    Backward-compatible wrapper

models/
  torchlogix/                       Example model input area and config

da4ml/                              Vendored/local DA4ML checkout
torchlogix/                         Vendored/local TorchLogix checkout
tests/                              Unit tests for parser and tools
```

## Troubleshooting

### `Unable to infer input shape`

Pass `--input-shape` explicitly:

```bash
--input-shape 1,9
```

### `Checkpoint contains a state_dict, so the model architecture is required`

A `state_dict` only stores weights. Pass:

```bash
--architecture <TorchLogixModelName>
```

or:

```bash
--model-factory package.module:make_model
```

### `Unknown option '-global_retiming'`

This is a Vivado TCL compatibility issue. Regenerate the project with the current converter, or remove the option manually from `build_vivado_prj.tcl` as shown above.

### Unsupported layer/operator errors

The model uses an operation that DA4ML Torch replay does not implement yet. Add a replay implementation under `src/da4ml_torch/layers/` or rewrite the model using supported logic layers.

### Synthesis tool not found

Make sure the tool is on `PATH`:

```bash
which vivado
which quartus_sh
```

Then rerun with:

```bash
--synthesis-tool vivado
```

or:

```bash
--synthesis-tool quartus
```

## Development Checks

Run focused tool tests:

```bash
pytest -q tests/test_tools.py
```

Run syntax checks:

```bash
python -m py_compile src/da4ml_torch/tools/torchlogix_to_da4ml.py tests/test_tools.py
```

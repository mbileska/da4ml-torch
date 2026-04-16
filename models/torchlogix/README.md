# DA4ML Torch Conversion Inputs

Place Torch/TorchLogix checkpoints and optional per-model conversion configs in
this tree. Generated RTL projects should go under `build/da4ml/` or another
ignored output directory, not under `models/`.

The converter can load:

- a full `torch.nn.Module` checkpoint,
- a raw `state_dict`,
- a checkpoint dict containing `state_dict`, `model_state_dict`, or `model`.

For full-module checkpoints that only need to run child modules in registration
order, the converter can auto-load a missing top-level model class. That removes
the need for one-off per-checkpoint compatibility classes.
If a checkpoint has branches, multiple inputs, or custom control flow, provide
`--model-factory package.module:make_model` so the real `forward` method is
available.

For `state_dict` checkpoints, provide either:

- `training_config.json` beside the checkpoint,
- `--architecture <TorchLogixModelName>`,
- or `--model-factory package.module:make_model`.

General one-shot conversion:

```bash
scripts/convert_torch_to_da4ml.py \
  --model <experiment-or-checkpoint> \
  --models-root models/torchlogix \
  --overwrite
```

Run conversion and vendor synthesis in the same command when Vivado or Quartus
is on `PATH`:

```bash
scripts/convert_torch_to_da4ml.py \
  --model <experiment-or-checkpoint> \
  --models-root models/torchlogix \
  --synthesis-tool vivado \
  --overwrite
```

Print timing/resource estimates from an existing generated project:

```bash
scripts/estimate_da4ml_torch.py build/da4ml/<model>
```

Example:

```bash
scripts/convert_torch_to_da4ml.py \
  --config models/torchlogix/example_da4ml_config.json \
  --overwrite
```

If a model checkpoint is named `best_model.pt`, `best_model.pth`,
`bestmodel.pt`, `bestmodel.pth`, or `bestmodelpth`, and it is the only matching
file under `models/`, the converter will find it automatically.

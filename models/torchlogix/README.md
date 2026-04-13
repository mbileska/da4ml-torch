# TorchLogix DA4ML Conversion Inputs

Place TorchLogix checkpoints and optional per-model conversion configs in this
tree. Generated RTL projects should go under `build/da4ml/` or another ignored
output directory, not under `models/`.

The converter can load:

- a full `torch.nn.Module` checkpoint,
- a raw `state_dict`,
- a checkpoint dict containing `state_dict`, `model_state_dict`, or `model`.

For `state_dict` checkpoints, provide either:

- `training_config.json` beside the checkpoint,
- `--architecture <TorchLogixModelName>`,
- or `--model-factory package.module:make_model`.

Example:

```bash
scripts/convert_torchlogix_to_da4ml.py \
  --config models/torchlogix/example_da4ml_config.json \
  --overwrite
```

If a model checkpoint is named `best_model.pt`, `best_model.pth`,
`bestmodel.pt`, `bestmodel.pth`, or `bestmodelpth`, and it is the only matching
file under `models/`, the converter will find it automatically.

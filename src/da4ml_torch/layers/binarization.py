import numpy as np
from da4ml.trace import FixedVariableArray
from torchlogix.layers.binarization import DummyBinarization, FixedBinarization, LearnableBinarization, SoftBinarization

from ._base import ReplayBase


def _merge_dim_with_last(x: FixedVariableArray, dim: int) -> FixedVariableArray:
    ndim = x.ndim
    if ndim < 2:
        raise ValueError('Need at least two dimensions to merge with the last dimension.')
    if dim < 0:
        dim += ndim
    if not 0 <= dim < ndim - 1:
        raise ValueError(f'dim must be in [0, {ndim - 2}], got {dim}.')

    last = ndim - 1
    perm = list(range(ndim))
    perm.pop(last)
    perm.insert(dim + 1, last)

    y = x.transpose(tuple(perm))
    shape = list(y.shape)
    shape[dim] *= shape[dim + 1]
    shape.pop(dim + 1)
    return y.reshape(*shape)


class ReplayDummyBinarization(ReplayBase):
    handles = (DummyBinarization,)

    def call(self, inputs: FixedVariableArray):
        return inputs


class ReplayThresholdBinarization(ReplayBase):
    handles = (FixedBinarization, SoftBinarization, LearnableBinarization)

    def call(self, inputs: FixedVariableArray):
        thresholds = self.module.get_thresholds()
        thresholds = thresholds.detach().cpu().numpy()

        x = inputs[..., None]
        if thresholds.ndim == 2 and x.ndim == 5:
            thresholds = thresholds.reshape(1, -1, 1, 1, thresholds.shape[1])

        outputs = x > thresholds
        return _merge_dim_with_last(outputs, self.module.feature_dim)

import numpy as np
import torch
from da4ml.trace import FixedVariableArray

from ._base import ReplayBase


class ReplayFlatten(ReplayBase):
    handles = (torch.nn.Flatten,)

    def call(self, input: FixedVariableArray):
        if input.ndim <= 1:
            return input.flatten()
        return input.reshape(input.shape[0], -1)

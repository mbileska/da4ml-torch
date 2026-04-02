import numpy as np
from da4ml.trace import FixedVariableArray

from ._base import ReplayBase
from torchlogix.layers import GroupSum, LogicConv2d, LogicDense, OrPooling2d


class ReplayLogicConv2d(ReplayBase):
    handles = (LogicConv2d,)

    def call(self, inputs: FixedVariableArray):
        self.module: LogicConv2d

        # Save solver_options for conversion back
        solver_options = inputs.solver_options

        # Convert FixedVariableArray to numpy array of FixedVariable objects
        inputs_np = np.array(inputs)

        # Call torchlogix forward (which now works with numpy arrays)
        result_np = self.module(inputs_np)

        # Convert back to FixedVariableArray
        return FixedVariableArray(result_np, solver_options=solver_options)


class ReplayLogicDense(ReplayBase):
    handles = (LogicDense,)

    def call(self, inputs: FixedVariableArray):
        self.module: LogicDense

        # Save solver_options for conversion back
        solver_options = inputs.solver_options

        # Convert FixedVariableArray to numpy array of FixedVariable objects
        inputs_np = np.array(inputs)

        # Call torchlogix forward (which now works with numpy arrays)
        result_np = self.module(inputs_np)

        # Convert back to FixedVariableArray
        return FixedVariableArray(result_np, solver_options=solver_options)


def im2col(inp, px_in):
    inp_col = np.lib.stride_tricks.sliding_window_view(  # type: ignore
        inp,  # type: ignore
        px_in,
        axis=tuple(range(len(px_in))),  # type: ignore
    )
    inp_col = np.moveaxis(inp_col, len(px_in), -1).reshape(*inp_col.shape[: len(px_in)], -1)
    return inp_col


class ReplayOrPooling2d(ReplayBase):
    handles = (OrPooling2d,)

    def call(self, inputs: FixedVariableArray):
        self.module: OrPooling2d

        # Save solver_options for conversion back
        solver_options = inputs.solver_options

        # Convert FixedVariableArray to numpy array of FixedVariable objects
        inputs_np = np.array(inputs)

        # Call torchlogix forward (which now works with numpy arrays)
        result_np = self.module(inputs_np)

        # Convert back to FixedVariableArray
        return FixedVariableArray(result_np, solver_options=solver_options)


# class ReplayOrPooling3d(ReplayBase):
#     handles = (OrPooling3d,)

#     def call(self, inputs: FixedVariableArray):
#         self.module: OrPooling3d

#         solver_options = inputs.solver_options
#         inputs_np = np.array(inputs)
#         result_np = self.module(inputs_np)
#         return FixedVariableArray(result_np, solver_options=solver_options)


class ReplayGroupSum(ReplayBase):
    handles = (GroupSum,)

    def call(self, inputs: FixedVariableArray):
        self.module: GroupSum

        # Save solver_options for conversion back
        solver_options = inputs.solver_options

        # Convert FixedVariableArray to numpy array of FixedVariable objects
        inputs_np = np.array(inputs)

        # Call torchlogix forward (which now works with numpy arrays)
        result_np = self.module(inputs_np)

        # Convert back to FixedVariableArray
        return FixedVariableArray(result_np, solver_options=solver_options)

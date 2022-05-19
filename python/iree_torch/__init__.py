# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np

import torch
from torch.utils._pytree import tree_map

import iree.runtime as ireert
import iree.compiler as ireec


class IREEInvoker:
    """A wrapper around an IREE module that provides a Pythonic interface.
    
    Specifically, this adapts `module.forward(...)` and similar calls into
    lower-level calls into the functions in the IREE module, and also converts
    between the IREE and Torch types.
    """

    def __init__(self, iree_module):
        self._iree_module = iree_module
        self.device = iree_module._context.config.device

    def __getattr__(self, function_name: str):
        def invoke(*args):
            def wrap(x):
                if isinstance(x, torch.Tensor):
                    return ireert.asdevicearray(self.device, x)
                return x
            def unwrap(x):
                if isinstance(x, ireert.DeviceArray):
                    return torch.from_numpy(np.asarray(x).copy())
                return x
            # TODO: Investigate how to share CUDA arrays between IREE and Torch.
            iree_args = tree_map(wrap, args)
            result = self._iree_module[function_name](*iree_args)
            # TODO: Investigate why a copy is needed here.
            # Without the copy, certain sets of tests, when run together, will
            # cause a segfault when the process is exiting.
            # It seems to be related to Torch attempting to free a Numpy array
            # that is backed by IREE memory, resulting in
            # iree_hal_buffer_view_release reading from a null pointer.
            return tree_map(unwrap, result)
        return invoke


class NumpyIREEInvoker:
    """A wrapper around IREEInvoker which accepts and returns numpy types"""

    def __init__(self, iree_invoker):
        self.iree_invoker = iree_invoker

    def __getattr__(self, function_name: str):
        def invoke(*args):
            def wrap(x):
                if isinstance(x, np.ndarray):
                    return torch.from_numpy(x)
                return x
            def unwrap(x):
                if isinstance(x, torch.Tensor):
                    return x.numpy()
                return x
            torch_args = tree_map(wrap, args)
            result = getattr(self.iree_invoker, function_name)(*torch_args)
            return tree_map(unwrap, result)
        return invoke


def compile_to_vmfb(mlir_module, target_backend="dylib"):
    """Compile an MLIR module to an IREE Flatbuffer.

    The module is expected to be in the format produced by `torch_mlir.compile`
    with `OutputType.LINALG_ON_TENSORS`.

    TODO: Expose more compiler options.
    """
    # Here, mlir_module is typically going to be coming from the Torch-MLIR
    # MLIR CAPI assembly. We stringify it to cross the border into the
    # IREE MLIR CAPI assembly.
    return ireec.compile_str(str(mlir_module),
                             target_backends=[target_backend],
                             input_type=ireec.InputType.TM_TENSOR,
                             extra_args=["--iree-flow-demote-i64-to-i32"])


def load_vmfb(flatbuffer, driver="dylib"):
    """Load an IREE Flatbuffer into an in-process runtime wrapper.

    The wrapper accepts and returns `torch.Tensor` types.
    """
    vm_module = ireert.VmModule.from_flatbuffer(flatbuffer)
    config = ireert.Config(driver_name=driver)
    ctx = ireert.SystemContext(config=config)
    ctx.add_vm_module(vm_module)
    return IREEInvoker(ctx.modules.module)

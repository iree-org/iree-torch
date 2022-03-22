# Copyright 2021 Google LLC
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

import iree.runtime as ireert
import iree.compiler as ireec

from torch_mlir_e2e_test.linalg_on_tensors_backends.abc import LinalgOnTensorsBackend
from torch_mlir_e2e_test.torchscript.configs import LinalgOnTensorsBackendTestConfig


class IREEInvoker:
    def __init__(self, iree_module):
        self._iree_module = iree_module

    def __getattr__(self, function_name: str):
        def invoke(*args):
            return self._iree_module[function_name](*args)
        return invoke


class IREELinalgOnTensorsBackend(LinalgOnTensorsBackend):
    """Main entry-point for the reference backend."""

    def __init__(self):
        super().__init__()

    def compile(self, imported_module):
        """Compiles an imported module, with a flat list of functions.
        The module is expected to be in linalg-on-tensors + scalar code form.
        TODO: More clearly define the backend contract. Generally this will
        extend to support globals, lists, and other stuff.

        Args:
          imported_module: The MLIR module consisting of funcs in the torch
            dialect.
        Returns:
          An opaque, backend specific compiled artifact object that can be
          passed to `load`.
        """
        return ireec.compile_str(str(imported_module),
                                 target_backends=["dylib-llvm-aot"])

    def load(self, flatbuffer) -> IREEInvoker:
        """Loads a compiled artifact into the runtime."""
        vm_module = ireert.VmModule.from_flatbuffer(flatbuffer)
        config = ireert.Config(driver_name="dylib")
        ctx = ireert.SystemContext(config=config)
        ctx.add_vm_module(vm_module)
        return IREEInvoker(ctx.modules.module)


config = LinalgOnTensorsBackendTestConfig(IREELinalgOnTensorsBackend())
xfail_set = COMMON_TORCH_MLIR_LOWERING_XFAILS

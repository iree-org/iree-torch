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

from typing import Any

import argparse
import re
import sys

import numpy as np

import iree.runtime as ireert
import iree.compiler as ireec

from torch_mlir_e2e_test.linalg_on_tensors_backends.abc import LinalgOnTensorsBackend
from torch_mlir_e2e_test.torchscript.configs import LinalgOnTensorsBackendTestConfig
from torch_mlir_e2e_test.torchscript.registry import GLOBAL_TEST_REGISTRY
from torch_mlir_e2e_test.torchscript.framework import run_tests
from torch_mlir_e2e_test.torchscript.reporting import report_results
from torch_mlir_e2e_test.test_suite import COMMON_TORCH_MLIR_LOWERING_XFAILS

# Import tests to register them in the global registry.
from torch_mlir_e2e_test.test_suite import register_all_tests
register_all_tests()

# Tests that fail due to incomplete support for RNG.
# In particular, the torch_c.get_next_seed op.
_common_rng_xfail_set = {
    "DropoutTrainModule_basic",
    "UniformModule_basic",
    "UniformStaticModule_basic",
    "BernoulliModule_basic",
    "BernoulliZerosModule_basic",
    "BernoulliOnesModule_basic",
    "BernoulliFloatModule_basic",
    "BernoulliTensorModule_basic",
}

# F64-related failures: https://github.com/google/iree/issues/8826
_common_f64_xfail_set = {
    "SoftmaxIntArgTypeF64Module_basic",
    "LogSoftmaxIntModule_basic",
    "NumToTensorFloatModule_basic",
    "ElementwiseWhereScalarOtherModule_basic",
    "ElementwiseWhereScalarSelfModule_basic",
    "ElementwiseMulTensorFloatModule_basic",
    "ElementwiseDivTensorFloatModule_basic",
    "TypePromotionSameCategoryZeroRankWider_basic",
    "TypeConversionF32ToF64Module_basic",
    "TypeConversionF64ToF32Module_basic",
    "TypeConversionI1ToF64Module_basic",
    "ReduceSumDtypeFloatModule_basic",
    "ReduceSumDimIntListDtypeFloatModule_basic",
    "ReduceMeanDtypeModule_basic",
    "ReduceMaxAlongDim_basic",
    "ReduceMaxAlongDimNegative_basic",
    "ReduceMaxKeepDim_basic",
    "OnesLikeModule_falsePinMemory",
    "Fill_TensorFloat64WithFloat64_basic",
    "Fill_TensorFloat64WithInt64_basic",
    "TensorToFloatZeroRank_basic",
    "TensorToFloat_basic",
    "DivFloatModule_basic",
}

# https://github.com/google/iree/issues/9020
_issue_9020_xfail_set = {
    "SoftmaxIntModule_basic",
    "_SoftmaxModule_basic",
    "SoftmaxIntNegDimModule_basic",
    "_LogSoftmaxModule_basic",
    "ArgmaxModule_with_dim",
    "ArgmaxModule_keepDim",
}

DYLIB_XFAIL_SET = COMMON_TORCH_MLIR_LOWERING_XFAILS | _common_rng_xfail_set | _common_f64_xfail_set | _issue_9020_xfail_set
VMVX_XFAIL_SET = COMMON_TORCH_MLIR_LOWERING_XFAILS | _common_rng_xfail_set | _common_f64_xfail_set


def recursively_convert_to_numpy(o: Any):
    if isinstance(o, ireert.DeviceArray):
        # TODO: Investigate why a copy is needed here.
        # Without the copy, certain sets of tests, when run together, will
        # cause a segfault when the process is exiting.
        # It seems to be related to Torch attempting to free a Numpy array
        # that is backed by IREE memory, resulting in
        # iree_hal_buffer_view_release reading from a null pointer.
        return np.asarray(o).copy()
    if isinstance(o, tuple):
        return tuple(recursively_convert_to_numpy(x) for x in o)
    if isinstance(o, list):
        return [recursively_convert_to_numpy(x) for x in o]
    if isinstance(o, dict):
        return {k: recursively_convert_to_numpy(v) for k, v in o.items()}
    # No-op cases. Explicitly enumerated to avoid things sneaking through.
    if isinstance(o, str):
        return o
    if isinstance(o, float):
        return o
    if isinstance(o, int):
        return o
    raise Exception(f"Unexpected Python type: {o}")


class IREEInvoker:
    def __init__(self, iree_module):
        self._iree_module = iree_module

    def __getattr__(self, function_name: str):
        def invoke(*args):
            result = self._iree_module[function_name](*args)
            return recursively_convert_to_numpy(result)
        return invoke


class IREELinalgOnTensorsBackend(LinalgOnTensorsBackend):
    """Main entry-point for the reference backend."""

    def __init__(self, backend: str):
        super().__init__()
        self.backend = backend

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
                                 target_backends=[self.backend],
                                 input_type=ireec.InputType.TM_TENSOR)

    def load(self, flatbuffer) -> IREEInvoker:
        """Loads a compiled artifact into the runtime."""
        vm_module = ireert.VmModule.from_flatbuffer(flatbuffer)
        config = ireert.Config(driver_name=self.backend)
        ctx = ireert.SystemContext(config=config)
        ctx.add_vm_module(vm_module)
        return IREEInvoker(ctx.modules.module)


# ==============================================================================
# Main-related things
# ==============================================================================

def _get_argparse():
    # TODO: Add CUDA and Vulkan.
    config_choices = ['dylib', 'vmvx']
    parser = argparse.ArgumentParser(description='Run torchscript e2e tests.')
    parser.add_argument('-c', '--config',
                        choices=config_choices,
                        default='dylib',
                        help=f'''
Meaning of options:
"dylib": run through IREE's dylib backend
"vmvx": run through IREE's VMVX backend
''')
    parser.add_argument('-f', '--filter', default='.*', help='''
Regular expression specifying which tests to include in this run.
''')
    parser.add_argument('-v', '--verbose',
                        default=False,
                        action='store_true',
                        help='report test results with additional detail')
    return parser


def main():
    args = _get_argparse().parse_args()
    tests = [
        test for test in GLOBAL_TEST_REGISTRY
        if re.match(args.filter, test.unique_name)
    ]
    if len(tests) == 0:
        print(
            f'ERROR: the provided filter {args.filter!r} does not match any tests'
        )
        print('The available tests are:')
        for test in GLOBAL_TEST_REGISTRY:
            print(test.unique_name)
        sys.exit(1)

    if args.config == "dylib":
        iree_backend = IREELinalgOnTensorsBackend("dylib")
        xfail_set = DYLIB_XFAIL_SET
    elif args.config == "vmvx":
        iree_backend = IREELinalgOnTensorsBackend("vmvx")
        xfail_set = VMVX_XFAIL_SET

    config = LinalgOnTensorsBackendTestConfig(iree_backend)
    results = run_tests(tests, config)
    failed = report_results(results, xfail_set, args.verbose)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()

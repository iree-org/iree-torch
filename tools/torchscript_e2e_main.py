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
import os
import re
import sys

import torch
from torch.utils._pytree import tree_map

import iree_torch

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

# F64 and i64 related failures: https://github.com/google/iree/issues/8826
_common_unsupported_data_types_xfail_set = {
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
    "TorchPrimLoopWhileLikeModule_basic",
    "ToDtypeLayoutNoneModule_basic",
    "ToDtypeLayoutStridedModule_basic",
    "MeanDimDtypeModule_basic",
    "MeanDtypeModule_basic",
    "CeilFloatModule_basic",
    "GeFloatIntModule_basic",
    "GtFloatIntModule_basic",
    "NeFloatIntModule_basic",
    "ScalarImplicitFloatModule_basic",
}

DYLIB_XFAIL_SET = COMMON_TORCH_MLIR_LOWERING_XFAILS | _common_rng_xfail_set | _common_unsupported_data_types_xfail_set
VMVX_XFAIL_SET = COMMON_TORCH_MLIR_LOWERING_XFAILS | _common_rng_xfail_set | _common_unsupported_data_types_xfail_set

# Tests that we need to globally exclude from the list.
# These are actually F64-related issues, but because of how the test works,
# the garbage that IREE returns sometimes passes the test. So the result
# is nondeterministic and cannot be XFAIL'ed.
GLOBALLY_EXCLUDED_TESTS = {
    "NewEmptyModuleNonDefaultFloatDtype_basic",
    "ZerosLikeModule_falsePinMemory",
    "EmptyLikeModule_falsePinMemory",
}

class IREELinalgOnTensorsBackend(LinalgOnTensorsBackend):

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
        return iree_torch.compile_to_vmfb(imported_module, self.backend)

    def load(self, flatbuffer):
        """Loads a compiled artifact into the runtime."""
        return iree_torch.NumpyIREEInvoker(iree_torch.load_vmfb(flatbuffer,
                                                                self.backend))


# ==============================================================================
# Artifact dumping
# ==============================================================================


def dump_standalone_test_artifacts(artifact_dump_dir: str, tests):
    os.makedirs(artifact_dump_dir, exist_ok=True)
    for test in tests:
        captured_imported_module = None

        class CaptureImportedModule(LinalgOnTensorsBackend):
            def compile(self, imported_module):
                nonlocal captured_imported_module
                captured_imported_module = imported_module
                return None

            def load(self, artifact):
                return None

        try:
            LinalgOnTensorsBackendTestConfig(
                CaptureImportedModule()).compile(test.program_factory())
        except:
            continue

        assert captured_imported_module is not None

        with open(os.path.join(artifact_dump_dir, test.unique_name + ".mlir"), "w") as f:
            f.write(str(captured_imported_module))


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
    parser.add_argument('--dump-standalone-test-artifacts',
                        default=False,
                        action='store_true',
                        help='Dump artifacts for standalone IREE testing.')
    parser.add_argument('--dump-standalone-test-artifacts-dir', help='''
Directory in which to dump standalone testing artifacts
''')
    return parser


def main():
    args = _get_argparse().parse_args()

    all_tests_to_attempt = list(sorted(
        test for test in GLOBAL_TEST_REGISTRY if test.unique_name not in GLOBALLY_EXCLUDED_TESTS))
    tests = [
        test for test in all_tests_to_attempt
        if re.match(args.filter, test.unique_name)
    ]
    if len(tests) == 0:
        print(
            f'ERROR: the provided filter {args.filter!r} does not match any tests'
        )
        print('The available tests are:')
        for test in all_tests_to_attempt:
            print(test.unique_name)
        sys.exit(1)

    if args.config == "dylib":
        iree_backend = IREELinalgOnTensorsBackend("dylib")
        xfail_set = DYLIB_XFAIL_SET
    elif args.config == "vmvx":
        iree_backend = IREELinalgOnTensorsBackend("vmvx")
        xfail_set = VMVX_XFAIL_SET

    config = LinalgOnTensorsBackendTestConfig(iree_backend)
    if args.dump_standalone_test_artifacts:
        dump_standalone_test_artifacts(
            args.dump_standalone_test_artifacts_dir, tests)
        sys.exit(0)
    results = run_tests(tests, config)
    failed = report_results(results, xfail_set, args.verbose)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()

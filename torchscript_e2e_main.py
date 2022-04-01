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

# Import tests to register them in the global registry.
from torch_mlir_e2e_test.test_suite import register_all_tests
register_all_tests()


class IREEInvoker:
    def __init__(self, iree_module):
        self._iree_module = iree_module

    def __getattr__(self, function_name: str):
        def invoke(*args):
            result = self._iree_module[function_name](*args)
            return np.asarray(result)
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
                                 target_backends=[self.backend])

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
    elif args.config == "vmvx":
        iree_backend = IREELinalgOnTensorsBackend("vmvx")

    config = LinalgOnTensorsBackendTestConfig(iree_backend)
    results = run_tests(tests, config)
    failed = report_results(results, set(), args.verbose)
    sys.exit(1 if failed else 0)

if __name__ == "__main__":
    main()

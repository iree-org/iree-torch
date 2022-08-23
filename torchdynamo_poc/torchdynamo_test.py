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
import unittest
import torch
from torchbench import _unwrap_single_tuple_return, torch_mlir_compiler

class TestTorchFXTupleReturnUnwrapping(unittest.TestCase):
    @staticmethod
    def maybe_unwrap_and_test_graph(func, unwrap_result_test):
        fx_graph = torch.fx.symbolic_trace(func)
        was_unwrapped = _unwrap_single_tuple_return(fx_graph)
        unwrap_result_test(fx_graph, was_unwrapped)

    def test_no_tuple(self):
        def unwrap_result_test(fx_graph, was_unwrapped: bool):
            self.assertFalse(was_unwrapped)
            result = fx_graph(torch.rand(10))
            self.assertIsInstance(result, torch.Tensor)
        self.maybe_unwrap_and_test_graph(lambda x: x, unwrap_result_test)

    def test_zero_elem_tuple(self):
        def unwrap_result_test(fx_graph, was_unwrapped: bool):
            self.assertFalse(was_unwrapped)
            result = fx_graph(torch.rand(10))
            self.assertIsInstance(result, tuple)
            self.assertEqual(len(result), 0)
        self.maybe_unwrap_and_test_graph(lambda x: tuple(), unwrap_result_test)

    def test_two_elem_tuple(self):
        def unwrap_result_test(fx_graph, was_unwrapped: bool):
            self.assertFalse(was_unwrapped)
            result = fx_graph(torch.rand(10))
            self.assertIsInstance(result, tuple)
            self.assertEqual(len(result), 2)
        self.maybe_unwrap_and_test_graph(lambda x: (x, x), unwrap_result_test)

    def test_one_elem_tuple(self):
        def unwrap_result_test(fx_graph, was_unwrapped: bool):
            self.assertTrue(was_unwrapped)
            result = fx_graph(torch.rand(10))
            self.assertIsInstance(result, torch.Tensor)
        self.maybe_unwrap_and_test_graph(lambda x: (x,), unwrap_result_test)


class TestTorchMLIRCompilerTupleReturn(unittest.TestCase):
    @staticmethod
    def compile_func_and_test_output(func, example_inputs, output_test):
        fx_graph = torch.fx.symbolic_trace(func)
        compiled_function = torch_mlir_compiler(fx_graph, example_inputs, use_tracing=True)
        result = compiled_function(*example_inputs)
        output_test(result)

    def test_no_tuple(self):
        def output_test(result):
            self.assertIsInstance(result, torch.Tensor)

        self.compile_func_and_test_output(lambda x: x, [torch.rand(10)], output_test)

    def test_zero_elem_tuple(self):
        def output_test(result):
            self.assertIsInstance(result, tuple)
            self.assertTrue(len(result) == 0)

        self.compile_func_and_test_output(lambda x: tuple(), [torch.rand(10)], output_test)

    def test_one_elem_tuple(self):
        def output_test(result):
            self.assertIsInstance(result, tuple)
            self.assertTrue(len(result) == 1)
            self.assertIsInstance(result[0], torch.Tensor)

        self.compile_func_and_test_output(lambda x: (x,), [torch.rand(10)], output_test)

    def test_two_elem_tuple(self):
        def output_test(result):
            self.assertIsInstance(result, tuple)
            self.assertTrue(len(result) == 2)
            self.assertIsInstance(result[0], torch.Tensor)
            self.assertIsInstance(result[1], torch.Tensor)

        self.compile_func_and_test_output(lambda x: (x, x), [torch.rand(10)], output_test)


if __name__ == "__main__":
    unittest.main()

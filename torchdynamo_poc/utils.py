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

import functools
import time
from typing import List, Optional
import torch

import torch_mlir
import iree_torch


DEVICE_TO_IREE_BACKEND = { "cpu" : "dylib",
                           "cuda" : "cuda" }


def timeit(*, append_time_to: Optional[List] = None):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time_ns()
            result = func(*args, **kwargs)
            end_time = time.time_ns()

            if append_time_to is not None:
                append_time_to.append(end_time - start_time)
            return result
        return wrapper
    return decorator


def _returns_nothing(fx_g: torch.fx.GraphModule) -> bool:
    for node in fx_g.graph.nodes:
        if node.op == "output":
            assert len(node.args) == 1, "Output node must have a single argument"
            node_arg = node.args[0]
            if isinstance(node_arg, tuple):
                return len(node_arg) == 0
    return False


def _unwrap_single_tuple_return(fx_g: torch.fx.GraphModule) -> Optional[torch.fx.GraphModule]:
    """Replace tuple with tuple element in functions that return one-element tuples."""
    unwrapped_tuple = False
    for node in fx_g.graph.nodes:
        if node.op == "output":
            assert len(node.args) == 1, "Output node must have a single argument"
            node_arg = node.args[0]
            if isinstance(node_arg, tuple):
                if len(node_arg) == 1:
                    node.args = (node_arg[0],)
                    unwrapped_tuple = True
                else:
                    return None

    if not unwrapped_tuple:
        return None

    fx_g.graph.lint()
    fx_g.recompile()
    return fx_g


def torch_mlir_compiler(fx_graph: torch.fx.GraphModule,
                        example_inputs: List[torch.Tensor],
                        use_tracing: bool, device: str):
    """Compile GraphModule using torch-mlir + IREE."""
    if _returns_nothing(fx_graph):
        return fx_graph

    fx_graph_unwrapped = _unwrap_single_tuple_return(fx_graph)
    was_unwrapped = fx_graph_unwrapped is not None
    fx_graph = fx_graph_unwrapped if was_unwrapped else fx_graph
    ts_compiler = torch.jit.trace if use_tracing else torch.jit.script
    ts_graph = ts_compiler(fx_graph, example_inputs)
    linalg_module = torch_mlir.compile(ts_graph, example_inputs,
                                       output_type=torch_mlir.OutputType.LINALG_ON_TENSORS)
    backend = DEVICE_TO_IREE_BACKEND[device]
    compiled_module = iree_torch.compile_to_vmfb(linalg_module, backend)
    loaded_module = iree_torch.load_vmfb(compiled_module, backend)

    def forward(*inputs):
        result = loaded_module.forward(*inputs)
        result = tuple() if result is None else result
        return (result,) if was_unwrapped else result
    return forward


def check_results(compiled_results, eager_results):
    for compiled_result, eager_result in zip(compiled_results, eager_results):
        if abs(torch.mean(compiled_result) - torch.mean(eager_result)) > 0.001:
            print("Compiled result does not match eager result")
            return
    print("Compiled result matches eager result!")


def print_time_stats(times, *, warmup_iters: int = 0):
    iter_times = torch.tensor(times[warmup_iters:])
    print(f"Mean: {torch.mean(iter_times.to(float))} ns")
    print(f"STD: {torch.std(iter_times.to(float))} ns")
    print(f"Total: {torch.sum(iter_times)} ns")
    print()

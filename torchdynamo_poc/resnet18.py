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
"""
Example script for running Resnet18 on IREE using Torchdynamo.

To run this script, add the following to you PYTHONPATH:

- path-to/torch-mlir/build/tools/torch-mlir/python_packages/torch_mlir
- path-to/torch-mlir/examples
- path-to/iree-torch/python

and make sure that torchdynamo is installed by following their instructions:
https://github.com/pytorch/torchdynamo#requirements-and-setup

Command to run: python resnet18.py
"""

import sys
from typing import List

import torch
from torchvision import models
import torch_mlir
import iree_torch
from torchscript_resnet18 import load_and_preprocess_image, load_labels, top3_possibilities
import torchdynamo


def _unwrap_single_tuple_return(fx_g: torch.fx.GraphModule):
    """Replace tuple with tuple element in functions that return one-element tuples."""

    for node in fx_g.graph.nodes:
        if node.op == "output":
            assert len(node.args) == 1, "Output node must have a single argument"
            node_arg = node.args[0]
            if isinstance(node_arg, tuple) and len(node_arg) == 1:
                node.args = (node_arg[0],)
    fx_g.graph.lint()
    fx_g.recompile()
    return fx_g


def torch_mlir_compiler(fx_graph: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    """Compile GraphModule using torch-mlir + IREE."""

    fx_graph = _unwrap_single_tuple_return(fx_graph)
    ts_graph = torch.jit.script(fx_graph)
    linalg_module = torch_mlir.compile(ts_graph, example_inputs,
                                       output_type=torch_mlir.OutputType.LINALG_ON_TENSORS)
    compiled_module = iree_torch.compile_to_vmfb(linalg_module)
    loaded_module = iree_torch.load_vmfb(compiled_module)
    def forward(*inputs):
        return (loaded_module.forward(*inputs),)
    return forward


if __name__ == "__main__":
    IMAGE_URL = "https://upload.wikimedia.org/wikipedia/commons/2/26/YellowLabradorLooking_new.jpg"
    print(f"loading image from {IMAGE_URL}", file=sys.stderr)
    img = load_and_preprocess_image(IMAGE_URL)
    labels = load_labels()

    resnet18 = models.resnet18(pretrained=True)
    resnet18.train(False)

    with torchdynamo.optimize(torch_mlir_compiler):
        result = resnet18.forward(img)
    print(top3_possibilities(result))

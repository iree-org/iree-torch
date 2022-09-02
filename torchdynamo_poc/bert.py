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

from typing import List, Callable
import argparse

import torch
import torchdynamo
from transformers import BertConfig, AutoModelForMaskedLM

from utils import check_results, print_time_stats, torch_mlir_compiler, timeit


def run(func: Callable[[], List[torch.Tensor]], iters):
    """Run a function a number of times."""
    results = []
    for _ in range(iters):
        results += func()
    return results


def benchmark_model(model, input_tensor, compiler, device, iters):
    model_device = model.to(device)
    input_device = input_tensor.to(device)

    iteration_times = []
    @timeit(append_time_to=iteration_times)
    def run_model_compiled():
        with torchdynamo.optimize(compiler):
            return model_device.forward(input_device)["logits"]

    torchdynamo.reset()
    compiled_results = run(run_model_compiled, iters)
    return compiled_results, iteration_times


def main():
    parser = argparse.ArgumentParser(description="Run Bert using Torch-MLIR + IREE.")
    parser.add_argument("--iters", type=int, default=1,
                        help="Number of iterations to run model for.")
    parser.add_argument("--warmup-iters", type=int, default=0,
                        help="Number of iterations to run model for warmup.")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cpu",
                        help="Device to run model on.")
    args = parser.parse_args()

    max_length = 128
    vocab_size = 3200
    input_tensor = torch.randint(0, vocab_size, (1, max_length))
    config = BertConfig(vocab_size=vocab_size)
    model = AutoModelForMaskedLM.from_config(config)
    model.eval()

    def compiler(graph, inputs):
        return torch_mlir_compiler(graph, inputs, use_tracing=True, device=args.device)

    total_iters = args.warmup_iters + args.iters
    compiled_results, compiled_iteration_times = benchmark_model(
        model, input_tensor, compiler, "cpu", total_iters)
    eager_results, eager_iteration_times = benchmark_model(
        model, input_tensor, "eager", args.device, total_iters)
    print("Compiled iteration times")
    print_time_stats(compiled_iteration_times[args.warmup_iters:])
    print("Eager iteration times")
    print_time_stats(eager_iteration_times[args.warmup_iters:])

    check_results(compiled_results, eager_results)


if __name__ == "__main__":
    main()

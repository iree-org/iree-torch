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
import torch
import torchdynamo
from transformers import BertConfig, AutoModelForMaskedLM

from utils import check_results, print_time_stats, torch_mlir_compiler, timeit


def run(func: Callable[[], List[torch.Tensor]], num_iter):
    """Run a function a number of times."""
    results = []
    for _ in range(num_iter):
        results += func()
    return results


def main():
    max_length = 128
    vocab_size = 2
    input_tensor = torch.randint(0, vocab_size, (1, max_length))
    config = BertConfig(vocab_size=vocab_size)
    model = AutoModelForMaskedLM.from_config(config)
    model.eval()

    model_cpu = model.to("cpu")
    input_cpu = input_tensor.to("cpu")

    # TODO: make this an argument to the script
    device = "cpu"
    model_device = model.to(device)
    input_device = input_tensor.to(device)

    def compiler(graph, inputs):
        return torch_mlir_compiler(graph, inputs, use_tracing=True, device=device)

    compiled_iteration_times = []
    eager_iteration_times = []

    @timeit(append_time_to=compiled_iteration_times)
    def run_model_compiled():
        with torchdynamo.optimize(compiler):
            return model_cpu.forward(input_cpu)["logits"]

    @timeit(append_time_to=eager_iteration_times)
    def run_model_eager():
        with torchdynamo.optimize("eager"):
            return model_device.forward(input_device)["logits"]

    # TODO: make `num_iters` and `warmup_iters` argparse parameters
    compiled_results = run(run_model_compiled, num_iter=10)
    eager_results = run(run_model_eager, num_iter=10)

    print("Compiled iteration times")
    print_time_stats(compiled_iteration_times, warmup_iters=5)
    print("Eager iteration times")
    print_time_stats(eager_iteration_times, warmup_iters=5)

    check_results(compiled_results, eager_results)


if __name__ == "__main__":
    main()

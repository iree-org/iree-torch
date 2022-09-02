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
Script for running torchbenchmark models using torch-mlir + IREE.

Run `python torchbench.py -h` for more info.
"""
import argparse
import sys
from typing import List, Callable
import torch
from torchbenchmark import load_model_by_name
import torchdynamo

from utils import check_results, print_time_stats, torch_mlir_compiler, timeit


COMPILED_ITERATION_TIMES = []
EAGER_ITERATION_TIMES = []


def run(func: Callable[[], List[torch.Tensor]], num_iter):
    """Run a function a number of times."""
    results = []
    for _ in range(num_iter):
        results += func()
    return results


def main():
    parser = argparse.ArgumentParser(description="Run Torchbench using Torch-MLIR + IREE.")
    parser.add_argument("model", help="Model to run in torchbenchmark.")
    parser.add_argument("--train", action="store_true", help="Run model in training mode.")
    parser.add_argument("--iters", type=int, default=1,
                        help="Number of iterations to run model for.")
    parser.add_argument("--warmup-iters", type=int, default=0,
                        help="Number of iterations to run model for warmup.")
    parser.add_argument("--batchsize", type=int, default=0,
                        help="Batch size to use in model.")
    parser.add_argument("--trace", action="store_true", help="Use torch.jit.trace on model.")
    parser.add_argument("--exit-on-error", action="store_true", help="Exit on compiler error.")
    parser.add_argument("--check-with-eager", action="store_true",
                        help="Verify results with PyTorch eager-mode.")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cpu",
                        help="Device to run model on.")
    args = parser.parse_args()

    Model = load_model_by_name(args.model)
    if not Model:
        print(f"Model {args.model} not found in torchbench.")
        return

    test = "train" if args.train else "eval"
    # Use `device=cpu` for the model passed to IREE
    # TODO: This is not completely safe because depending on the model,
    # the input might be of a different shape for CPU.
    # See: https://github.com/pytorch/benchmark/blob/1ec85e202ff7a85ff85af49ac9c5c51d712ae10c/torchbenchmark/util/framework/huggingface/model_factory.py#L73
    model = Model(device="cpu", test=test, jit=False, batch_size=args.batchsize)
    print(f"Running model {args.model}")

    def compiler(graph, inputs):
        if args.exit_on_error:
            try:
                return torch_mlir_compiler(graph, inputs, args.trace, args.device)
            except Exception as err:
                print(err)
                sys.exit(1)
        return torch_mlir_compiler(graph, inputs, args.trace, args.device)

    @timeit(append_time_to=COMPILED_ITERATION_TIMES)
    def run_model_compiled():
        with torchdynamo.optimize(compiler):
            return list(model.invoke())

    total_iters = args.warmup_iters + args.iters
    compiled_results = run(run_model_compiled, total_iters)
    print("Compiled iteration times")
    print_time_stats(COMPILED_ITERATION_TIMES[args.warmup_iters:])

    if args.check_with_eager:
        if args.device != "cpu":
            model = Model(device=args.device, test=test, jit=False,
                          batch_size=args.batchsize)

        @timeit(append_time_to=EAGER_ITERATION_TIMES)
        def run_model_eager():
            with torchdynamo.optimize("eager"):
                return list(model.invoke())
        torchdynamo.reset()
        eager_results = run(run_model_eager, total_iters)
        print("Eager iteration times")
        print_time_stats(EAGER_ITERATION_TIMES[args.warmup_iters:])
        check_results(compiled_results, eager_results)


if __name__ == "__main__":
    main()

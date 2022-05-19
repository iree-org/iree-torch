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

import argparse

import torch
import torch_mlir
import iree_torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification


def prepare_sentence_tokens(hf_model: str, sentence: str):
    tokenizer = AutoTokenizer.from_pretrained(hf_model)
    return torch.tensor([tokenizer.encode(sentence)])


class OnlyLogitsHuggingFaceModel(torch.nn.Module):
    """Wrapper that returns only the logits from a HuggingFace model."""

    def __init__(self, model_name: str):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,  # The pretrained model name.
            # The number of output labels--2 for binary classification.
            num_labels=2,
            # Whether the model returns attentions weights.
            output_attentions=False,
            # Whether the model returns all hidden-states.
            output_hidden_states=False,
            torchscript=True,
        )
        self.model.eval()

    def forward(self, input):
        # Return only the logits.
        return self.model(input)[0]


def _suppress_warnings():
    import warnings
    warnings.simplefilter("ignore")
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "true"


def _get_argparse():
    parser = argparse.ArgumentParser(
        description="Run a HuggingFace BERT Model.")
    parser.add_argument("--model-name",
                        default="philschmid/MiniLM-L6-H384-uncased-sst2",
                        help="The HuggingFace model name to use.")
    parser.add_argument("--sentence",
                        default="The quick brown fox jumps over the lazy dog.",
                        help="sentence to run the model on.")
    iree_backend_choices = ["dylib", "vmvx", "vulkan", "cuda"]
    parser.add_argument("--iree-backend",
        choices=iree_backend_choices,
        default="dylib",
        help=f"""
Meaning of options:
dylib - cpu, native code
vmvx - cpu, interpreted
vulkan - GPU for general GPU devices
cuda - GPU for NVIDIA devices
""")
    return parser


def main():
    _suppress_warnings()
    args = _get_argparse().parse_args()
    print("Parsing sentence tokens.")
    example_input = prepare_sentence_tokens(args.model_name, args.sentence)
    print("Instantiating model.")
    model = OnlyLogitsHuggingFaceModel(args.model_name)

    # TODO: Wrap up all these steps into a convenient, well-tested API.
    # TODO: Add ability to run on IREE CUDA backend.
    print("Tracing model.")
    traced = torch.jit.trace(model, example_input)
    print("Compiling with Torch-MLIR")
    linalg_on_tensors_mlir = torch_mlir.compile(traced, example_input,
                                                output_type=torch_mlir.OutputType.LINALG_ON_TENSORS)
    print("Compiling with IREE")
    iree_vmfb = iree_torch.compile_to_vmfb(linalg_on_tensors_mlir, args.iree_backend)
    print("Loading in IREE")
    invoker = iree_torch.load_vmfb(iree_vmfb, args.iree_backend)
    print("Running on IREE")
    import time
    start = time.time()
    result = invoker.forward(example_input)
    end = time.time()
    print("RESULT:", result)
    print(f"Model execution took {end - start} seconds.")


if __name__ == "__main__":
    main()

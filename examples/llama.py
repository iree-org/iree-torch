import io
from pathlib import Path
import argparse

import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
import torch_mlir

class LLaMA(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = LlamaForCausalLM.from_pretrained(path)

    def forward(self, input_tensor):
        return self.model.generate(input_tensor, max_length=10, top_p=0.95, top_k=None)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True, help="MLIR file to save bytecode into")
    parser.add_argument("--weights-dir", type=Path, required=True,
                        help="Directory of LLaMA weights in Hugging Face Transformers format. \
                        See: https://huggingface.co/docs/transformers/main/en/model_doc/llama#overview \
                        for how to convert")
    args = parser.parse_args()

    prompt = "Hello world"
    tokenizer = LlamaTokenizer.from_pretrained(args.weights_dir)
    inputs = tokenizer(prompt, return_tensors="pt")
    model = LLaMA()
    mlir = torch_mlir.compile(model, inputs.input_ids, output_type="linalg-on-tensors",
                              use_tracing=True, use_external_references_if_numel_exceeds=1)

    with open(args.output, "bw") as f:
        bytecode_stream = io.BytesIO()
        mlir.operation.write_bytecode(bytecode_stream)
        f.write(bytecode_stream.getbuffer())


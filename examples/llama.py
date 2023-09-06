import io
from pathlib import Path
import argparse

import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
import torch_mlir
import iree.compiler as ireec
import iree.runtime as ireert
import numpy as np


class LLaMA(torch.nn.Module):
    def __init__(self, weights_dir):
        super().__init__()
        self.model = LlamaForCausalLM.from_pretrained(weights_dir)

    def forward(self, input_tensor):
        return self.model.generate(input_tensor, max_length=10, top_p=0.95, top_k=None)


def compile_and_load(mlir: bytes):
    iree_backend = "llvm-cpu"
    iree_input_type = "tm_tensor"
    iree_vmfb = ireec.compile_str(
        mlir, target_backends=[iree_backend], input_type=iree_input_type
    )

    config = ireert.Config(driver_name="local-sync")
    ctx = ireert.SystemContext(config=config)
    vm_module = ireert.VmModule.from_flatbuffer(ctx.instance, iree_vmfb)
    ctx.add_vm_module(vm_module)
    invoker = ctx.modules.module
    return invoker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        required=False,
        help="MLIR file to save linalg bytecode output into",
    )
    parser.add_argument(
        "--weights-dir",
        type=Path,
        required=True,
        help="Directory of LLaMA weights in Hugging Face Transformers format. \
                        See: https://huggingface.co/docs/transformers/main/en/model_doc/llama#overview \
                        for how to convert",
    )
    parser.add_argument(
        "--use-linalg-mlir",
        type=Path,
        required=False,
        help="Linalg MLIR bytecode to load into IREE. If this is not specified, \
        Torch-MLIR will be used to generate the linalg MLIR",
    )
    parser.add_argument(
        "--do-not-invoke-iree",
        action=argparse.BooleanOptionalAction,
        help="Run script skipping the parts that invoke IREE. \
        Useful when wanting to generate linalg MLIR for LLaMA without compiling and running it",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    prompt = "Hello world"
    tokenizer = LlamaTokenizer.from_pretrained(args.weights_dir)
    inputs = tokenizer(prompt, return_tensors="pt")

    if args.use_linalg_mlir is None:
        model = LLaMA(args.weights_dir)
        mlir = torch_mlir.compile(
            model, inputs.input_ids, output_type="linalg-on-tensors", use_tracing=True
        )
        bytecode_stream = io.BytesIO()
        mlir.operation.write_bytecode(bytecode_stream)
        mlir_bytes = bytecode_stream.getbuffer()
    else:
        with open(args.use_linalg_mlir, "br") as f:
            mlir_bytes = f.read()

    if args.output is not None:
        with open(args.output, "bw") as f:
            f.write(mlir_bytes)

    if not args.do_not_invoke_iree:
        invoker = compile_and_load(mlir_bytes)
        input_ids = inputs.input_ids.numpy()
        result = invoker.forward(input_ids)
        generated_ids = np.asarray(result)
        generated_text = tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        print("Input text:", prompt)
        print("Input IDs:", input_ids)
        print("Generated IDs:", generated_ids)
        print("Generated text:", generated_text)

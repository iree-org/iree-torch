import io
import os
import argparse
from pathlib import Path

from auto_gptq import AutoGPTQForCausalLM
import torch
from torch.fx.experimental.proxy_tensor import make_fx
from torch._subclasses.fake_tensor import FakeTensorMode
from torch._functorch.compile_utils import strip_overloads
import torch_mlir

parser = argparse.ArgumentParser()
parser.add_argument(
    "output",
    type=Path,
    help="MLIR file to save linalg bytecode output into",
)
args = parser.parse_args()

model_name_or_path = f"{os.path.dirname(__file__)}/falcon-7b-instruct-GPTQ"
model = AutoGPTQForCausalLM.from_quantized(
    model_name_or_path,
    use_safetensors=True,
    trust_remote_code=True,
    device="cpu",
    disable_exllamav2=True,
    quantize_config=None,
)

with FakeTensorMode(allow_non_fake_inputs=True):
    input_ids = torch.randint(low=1, high=10000, size=(1, 100))
    model_fx = make_fx(lambda x: model(x).logits, tracing_mode="fake")(input_ids)
    strip_overloads(model_fx)
    mlir = torch_mlir.compile(model_fx, input_ids, output_type="linalg-on-tensors")

bytecode_stream = io.BytesIO()
mlir.operation.write_bytecode(bytecode_stream)
mlir_bytes = bytecode_stream.getbuffer()
with open(args.output, "bw") as f:
    f.write(mlir_bytes)

import io
import logging

import torch
from torch.fx.experimental.proxy_tensor import make_fx
from torch._subclasses.fake_tensor import FakeTensorMode
from torch._functorch.compile_utils import strip_overloads
import torch_mlir
from transformers import AutoModelForCausalLM

logging.basicConfig(level=logging.INFO)

PATH = "/home/ramiroleal/falcon/quantization-examples/falcon-7b-int8-gptq"
OUTPUT = "/tmp/falcon-int8-raw.mlir"
OUTPUT_STR = "/tmp/falcon-int8-raw-str.mlir"
model = AutoModelForCausalLM.from_pretrained(PATH)

INPUT_SIZE = (1, 100)
for module in model.modules():
    if hasattr(module, "unpack"):
        print(f"Calling {module}.unpack()")
        module.unpack()

        x = torch.rand((1, 1, module.infeatures), dtype=torch.float16)
        new = module.forward(x)
        old = module.forward_old(x)

        if not torch.allclose(new, old):
            print("Max:", torch.max(torch.abs(new - old)))
            print("STD:", torch.std(new - old))
            print("Mean:", torch.mean(new - old))
            print(
                "Corr:",
                torch.corrcoef(
                    torch.stack(
                        [
                            new.flatten().to(torch.float32),
                            old.flatten().to(torch.float32),
                        ]
                    )
                ),
            )

assert False
with FakeTensorMode(allow_non_fake_inputs=True):
    input_ids = torch.randint(low=1, high=10000, size=INPUT_SIZE)
    model_fx = make_fx(lambda x: model(x).logits, tracing_mode="fake")(input_ids)
    strip_overloads(model_fx)
    mlir = torch_mlir.compile(model_fx, input_ids, output_type="torch")

# with open(OUTPUT_STR, "w") as f:
#     f.write(str(mlir))

bytecode_stream = io.BytesIO()
mlir.operation.write_bytecode(bytecode_stream)
mlir_bytes = bytecode_stream.getbuffer()
with open(OUTPUT, "bw") as f:
    f.write(mlir_bytes)

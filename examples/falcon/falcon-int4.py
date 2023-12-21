import io
import logging

import torch
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch._subclasses.fake_tensor import FakeTensorMode
from torch._functorch.compile_utils import strip_overloads
import torch_mlir
from transformers import AutoModelForCausalLM

logging.basicConfig(level=logging.INFO)

PATH = "/home/ramiroleal/falcon/quantization-examples/falcon-7b-int4-gptq"
OUTPUT = "/tmp/falcon-int8-raw.mlir"
model = AutoModelForCausalLM.from_pretrained(PATH)

INPUT_SIZE = (1, 100)
for module in model.modules():
    if hasattr(module, "unpack"):
        print(f"Calling {module}.unpack()")
        module.unpack()

        continue
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


def add_cast_to_uint4(gm):
    for node in gm.graph.nodes:
        if node.target == torch.ops.aten.bitwise_not:
            node.target = torch.ops.autogptq.cast_to_uint4
    gm.recompile()

shape_env = ShapeEnv()
with FakeTensorMode(allow_non_fake_inputs=True, shape_env=shape_env):
    input_ids = torch.randint(low=1, high=10000, size=INPUT_SIZE)
    torch._dynamo.allow_in_graph(torch.ops.autogptq.cast_to_uint4.default)
    from auto_gptq.nn_modules.qlinear.qlinear_cuda_old import cast_to_uint4
    torch.fx.wrap(cast_to_uint4)
    #torch._dynamo.allow_in_graph('cast_to_uint4')
    model_fx = make_fx(lambda x: model(x).logits, tracing_mode="symbolic", pre_dispatch=False)(input_ids)
    #model_fx_2 = make_fx(model_fx, tracing_mode="symbolic", pre_dispatch=False)(input_ids)
    strip_overloads(model_fx)
    add_cast_to_uint4(model_fx)
#    def model_call(x):
#        return model(x).logits
#    model_fx = torch.export.export(model_call, (input_ids,))
    mlir = torch_mlir.compile(model_fx, input_ids, output_type="raw")

# with open(OUTPUT_STR, "w") as f:
#     f.write(str(mlir))

bytecode_stream = io.BytesIO()
mlir.operation.write_bytecode(bytecode_stream)
mlir_bytes = bytecode_stream.getbuffer()
with open(OUTPUT, "bw") as f:
    f.write(mlir_bytes)

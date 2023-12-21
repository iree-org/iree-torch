# Example from Huggingface docs:
# https://huggingface.co/docs/transformers/quantization

from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig

model_id = "tiiuae/falcon-7b"
tokenizer = AutoTokenizer.from_pretrained(model_id)
# Using group_size=64 because that is what TheBloke used for
# Falcon-7B-Instruct-GPTQ and because using group_size=128 results in
# an error in AutoGPTQ.
gptq_config = GPTQConfig(bits=4, dataset="c4", tokenizer=tokenizer, group_size=64, disable_exllama=True)

quantized_model = AutoModelForCausalLM.from_pretrained(
    model_id, device_map="auto", quantization_config=gptq_config
)

quantized_model.to("cpu")
quantized_model.save_pretrained("falcon-7b-int4-gptq")

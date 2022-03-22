# Torch frontend for IREE.

This project provides end-to-end flows supporting users of PyTorch that want to target IREE as a compiler backend. We use the [Torch-MLIR](https://github.com/llvm/torch-mlir) project to provide our PyTorch frontend.

# Planned features

- Python (or, if absolutely necessary, C++) code that pulls in the bindings   from both projects into an end-to-end flow for users.
- Docker images for users to be able to quickly get started
- CI of the Torch-MLIR end-to-end tests, with IREE plugged in as a backend
- User examples:
  - Jupyter notebooks using the above to demonstrate interactive use of the tools
  - Standalone user-level Python code demonstrating various deployment flows (mobile, embedded).

# Running end-to-end correctness tests

Setup the venv for running:

```
# Install torch-mlir dependencies:
(iree-samples.venv) $ pip install --pre --upgrade torch torchvision -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html
# Set up PYTHONPATH pointing at a built torch-mlir:
(iree-samples.venv) $ export PYTHONPATH="${PYTHONPATH}:${TORCHMLIR_SRC_ROOT}/build/tools/torch-mlir/python_packages/torch_mlir"
```

Run the torch-mlir e2e test suite for TorchScript:
```
(iree-samples.venv) $ "${TORCHMLIR_SRC_ROOT}/tools/torchscript_e2e_test.sh" -c external --external-config "${IREE_SAMPLES_SRC_ROOT}/iree-torch/torchscript_e2e_config.py"
```

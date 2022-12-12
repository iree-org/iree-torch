# Torch Frontend for IREE

This project provides end-to-end flows supporting users of PyTorch that want to target [IREE](https://iree-org.github.io/iree/) as a compiler backend, which offers a [number of benefits](https://iree-org.github.io/iree/#key-features). We use the [Torch-MLIR](https://github.com/llvm/torch-mlir) project to provide our PyTorch frontend.

This project is under active development and is subject to frequent changes.

# Example Usage

## Training & Inference (`functorch`-based)

An end-to-end example of training a PyTorch basic regression model on IREE can be found in [this script](https://github.com/iree-org/iree-torch/blob/main/examples/regression.py).  This script uses `functorch` to define the model's forward and backward pass.

## Inference (`nn.Module`-based)

An end-to-end example of compiling an `nn.Module`-based PyTorch BERT model to IREE can be found in [this notebook](https://github.com/iree-org/iree-torch/blob/main/examples/bert.ipynb). The notebook also demonstrates the significantly smaller runtime size of the compiled model when compared to PyTorch (**~4MB versus ~700MB**).

## Native, On-device Training

A small (~100-250KB), self-contained binary can be built for deploying to resource-constrained environments. An example illustrating this can be found in [this example](https://github.com/iree-org/iree-torch/tree/main/examples/native_training). This binary runs a model without a Python interpreter.

# Planned features

- Python (or, if absolutely necessary, C++) code that pulls in the bindings from both projects into an end-to-end flow for users.
- Docker images for users to be able to quickly get started
- CI of the Torch-MLIR end-to-end tests, with IREE plugged in as a backend
- User examples:
  - Jupyter notebooks using the above to demonstrate interactive use of the tools
  - Standalone user-level Python code demonstrating various deployment flows (mobile, embedded).

# Running end-to-end correctness tests

Setup the venv for running:

```bash
# Create a Python virtual environment.
$ python -m venv iree-torch.venv
$ source iree-torch.venv/bin/activate

# Option 1: Install Torch-MLIR and IREE from nightly packages:
(iree-torch.venv) $ python -m pip install -r "${IREE_TORCH_SRC_ROOT}/requirements.txt"

# Option 2: For development, build from source and set `PYTHONPATH`:
ninja -C "${TORCH_MLIR_BUILD_ROOT}" TorchMLIRPythonModules
ninja -C "${IREE_BUILD_ROOT}" IREECompilerPythonModules bindings_python_iree_runtime_runtime
export PYTHONPATH="${IREE_BUILD_ROOT}/runtime/bindings/python:${IREE_BUILD_ROOT}/compiler/bindings/python:${TORCH_MLIR_BUILD_ROOT}/tools/torch-mlir/python_packages/torch_mlir:${PYTHONPATH}"
```

Run the Torch-MLIR TorchScript e2e test suite on IREE:
```bash
# Run all the tests on the default backend (`llvm-cpu`).
(iree-torch.venv) $ tools/e2e_test.sh
# Run all tests on the `vmvx` backend.
(iree-torch.venv) $ tools/e2e_test.sh --config vmvx
# Filter the tests (with a regex) and report failures with verbose error messages.
# This is good for drilling down on a single test as well.
(iree-torch.venv) $ tools/e2e_test.sh --filter Elementwise --verbose
# Shorter option names.
(iree-torch.venv) $ tools/e2e_test.sh -f Elementwise -v
```

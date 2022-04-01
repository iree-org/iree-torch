# Torch frontend for IREE [Under Construction]

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
# TODO: Make the torch-mlir package automatically install the dependencies.
(iree-torch.venv) $ python -m pip install -r "${TORCH_MLIR_SRC_ROOT}/requirements.txt"

# Option 1: Install Torch-MLIR and IREE from nightly packages:
(iree-torch.venv) $ python -m pip install iree-compiler iree-runtime -f https://github.com/google/iree/releases
(iree-torch.venv) $ python -m pip install torch-mlir -f https://github.com/llvm/torch-mlir/releases

# Option 2: For development, build from source and set `PYTHONPATH`:
ninja -C "${TORCH_MLIR_BUILD_ROOT}" TorchMLIRPythonModules
ninja -C "${IREE_BUILD_ROOT}" IREECompilerPythonModules bindings_python_iree_runtime_runtime
export PYTHONPATH="${IREE_BUILD_ROOT}/bindings/python:${IREE_BUILD_ROOT}/compiler-api/python_package:${TORCH_MLIR_BUILD_ROOT}/tools/torch-mlir/python_packages/torch_mlir:${PYTHONPATH}"
```

Run the Torch-MLIR TorchScript e2e test suite on IREE
```
# Run all the tests on the default backend (`dylib`).
(iree-torch.venv) $ python torchscript_e2e_main.py
# Run all tests on the `vmvx` backend.
(iree-torch.venv) $ python torchscript_e2e_main.py --config vmvx
# Filter the tests (with a regex) and report failures with verbose error messages.
# This is good for drilling down on a single test as well.
(iree-torch.venv) $ python torchscript_e2e_main.py --filter Elementwise --verbose
```

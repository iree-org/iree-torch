# Native Training Example

This example shows how to:

1. Build a PyTorch functional model for training
2. Import that model into IREE's compiler
3. Compile that model to an IREE VM bytecode module
4. Load the compiled module using IREE's high level runtime C API into a
   lightweight program
5. Train the loaded model

This example was built with the goal of allowing you to be able to build it
outside this repo in your own project with minimal changes.

The weights for the model are stored in the program itself and updated in
memory. This can be modified to be stored however you see fit.

The binary output by this example is small (100-250KB, depending on platform).

## Running the Example

Install `iree-torch` and other dependencies necessary for this example.
[iree-torch](https://github.com/iree-org/iree-torch) provides a number of
convenient wrappers around `torch-mlir` and `iree` compilation:

> **Note**
> We recommend installing Python packages inside a
> [virtual environment](https://docs.python.org/3/tutorial/venv.html).

```shell
pip install -f https://iree-org.github.io/iree/pip-release-links.html iree-compiler
pip install -f https://llvm.github.io/torch-mlir/package-index/ torch-mlir
pip install git+https://github.com/iree-org/iree-torch.git
```

Update submodules in this repo:

```shell
(cd $(git rev-parse --show-toplevel) && git submodule update --init --recursive)
```

Make sure you're in this example's directory:

```shell
cd $(git rev-parse --show-toplevel)/examples/native_training
```

Build the IREE runtime and native training example:

```shell
$ cmake -B build/ -DCMAKE_BUILD_TYPE=MinSizeRel -GNinja .
$ cmake --build build/ --target native_training
```
Generate the IREE VM bytecode for the model:

```shell
python native_training.py /tmp/native_training.vmfb
```

Run the native training model:

```shell
./native-training /tmp/native_training.vmfb
```

## Binary Size

This should produce a small, 100-250KB binary (depending on platform):

```bash
$ ls -lha build | grep native_training
-rwxr-x--- 1 me primarygroup 257K Dec  8 18:46 native_training
```
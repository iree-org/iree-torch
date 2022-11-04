# Embedded Training Example

This is a fork of IREE's [simple embedding example](https://github.com/iree-org/iree/tree/main/samples/simple_embedding).

Unlike that example, this example:

 * Uses PyTorch to generate the model
 * Uses Torch-MLIR to compile the model
 * Trains the model directly in the embedded binary

See the [upstream example](https://github.com/iree-org/iree/tree/main/samples/simple_embedding)
for more information on the supported execution environments.

## Build instructions

### IREE

Follow the [getting started steps for IREE](https://iree-org.github.io/iree/building-from-source/getting-started/).
Note the build directory you choose, as this will be the directory you will
pass into the next step.

### CMake (native and cross compilation)

Next, build the binaries:

```sh
cmake --build <build dir> --target examples/embedded/all
```

The resulting executables are listed as `embedded_<HAL devices>`.

## Code structure

### embedded.py

Creates a Linear Regression PyTorch model and outputs it as MLIR.

### embedded_test.mlir

The saved MLIR representation of the model.

### simple_embedding.c

Trains the model, outputting the updated weights and loss on each iteration.

### Other Files

See the [upstream example](https://github.com/iree-org/iree/tree/main/samples/simple_embedding)
for the files not covered here.

import argparse

import functorch
from functorch._src.compile_utils import strip_overloads
import iree_torch
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import torch
import torch_mlir


# Dataset
X, y, coef = make_regression(n_features=3, coef=True)
X = torch.from_numpy(X).to(dtype=torch.float32)
y = torch.from_numpy(y).to(dtype=torch.float32)
X, X_test, y, y_test = train_test_split(X, y)

# Weights
w = torch.zeros(X.shape[1:])
b = torch.tensor(0.)


def _get_argparse():
    parser = argparse.ArgumentParser(
        description="Train and run a regression model.")
    parser.add_argument("--iree-backend",
        default="llvm-cpu",
        help="See https://iree-org.github.io/iree/deployment-configurations/ "
             "for the full list of options.")
    return parser


def _suppress_warnings():
    import warnings
    warnings.simplefilter("ignore")
    import os


def forward(w, b, X):
    return torch.matmul(X, w) + b


def mse(y_pred, y):
    err = y_pred - y
    return torch.mean(torch.square(err))


def loss_fn(w, b, X, y):
    y_pred = forward(w, b, X)
    return mse(y_pred, y)


grad_fn = functorch.grad(loss_fn, argnums=(0, 1))


def update(w, b, grad_w, grad_b):
    new_w = w - grad_w * 0.05
    new_b = b - 0.05 * grad_b
    return new_w, new_b


def train(w, b, X, y):
    grad_w, grad_b = grad_fn(w, b, X, y)
    return update(w, b, grad_w, grad_b)


def main():
    global w, b, X_test, y_test
    _suppress_warnings()
    args = _get_argparse().parse_args()

    #
    # Training
    #
    print("Compiling training function with Torch-MLIR")
    train_args = (w, b, X_test, y_test)
    graph = functorch.make_fx(train)(*train_args)

    # BEFORE SUBMIT: Explain why we need to do this?  Embed this into
    # torch_mlir.compile as an argument?  Make it the default?
    strip_overloads(graph)

    linalg_on_tensors_mlir = torch_mlir.compile(
        graph,
        train_args,
        output_type=torch_mlir.OutputType.LINALG_ON_TENSORS,
        use_tracing=False)

    print("Loading into IREE")
    iree_vmfb = iree_torch.compile_to_vmfb(
        linalg_on_tensors_mlir, args.iree_backend)
    invoker = iree_torch.load_vmfb(iree_vmfb, args.iree_backend)

    print("Training on IREE")
    for _ in range(30):
        w, b = invoker.forward(*train_args)
        train_args = (w, b, X_test, y_test)
        print("Loss:", loss_fn(w, b, X_test, y_test))
    print()

    #
    # Inference
    #
    print("Compiling inference function with Torch-MLIR")
    graph = functorch.make_fx(forward)(*train_args[:3])
    strip_overloads(graph)
    linalg_on_tensors_mlir = torch_mlir.compile(
        graph,
        train_args[:3],
        output_type=torch_mlir.OutputType.LINALG_ON_TENSORS,
        use_tracing=False)

    print("Loading inference function into IREE")
    iree_vmfb = iree_torch.compile_to_vmfb(
        linalg_on_tensors_mlir, args.iree_backend)
    invoker = iree_torch.load_vmfb(iree_vmfb, args.iree_backend)

    print("Running inference on IREE")
    y_pred = invoker.forward(*train_args[:3])
    print("Actual output:", y_pred)
    print("Expected output:", train_args[3])
    print("MSE:", mse(y_pred, train_args[3]))


if __name__ == "__main__":
    main()
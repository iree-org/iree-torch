import argparse

import functorch
from functorch._src.compile_utils import strip_overloads
import iree_torch
import torch
import torch_mlir


NUM_TRAINING_SAMPLES = 1000


# Dataset
def make_training_data():
    coefficients = torch.tensor([3.0, 4.0, 5.0], dtype=torch.float32)
    bias = torch.tensor(6.0, dtype=torch.float32)
    X, y = [], []
    for i in range(NUM_TRAINING_SAMPLES):
        X.append(torch.rand(3))
        y_without_noise = torch.matmul(coefficients, X[-1]) + bias
        y.append(torch.normal(y_without_noise, std=0.1))
    return torch.stack(X), torch.stack(y)

X, y = make_training_data()
split_idx = int(NUM_TRAINING_SAMPLES * 0.9)
X, y, X_test, y_test = (X[:split_idx],
                        y[:split_idx],
                        X[split_idx:],
                        y[split_idx:])

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
    learning_rate = 0.05
    new_w = w - grad_w * learning_rate
    new_b = b - grad_b * learning_rate
    return new_w, new_b


def train(w, b, X, y):
    grad_w, grad_b = grad_fn(w, b, X, y)
    return update(w, b, grad_w, grad_b)


def main():
    global w, b, X, y
    _suppress_warnings()
    args = _get_argparse().parse_args()

    #
    # Training
    #
    print("Compiling training function with Torch-MLIR")
    train_args = (w, b, X, y)
    graph = functorch.make_fx(train)(*train_args)

    linalg_on_tensors_mlir = torch_mlir.compile(
        graph,
        train_args,
        output_type=torch_mlir.OutputType.LINALG_ON_TENSORS)

    print("Loading into IREE")
    iree_vmfb = iree_torch.compile_to_vmfb(
        linalg_on_tensors_mlir, args.iree_backend)
    invoker = iree_torch.load_vmfb(iree_vmfb, args.iree_backend)

    print("Training on IREE")
    for _ in range(30):
        w, b = invoker.forward(*train_args)
        train_args = (w, b, X, y)
        print("Loss:", loss_fn(w, b, X, y))
    print()

    #
    # Inference
    #
    print("Compiling inference function with Torch-MLIR")
    inference_args = (w, b, X_test)
    graph = functorch.make_fx(forward)(*inference_args)
    strip_overloads(graph)
    linalg_on_tensors_mlir = torch_mlir.compile(
        graph,
        inference_args,
        output_type="linalg-on-tensors")

    print("Loading inference function into IREE")
    iree_vmfb = iree_torch.compile_to_vmfb(
        linalg_on_tensors_mlir, args.iree_backend)
    invoker = iree_torch.load_vmfb(iree_vmfb, args.iree_backend)

    print("Running inference on IREE")
    y_pred = invoker.forward(*inference_args)
    print("Actual output (first 10):", y_pred[:10])
    print("Expected output (first 10):", train_args[3][:10])
    print("MSE:", mse(y_pred, y_test))


if __name__ == "__main__":
    main()
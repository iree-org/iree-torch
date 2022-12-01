import argparse

import functorch
from functorch._src.compile_utils import strip_overloads
import iree_torch
import torch
import torch_mlir


# Dataset
X = torch.tensor((
    (-0.14083656, 1.34831313, -0.4067572),
    (-2.90018955, -0.85372433, 0.76154407),
    (-0.94357427, -2.88796765, -1.16174819),
    (-0.23726768, -1.78838551, 0.32511812),
    (1.21389698, 0.73701257, -0.03101347),
), dtype=torch.float32)
y = torch.tensor((
    -35.37357027,
    -32.02257999,
    36.08020568,
    -177.70424428,
    -22.68094929,
), dtype=torch.float32)

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
    graph = functorch.make_fx(forward)(*train_args[:3])
    strip_overloads(graph)
    inference_args = train_args[:3]  # Remove the labels for inference.
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
    print("Actual output:", y_pred)
    print("Expected output:", train_args[3])
    print("MSE:", mse(y_pred, train_args[3]))


if __name__ == "__main__":
    main()
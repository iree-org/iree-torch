# TorchDynamo POC

Installation instruction

In a new venv:

1. `python -m pip install --upgrade pip`
2. In `iree-torch/torchdynamo_poc`
   - `python -m pip install -r requirements.txt`
3. Install torchdynamo
   - `git clone https://github.com/pytorch/torchdynamo.git`
   - `cd torchdynamo`
   - `python -m pip install -r requirements.txt`
   - `python setup.py develop`
4. Install benchmark
   - `git clone https://github.com/pytorch/benchmark.git`
   - `cd benchmark`
   - `python install.py --continue_on_fail`
   - Add this directory to your `PYTHONPATH` with: `echo $PWD > "$(python -c 'import site; print(site.getsitepackages()[0])')/torchbenchmark.pth"`
5. Install functorch
   - `python -m pip install "git+https://github.com/pytorch/functorch.git"`
6. Install iree-torch. From the `iree-torch` root dir:
   - `python setup.py develop`
   
If you want to use PyTorch+CUDA, first uninstall `torch`, `torchvision`, and
`torchtext`, then install the nightly CUDA version. For example, for CUDA 11.6:

```
python -m pip install --pre torch torchvision torchtext --extra-index-url https://download.pytorch.org/whl/nightly/cu116
```

# Running Torchbench Example

This POC includes a script that runs PyTorch benchmarks using TorchDynamo to
compile and run the models with IREE. An example for how to run the benchmark
for the `hf_Bert` model is:

```
python torchbench.py hf_Bert --trace --warmup-iters 5 --iters 10 --device=cuda
```

For more info on the other flags supported, run `python torchbench.py -h`

# Running Bert Example

This POC includes a script that runs HuggingFace's Bert for language modeling
using TorchDynamo to compile and run the model with IREE. The script compares
the performance against PyTorch eager. An example usage of the script is:

```
python bert.py --device cuda --iters 10 --warmup-iters 5
```

For more info on the flags supported, run `python bert.py -h`.

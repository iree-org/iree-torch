python -m pip install --pre torch-mlir \
  -f https://llvm.github.io/torch-mlir/package-index/ \
  --extra-index-url https://download.pytorch.org/whl/nightly/cpu
python -m pip install einops

# Nightly needed to get recent upstream fixes
git clone https://github.com/PanQiWei/AutoGPTQ.git
pushd AutoGPTQ
BUILD_CUDA_EXT=0 pip install -v .
popd

# Small patching needed. See `dtype-fix.diff`
git lfs install
git clone https://huggingface.co/TheBloke/falcon-7b-instruct-GPTQ
pushd falcon-7b-instruct-GPTQ
git apply ../dtype-fix.diff
popd

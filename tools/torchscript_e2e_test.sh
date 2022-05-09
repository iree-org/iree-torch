#!/bin/bash
set -euo pipefail

src_dir="$(realpath $(dirname $0)/..)"

cd "$src_dir"

# Ensure PYTHONPATH is set for export to child processes, even if empty.
export PYTHONPATH=${PYTHONPATH-}
export PYTHONPATH="$PYTHONPATH:${src_dir}/python"

python tools/torchscript_e2e_main.py "$@"

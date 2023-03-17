#!/bin/bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX
export PYTHONPATH=`pwd`
export PATH="$CONDA_PREFIX/lib:$PATH"

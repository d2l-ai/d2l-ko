#!/bin/bash
set -ex

if [ $# -ne 1 ]; then
    echo "Usage: $0 LANG"
    echo "LANG can be en, zh, ..."
    exit -1
fi

D2L_LANG=$1

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda/lib64
export CUDA_VISIBLE_DEVICES=$((EXECUTOR_NUMBER)),$((EXECUTOR_NUMBER+1))

conda activate d2l-${D2L_LANG}-build
pip list
pip uninstall -y mxnet-cu92
pip list

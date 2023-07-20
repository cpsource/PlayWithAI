#!/bin/bash
# get to right python venv
source ~/venv/bin/activate
# get TensorFlow setup properly
CUDNN_PATH=$(dirname $(python3 -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
export LD_LIBRARY_PATH=$CUDNN_PATH/lib:$LD_LIBRARY_PATH
# reduce TensorFlow logging level
 export TF_CPP_MIN_LOG_LEVEL=3
# display some things
./nvcc.sh
./verify-tensor-flow.sh

#!/bin/bash
# get to right python venv
source ~/venv/bin/activate
# set POLYGON_API_KEY
export POLYGON_API_KEY=`cat finance/polygon.io/apikey.txt`
# get TensorFlow setup properly
CUDNN_PATH=$(dirname $(python3 -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
export LD_LIBRARY_PATH=$CUDNN_PATH/lib:$LD_LIBRARY_PATH
# reduce TensorFlow logging level
 export TF_CPP_MIN_LOG_LEVEL=3
# display some things here and there
./nvcc.sh
./verify-tensor-flow.sh
# time
TZ='America/New_York'; export TZ

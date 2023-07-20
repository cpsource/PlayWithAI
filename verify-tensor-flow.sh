#!/bin/bash
# Verify TensorFlow install
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('CPU'))"

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers

from matplotlib import pyplot as plt

print("Imported the modules.")

# Load the dataset
# all shooting
#dataset = pd.read_csv("export-6cbe7078-646b-4df3-b655-0f2e747db055.csv")
# mass shootings
dataset = pd.read_csv("export-f9bcaa58-5bc4-4994-a0bb-e66fbd5e4225.csv")

print("Loaded the dataset")

# from worldpopulationreview.com/states

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers

from matplotlib import pyplot as plt

print("Imported the modules.")

# Load the dataset

dataset = pd.read_csv("state-population-table.csv")

print("Loaded the dataset")

# from gunviolencearchive.org

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers

from matplotlib import pyplot as plt

# make an array of zeroes, one for each state and one for D.C.
totals = np.zeros(51)

def get_density(state,vics,dataSet):
    dataSet.reset_index()
    for index, row in dataSet.iterrows():
        if state == row['state']:
            totals[index] += vics
            return
    #print(state)

    #print(state,"Not Found")
    totals[50] += vics

print("Imported the modules.")

# Load the dataset
# all shooting
#dataset = pd.read_csv("export-6cbe7078-646b-4df3-b655-0f2e747db055.csv")
# mass shootings
dataset = pd.read_csv("export-f9bcaa58-5bc4-4994-a0bb-e66fbd5e4225.csv")
new_dataset = dataset[['State','# Victims Killed']]

state_dataset = pd.read_csv("state-population-table.csv")
new_state = state_dataset[['state','densityMi']]
#print(new_state)

print("Loaded the datasets")

new_dataset.reset_index()
#print(new_dataset)
for index, row in new_dataset.iterrows():
    #print(row['State'],row['# Victims Killed'])
    m = row['State']
    n = row['# Victims Killed']
    #print(type(m))
    x = get_density(m,n,new_state)

    #print("Density",get_density(row['State'],new_state))

#state_dataset.reset_index()
#print(new_state)

for index, row in new_state.iterrows():
    m = row['state']
    n = row['densityMi']
    p = totals[index]
    q = p/n
    print(q,m,n,p)

#print(totals)
#print(new_state)

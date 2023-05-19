import numpy as np
import pickle

from sklearn.preprocessing import MinMaxScaler 

minmax = MinMaxScaler()
minmax.fit(np.random.normal(0,5, (2,1000)))
   
with open("example_scaler.pkl", 'wb') as f:
  pickle.dump (minmax, f)


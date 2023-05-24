import pandas as pd
import pickle
import os

df = pd.DataFrame(
    {
        "Name": [
            "Braund, Mr. Owen Harris",
            "Allen, Mr. William Henry",
            "Bonnell, Miss. Elizabeth",
        ],
        "Age": [22, 35, 58],
        "Sex": ["male", "male", "female"],
    }
)

print(df)

#myp = pickle.dumps ( df )

# Store data (serialize)
with open('filename.pickle', 'wb') as handle:
    pickle.dump(df,handle,protocol=pickle.HIGHEST_PROTOCOL)
handle.close()
# Load data (deserialize)
with open('filename.pickle', 'rb') as handle:
    unserialized_data = pickle.load(handle)
handle.close()

os.remove('filename.pickle')

print(unserialized_data)
print('Done')


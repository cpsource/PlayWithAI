
import pandas as pd

pd.show_versions()

exit

data = {
  "name": ["Sally", "Mary", "John"],
  "qualified": [True, False, False]
}

df = pd.DataFrame(data,copy=True)
print(df)

#Make a copy:

newdf = df.copy()

print(newdf)


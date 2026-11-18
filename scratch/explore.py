import pandas as pd
import numpy as np

data = pd.read_csv("CHEMBL203_clean.csv") #loads data
 
print(data.head()) #prints first 5 rows
print(data.shape) #prints rows and cols

sorted_data = data.sort_values(by="pIC50",ascending=False)

print(sorted_data.head(n=10))

sorted_filtered_data = sorted_data[sorted_data["pIC50"] < 12]

print(sorted_filtered_data.shape)
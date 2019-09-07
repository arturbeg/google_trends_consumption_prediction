import pandas as pd
import numpy as np

cci = pd.read_csv("consumer_confidence_index_us.csv", header=0)
# print(cci.shape)

cci = cci.loc[cci['LOCATION'] == 'USA']

cci = cci[['TIME', 'Value']]
cci.columns = ['month', 'cci']
# print(cci.head())

# Analogous procedure was applied to the rest of the variables 
# to then merge them on their 'month' field
# to get a merged dataset of variables

import pandas as pd
import numpy as np

# Script makes sure all of the variables are converted to year-on-year growth rates
df = pd.read_csv('merged_dataset.csv', header=0)

print(df.shape)

df_google_indicators_and_other = df.copy()

del df_google_indicators_and_other['date'] # do not need to modify date
del df_google_indicators_and_other['consumption'] # already comes in yoy growth rates

print(df_google_indicators_and_other.shape)

final_df = (df_google_indicators_and_other - df_google_indicators_and_other.shift(12)) / df_google_indicators_and_other

print(final_df.head(15))

## merge back the date and consumption

final_df['date'] = df['date']
final_df['consumpton'] = df['consumption']

## export the csv file
final_df = final_df.dropna()
final_df.to_csv("merged_dataset_yoy.csv", index=None)

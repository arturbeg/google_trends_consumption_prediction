import pandas as pd
import numpy as np


# Apend one-month lags of GT yoy series to the dataset
df = pd.read_csv("final_yoy_dataset.csv", header=0)
df_google_only = df.copy()
del df_google_only['date']
del df_google_only['consumption']
del df_google_only['smp500']
del df_google_only['cci']
del df_google_only['michigan_index']
del df_google_only['tbill']
del df_google_only['income']

def get_adl_dataset(df_param):
  df_lags = df_param.copy()
  df_lags = df_lags.shift(1)
  df_lags = df_lags.dropna()

  df_lags_columns = df_lags.columns
  df_lags_columns = [x+'_l1' for x in df_lags_columns]
  df_lags.columns = df_lags_columns # updated the colum names for the lagged variables

  # remove first row from df_param
  df_contemp = df_param.copy()
  df_contemp = df_contemp.drop([0], axis=0) # dropping the first row

  # result = pd.concat([df1, df4], axis=1, join='inner')
  result = pd.concat([df_contemp, df_lags], axis=1, join='inner')
  return result


adl_google_df = get_adl_dataset(df_google_only)
# print(adl_google_df.columns)

# gather data into csv

adl_google_plus_other_vars = adl_google_df.copy()

df_minus_one_row = df.copy()
df_minus_one_row = df_minus_one_row.drop([0], axis=0)


adl_google_plus_other_vars['date'] = df_minus_one_row['date']
adl_google_plus_other_vars['consumption'] = df_minus_one_row['consumption']
adl_google_plus_other_vars['smp500'] = df_minus_one_row['smp500']
adl_google_plus_other_vars['cci'] = df_minus_one_row['cci']
adl_google_plus_other_vars['michigan_index'] = df_minus_one_row['michigan_index']
adl_google_plus_other_vars['tbill'] = df_minus_one_row['tbill']
adl_google_plus_other_vars['income'] = df_minus_one_row['income']

adl_google_plus_other_vars.to_csv('adl_dataset.csv', index=False)


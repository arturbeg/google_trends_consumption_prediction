import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA

# Apply the KPCA transformation to the ADL dataset = GT variables and their one-month lags
df = pd.read_csv("adl_dataset.csv", header=0)

df_google_only = df.copy()
del df_google_only['date']
del df_google_only['consumption']
del df_google_only['smp500']
del df_google_only['cci']
del df_google_only['michigan_index']
del df_google_only['tbill']
del df_google_only['income']

# Standard Scale the data
df_google_only_st = StandardScaler().fit_transform(df_google_only)

transformer = KernelPCA(kernel='rbf', gamma=10)
transformer = transformer.fit(df_google_only_st)

train_img = transformer.transform(df_google_only_st)

train_img = pd.DataFrame(data = train_img)

# gather data into a csv
df_after_pca = train_img.copy()
df_after_pca['date'] = df['date']
df_after_pca['consumption'] = df['consumption']
df_after_pca['smp500'] = df['smp500']
df_after_pca['cci'] = df['cci']
df_after_pca['michigan_index'] = df['michigan_index']
df_after_pca['tbill'] = df['tbill']
df_after_pca['income'] = df['income']

df_after_pca.to_csv('adl_after_kernel_pca.csv', index=False)



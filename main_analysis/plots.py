import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# An example script to produce plots of the variables used throughtout the main analysis
plt.close('all')

kpca_df = pd.read_csv("adl_after_kernel_pca.csv", header=0)
print(kpca_df.head())

dates = kpca_df['date']
michigan_idnex = kpca_df['michigan_index']
cci = kpca_df['cci']
consumption = kpca_df['consumption']
smp500 = kpca_df['smp500']

fig, ax1 = plt.subplots()

ax2 = ax1.twinx()
ax1.plot(dates, consumption, 'red')
ax2.plot(dates, michigan_idnex, 'blue')

ax1.set_xlabel('month')

tick_spacing = 30
ax1.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

ax1.set_ylabel('Consumption yoy growth rate', color='red')

ax2.set_ylabel('MCSI', color='blue')

ax1.legend(['Consumption yoy growth rate']) 

ax2.legend(['MCSI yoy growth rate']) 
plt.title('MCSI and Consumption yoy growth rates')
plt.show()




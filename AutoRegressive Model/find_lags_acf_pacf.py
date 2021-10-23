import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import statsmodels.graphics.tsaplots as sgt 
import statsmodels.tsa.stattools as sts 
from statsmodels.tsa.seasonal import seasonal_decompose
import seaborn as sns 
sns.set()

raw_csv_data = pd.read_csv('data/Index2018.csv')
df_comp = raw_csv_data.copy()
df_comp.date = pd.to_datetime(df_comp.date, dayfirst=True)
df_comp.set_index('date', inplace = True)
df_comp = df_comp.asfreq('b')
df_comp = df_comp.fillna(method = 'ffill')

# Removing surplus data
df_comp['market_value'] = df_comp.spx 
del df_comp['spx']
del df_comp['dax']
del df_comp['ftse']
del df_comp['nikkei']

size = int(len(df_comp)*0.8)
df, df_test = df_comp.iloc[:size], df_comp[size:]


# ACF
ax = sgt.plot_acf(df.market_value, zero=False, lags=40)
ax.figure.savefig('plots/ACF.png')

# PACF
ax = sgt.plot_pacf(df.market_value, zero=False, lags=40, method = ('olis'))
ax.figure.savefig('plots/PACF.png')


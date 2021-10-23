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

# white Noise
wn = np.random.normal(loc = df.market_value.mean(), scale = df.market_value.std(), size = len(df))
print(wn)

df['wn'] = wn
print(df.head())

ax = df.wn.plot(figsize= (10, 10), title ='WhiteNoise plot')
ax.figure.savefig('plots/white_noise.png')

ax_m = df.market_value.plot(figsize= (10, 10), title ='MarketValue plot')
ax.figure.savefig('plots/marketValue.png')



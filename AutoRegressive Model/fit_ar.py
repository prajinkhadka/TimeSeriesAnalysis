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
ax = sgt.plot_pacf(df.market_value, zero=False, lags=40, method = ('ols'))
ax.figure.savefig('plots/PACF.png')

# fit AR model
from statsmodels.tsa.arima.model import ARIMA
# # 1 -> Number of pase values we wist to incorporate in the model.
# 0 -> Not taking any of the residual values into consideration.ARMA

model_ar = ARIMA(df.market_value, order= (1,0,0))
result_ar = model_ar.fit()
print(result_ar.summary())

model_ar_2 = ARIMA(df.market_value, order= (2,0,0))
result_ar_2 = model_ar_2.fit()
print(result_ar_2.summary())

model_ar_3 = ARIMA(df.market_value, order= (3,0,0))
result_ar_3 = model_ar_3.fit()
print(result_ar_3.summary())

model_ar_4 = ARIMA(df.market_value, order= (4,0,0))
result_ar_4 = model_ar_4.fit()
print(result_ar_4.summary())

# Log Likelihood ratio test
from scipy.stats.distributions import chi2

def LLR_test(mod_1, mod_2, DF=1):
    L1 = mod_1.fit().llf
    L2 = mod_2.fit().llf
    LR = (2*(L2-L1))
    return chi2.sf(LR, DF).round(3) 

print(LLR_test(model_ar, model_ar_2))
print(LLR_test(model_ar_2, model_ar_3))
print(LLR_test(model_ar_3, model_ar_4))
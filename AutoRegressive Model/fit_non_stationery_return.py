from time import process_time
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


print(sts.adfuller(df.market_value))

# -1.7369847452352425, 
# 0.4121645696770626, -> p_value
# 18, 
# 5002, 
#   {
#   '1%': -3.431658008603046, 
#   '5%': -2.862117998412982, 
#   '10%': -2.567077669247375}, 

#   39904.880607487445)

# Since p_value is greater than 5% significce, 
#   we cant say data comes from stationery process.
#       So, AR model is not accirate for this process.

# Transforming the datset so, that it firs the stationery process

# Using Return

df['returns'] = df.market_value.pct_change(1).mul(100)
df = df.iloc[1:]
print(sts.adfuller(df.returns))

# (-17.03445719098114, 
# 8.28053702031742e-30, 
# 17, 5002, 
# {'1%': -3.431658008603046, 
# '5%':  -2.862117998412982, 
# '10%': -2.567077669247375}, 
#   16035.926219345132)

# The test statistic -17 falls way left of 1% signiniface, so we can say data comes from stationery process


# ACF
ax = sgt.plot_acf(df.returns, zero=False, lags=40)
ax.figure.savefig('plots/ACF_returns.png')

# PACF
ax = sgt.plot_pacf(df.returns, zero=False, lags=40, method = ('ols'))
ax.figure.savefig('plots/PACF_returns.png')

from statsmodels.tsa.arima.model import ARIMA
model_ar_ret = ARIMA(df.returns, order= (1,0,0))
result_ar_net = model_ar_ret.fit()
print(result_ar_net.summary())

model_ar_ret_2 = ARIMA(df.returns, order= (2,0,0))
result_ar_net_2 = model_ar_ret_2.fit()
print(result_ar_net_2.summary())

model_ar_ret_3 = ARIMA(df.returns, order= (3,0,0))
result_ar_net_3 = model_ar_ret_3.fit()
print(result_ar_net_3.summary())

from scipy.stats.distributions import chi2

def LLR_test(mod_1, mod_2, DF=1):
    L1 = mod_1.fit().llf
    L2 = mod_2.fit().llf
    LR = (2*(L2-L1))
    return chi2.sf(LR, DF).round(3) 

print(LLR_test(model_ar_ret, model_ar_ret_2))

print(LLR_test(model_ar_ret_2, model_ar_ret_3))

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

# print(sts.adfuller(df.market_value))

# (         -1.7369847452352425, -> test statistic   
#           0.4121645696770626, -> p_value ( Probabilty of data not being stationery.)
#           18, -> No of lags
#           5002, -> No of observations.
#        {
#           '1%': -3.431658008603046, -> 1% Confidence Value
#           '5%': -2.862117998412982, -> 5% Confidence value
#           '10%': -2.567077669247375 -> 10% Confidence Value  
# 
# }, 
# 39904.880607487445 -> Maximum Importamtion Criteria
# )

print(sts.adfuller(df.wn))
#  This is stationdery data so p-value is 0

# (-71.8252370370871, 
# 0.0, 0, 5020, 
#   {'1%': -3.431653316130827, 
#   '5%': -2.8621159253018247, 
#      '10%': -2.5670765656497516}, 
#  70709.91981230871)

 
s_dec_additive = seasonal_decompose(df.market_value, model = 'additive')
ax = s_dec_additive.plot()
ax.figure.savefig('plots/AdditiveSeasonalityDecompose.png')

s_dec_mul = seasonal_decompose(df.market_value, model = 'multiplicative')
ax1 = s_dec_additive.plot()
ax1.figure.savefig('plots/MulSeasonalityDecompose.png')

# AutoCorelation

# ACF Plot
ax_sgt = sgt.plot_acf(df.market_value, lags = 40, zero=False)
ax_sgt.title('ACF S&P')
ax_sgt.figure.savefig('plots/ACF_S&P.png')

# PACF Plot

ax_pacf = sgt.plot_pacf(df.market_value, lags = 40, zero=False, method = ('ols'))
ax_pacf.title('PACF S&P')
ax_sgt.figure.savefig('plots/PACF_S&P.png')



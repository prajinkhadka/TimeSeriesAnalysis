import pandas as pd 
import numpy as np 

raw_csv_data = pd.read_csv('data/Index2018.csv')
df_comp = raw_csv_data.copy()

# length of the timeperiod
print(df_comp.date.describe())

# Change to datetimne object
df_comp.date = pd.to_datetime(df_comp.date, dayfirst=True)

print(df_comp.date.describe())

# Setting Date Column as Index
df_comp.set_index('date', inplace = True)
print(df_comp.head())

# Setting the Frequency
df_comp = df_comp.asfreq('b')
print(df_comp.head())

# Filling Missing Values
print(df_comp.isna().sum())

# ffill() - Fills with previous data value
# bfill - Fills with next period
# fillingwith same value 

df_comp.spx = df_comp.spx.fillna(method = 'ffill')
df_comp.ftse = df_comp.ftse.fillna(method = 'bfill')
df_comp.dax = df_comp.dax.fillna(value = df_comp.dax.mean())
df_comp.nikkei = df_comp.nikkei.fillna(value = df_comp.nikkei.mean())

print(df_comp.isna().sum())

# Adding and deleting columns
df_comp['market_value'] = df_comp.spx 
del df_comp['spx'], df_comp['ftse'], df_comp['nikkei']

# Spligint the Data
size = int(0.8*(len(df_comp)))
df = df_comp.iloc[:size]
df_test = df_comp[size:]

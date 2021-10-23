import pandas as pd 
import numpy as np 

raw_csv_data = pd.read_csv('data/Index2018.csv')
df_comp = raw_csv_data.copy()

print(df_comp.head())

print(df_comp.describe())

print(df_comp.isna())

print(df_comp.isna().sum())


# plotting the Data
import matplotlib.pyplot as plt 
ax = df_comp.spx.plot(figsize= (10, 10), title ='SpX plot')
ax.figure.savefig('plots/SPX_plot.png')
ax1 = df_comp.ftse.plot(figsize= (10, 10), title ='FTSE plot')
ax.figure.savefig('plots/FTSE_plot.png')


# QQ Plot
    # Is used to determine if the data is distributed in a certain way.
    # Usually to showcase how the data fits a Normal Distribution

import scipy.stats
import pylab

scipy.stats.probplot(df_comp.spx, plot = pylab)


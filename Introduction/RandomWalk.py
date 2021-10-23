import pandas as pd
rw = pd.read_csv('data/RandWalk.csv')
rw.date = pd.to_datetime(rw.date, dayfirst=True)
rw.set_index('date', inplace = True)
rw = rw.asfreq('b')

rw['rw'] = rw.price

import matplotlib.pyplot as plt
ax = rw.rw.plot(figsize=(10, 10))
plt.title('RandomWalk')
ax.figure.savefig('plots/RandomWalk.png')

# If there existis randomWalk in the data, 
#   then the price cannot be predicted Properly.

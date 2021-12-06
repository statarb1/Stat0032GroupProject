ia# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 20:55:01 2021

@author: Pranav
"""
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 13:21:37 2021

@author: Pranav
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scipy as sp
import pylab 

data_raw = pd.read_csv('forestfires_data.csv')
data_raw.shape
data_raw_summary = data_raw.describe()

area_threshold = 0.01; # hectares

data = data_raw.loc[data_raw.area > area_threshold]
data_summary = data.describe()
data.shape
plt.hist(data.area, 50)
plt.hist(np.log(data.area + 1), 50)
plt.hist(np.log(data.area), 50)

plt.hist(np.log(data.area[data.month == 'aug']), 20)
plt.hist(np.log(data.area[data.month == 'sep']), 20)

plt.hist(data.month)

correl = data.corr()
sns.heatmap(correl, vmin=-1, vmax=1, cmap='BrBG')

#This last line doesnt work at the moment but might be quite useful to summarise the data
a = data.groupby('month').describe().unstack(1).reset_index().pivot(index='month', values=0, columns='level_1')

# Test for log normality - potential tests Shapiro-Wilk Test, Jarque-Bera Test, 
# D'Agostino-Pearson Test, Kolmogorov-Smirnov Goodness of Fit Test
p = sp.stats.mstats.normaltest(np.log(data.area), axis = 0).pvalue
sp.stats.probplot(np.log(data.area), dist = 'norm', plot = pylab)

# Shapiro - Wilk test
stat, p = sp.stats.shapiro(np.log(data.area))
# fail to reject log normality - that's not really that useful

# D’Agostino’s K-squared test
sp.stats.normaltest(np.log(data.area))

# Anderson-Darling Normality Test
sp.stats.anderson(np.log(data.area))

# Chi Sq test
sp.stats.chisquare(np.log(data.area))

# Kolmogorov Smirnov
sp.stats.kstest(np.log(data.area),'norm')

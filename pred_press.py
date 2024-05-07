# -*- coding: utf-8 -*-
"""
Created on Mon May  6 15:43:21 2024

# Initial Reservoir Pressure Prediction at shallower sequence

@author: akmalaulia
"""

import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# read input data
df = pd.read_csv("input_to_ml.csv")

# defined X_train, y_train
df_train = df.loc[df['Status']== "Train"]
X_train = df_train[['DEPTH',
                    'SWE_BASE',
                    'SW_J_BASE',
                    'VSH_BASE_1',
                    'PHIE_BASE_1',
                    'PERM_BASE',
                    'PHIT_BASE',
                    'GR',
                    'RESD',
                    'RESM',
                    'RESS',
                    'NPHI',
                    'RHOB',
                    'TVDSS']]
y_train = df_train[['Initial Pressure']]

# define X_test
df_test = df.loc[df['Status']== "Test"]
X_test = df_test[['DEPTH',
                    'SWE_BASE',
                    'SW_J_BASE',
                    'VSH_BASE_1',
                    'PHIE_BASE_1',
                    'PERM_BASE',
                    'PHIT_BASE',
                    'GR',
                    'RESD',
                    'RESM',
                    'RESS',
                    'NPHI',
                    'RHOB',
                    'TVDSS']]

# scaling
# from sklearn.preprocessing import StandardScaler  
scaler = StandardScaler()  
# Don't cheat - fit only on training data
scaler.fit(X_train)  
X_train = scaler.transform(X_train)  
# apply same transformation to test data
X_test = scaler.transform(X_test)  

# train neural net
# Reference: https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html
regr = MLPRegressor(random_state=1, max_iter=500).fit(X_train, y_train)

# predict initial pressure at TVDSS specified by the Test set (ie. df_test)
df_test['Initial Pressure'] = regr.predict(X_test)


# plotting results
p_train = df_train[['TVDSS', 'Initial Pressure', 'Status']] 
p_test = df_test[['TVDSS', 'Initial Pressure', 'Status']] 

p_train = p_train.sort_values(by=['TVDSS'], ascending=False)
p_test = p_test.sort_values(by=['TVDSS'], ascending=False)

p_all = p_train.append(p_test)
p_all = p_all.sort_values(by=['TVDSS'], ascending=False)
p_all.to_csv('predicted_pressure.csv', index=False)

plt.plot(p_train['Initial Pressure'], p_train['TVDSS'], 'b')
plt.plot(p_test['Initial Pressure'], p_test['TVDSS'], 'r')
plt.show()

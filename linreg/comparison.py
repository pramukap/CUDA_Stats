
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, Ridge

data = pd.read_csv('data/house_data.csv')
data = data.iloc[:1000,:]

print('--------- For Simple Linear Regression ----------\n')

target = data['price']
input = data[[col for col in data.columns if col != 'price']]
print('Input data\n',input.head())
print('Target data\n',target.head())
print('Betas: (Weights):')
lr = LinearRegression(fit_intercept=True)
lr.fit(input,target)
print(lr.intercept_, lr.coef_)
print('\n\n')


print('--------- For Ridge Regression (Regularized)----------\n')

rr = Ridge(alpha=0, solver='cholesky')
rr.fit(input, target)
print(rr.intercept_, rr.coef_)


# In[32]:
print('--------- For Ridge Regression (Regularize Intercept)----------\n')


print("Beta's from explicit ridge regression")
input['intercept'] = np.ones(input.shape[0])
print(np.matmul(np.linalg.inv(np.matmul(np.transpose(input), input) + 0.0655*np.identity(9)), np.matmul(np.transpose(input), target)))


# In[34]:
print('--------- For Ridge Regression (Regularize Intercept)----------\n')


rr = Ridge(alpha=0.0655, solver='cholesky',fit_intercept=False)
rr.fit(input, target)
print(rr.coef_)


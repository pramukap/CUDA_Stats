import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression


data = pd.read_csv('house_data.csv')
print(data.columns)
X = data[[col for col in data.columns if col not in ['price','id','date']]]
y = data['price']
#X = data[col for col in data.columns not in ['id','price','date']]

X = X.iloc[:1000, :]
y = y.iloc[:1000]

print(y)

lr = LinearRegression()
lr.fit(X, y)

print(lr.coef_)
print(lr.intercept_)

newX = X.copy()
print(newX)
newX['price'] = pd.Series(y)
newX['pred'] = lr.predict(X)
print(lr.predict(X).shape)
print(newX)

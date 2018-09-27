import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

# pandas options
pd.set_option('display.expand_frame_repr', False)

df = pd.read_csv("https://s3.amazonaws.com/codecademy-content/programs/data-science-path/linear_regression/honeyproduction.csv")

prod_per_year = df.groupby('year').totalprod.mean().reset_index()

# X, y values
X = prod_per_year['year']
X = X.values.reshape(-1,1)
y = prod_per_year['totalprod']


# model and fitting
regr = linear_model.LinearRegression()
regr.fit(X, y)

# print('Slope: ', regr.coef_)
# print('Intercept: ', regr.intercept_)

y_predict = regr.predict(X)

# predictions
X_future = np.array(range(2013, 2051))
X_future = X_future.reshape(-1,1)

future_predict = regr.predict(X_future)

# plots - what we know
plt.scatter(X,y, alpha=0.5)
# plots - Y prediction LR
plt.plot(X, y_predict,'-', color='orange')
# plots - the grim future
plt.plot(X_future, future_predict, '--', color='red')

plt.show()



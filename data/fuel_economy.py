import pandas as pd
import sklearn as sk
import sklearn.svm #necessary 
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("D:/data/fuel_economy.csv", sep = ',', index_col = False)

#define regression models
lin_reg = sk.linear_model.LinearRegression()
poly_reg = sk.preprocessing.PolynomialFeatures(degree=4)

#define variables
X = df['eng_displ'].values.reshape(-1,1)
Y = df['MPG'].values.reshape(-1,1)

#==============================================================================

#explorative scatter plot
plt.scatter(X, Y)

#explorative histograms
plt.hist(X, bins=15)
plt.hist(Y, bins=15)

#log-transformations of histograms
plt.hist(np.log(X), bins=15)
plt.hist(np.log(Y), bins=15)

#==============================================================================

#fit linear regression model
lin_reg.fit(X,Y)

#scatter plot with linear regression line
plt.plot(X, lin_reg.predict(X), label='Linear Regression', color='b')
plt.scatter(X, Y, label='Actual Test Data', color='g', alpha=.7)
plt.xlabel('Engine Displacement')
plt.ylabel('Fuel Economy')

#plot observed vs predicted
plt.scatter(Y, lin_reg.predict(X), label='Actual Test Data', color='g', alpha=.7)
plt.xlim([0,60])
plt.ylim([0,60])
plt.xlabel('Observed')
plt.ylabel('Predicted')

#==============================================================================

#fit polynomial regression model
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly, Y)
lin_reg.fit(X_poly, Y)

# arrange X to be plotted
X_plot = np.sort(X, axis=0)

#scatter plot with polynomial regression line
plt.scatter(X, Y, label='Actual Test Data', color='g', alpha=.7)
plt.plot(X_plot, lin_reg.predict(poly_reg.fit_transform(X_plot)), color = 'blue')
plt.xlabel('Engine Displacement')
plt.ylabel('Fuel Economy')

del X_plot, X_poly
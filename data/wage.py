import pandas as pd
import sklearn as sk
import sklearn.svm #necessary 

df = pd.read_csv("D:/data/wage.csv", sep = ',', index_col = False)

#define regression models
lin_reg = sk.linear_model.LinearRegression()
poly_reg = sk.preprocessing.PolynomialFeatures(degree=4)

#define variables
X = df['Female'].values.reshape(-1,1)
Y = df['LogWage'].values.reshape(-1,1)

#==============================================================================

#fit linear regression model and calculate residuals
lin_reg.fit(X,Y)
lin_reg.intercept_
lin_reg.coef_

residuals = (Y - lin_reg.predict(X))

#==============================================================================

#regress residuals on a constant and education (+22% on the unexplained part of wage)
Y = residuals
X = df['Educ'].values.reshape(-1,1)

lin_reg.fit(X,Y)
lin_reg.intercept_
lin_reg.coef_

#regress residuals on a constant and part-time job
Y = residuals
X = df['Parttime'].values.reshape(-1,1) 

lin_reg.fit(X,Y)
lin_reg.intercept_
lin_reg.coef_

#==============================================================================

#fit multiple linear regression model
X = df[['Female', 'Age', 'Educ', 'Parttime']]
Y = df['LogWage']

lin_reg.fit(X,Y)
lin_reg.intercept_
lin_reg.coef_

#==============================================================================

#fit multiple linear regression model with Education transformed into a dummy
X = df[['Female', 'Age', 'Educ', 'Parttime']]
Y = df['LogWage']

X.loc[X['Educ']==1, 'Educ'] = 0
X.loc[(X['Educ']==2) | (X['Educ']==3) | (X['Educ']==4), 'Educ'] = 1

lin_reg.fit(X,Y)
lin_reg.intercept_
lin_reg.coef_
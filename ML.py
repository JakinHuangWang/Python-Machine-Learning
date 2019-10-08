import matplotlib.pyplot as plt
import numpy as np
import scipy
import pandas as pd
import pylab
from sklearn import linear_model
from sklearn.metrics import r2_score

#1st dataset
#dataset = pd.read_csv("FuelConsumptionCo2.csv")

#mask = np.random.rand(len(dataset)) < 0.95
#train = dataset[mask]
#test = dataset[~mask]

#2nd dataset
dataset = pd.read_csv("china_gdp.csv")
#What the data looks like
plt.figure(figsize=(8, 5))
x, y = dataset['Year'].values, dataset['Value'].values
print(x, y)
print(type(x))
print(type(dataset[['Year']].unstack()))
print(dataset[['Year']], dataset[['Value']])
plt.plot(x, y, 'bo')
plt.show()

X = np.arange(-5.0, 5.0, 0.1)
#A logistic model that would match our dataset : 1/(1 + e^(-beta1*(X-beta2)))
Y = 1.0 / (1.0 + np.exp(-X))
plt.plot(X,Y) 
plt.ylabel('Dependent Variable')
plt.xlabel('Indepdendent Variable')
plt.show()

#Build our regression model
def sigmoid(x_data, beta1, beta2):
    return 1 / (1 + np.exp(-beta1 * (x_data - beta2)))

beta_1 = 0.10
beta_2 = 1990.0

#logistic function
Y_pred = sigmoid(x, beta_1 , beta_2)

#plot initial prediction against datapoints
plt.plot(x, Y_pred*15000000000000.)
plt.plot(x, y, 'ro')
plt.show()


#Now we want to optimize our line by using curve fit
from scipy.optimize import curve_fit
xdata = x/max(x)
ydata = y/max(y)
popt, pcov = curve_fit(sigmoid, xdata, ydata)
print(" beta_1 = %f, beta_2 = %f" % (popt[0], popt[1]))
x = np.linspace(1960, 2015, 55)
x = x/max(x)
plt.figure(figsize=(8,5))
y = sigmoid(x, *popt)
plt.plot(xdata, ydata, 'ro', label='data')
plt.plot(x,y, linewidth=3.0, label='fit')
plt.legend(loc='best')
plt.ylabel('GDP')
plt.xlabel('Year')
plt.show()


#Simple Linear Regression
#regr = linear_model.LinearRegression()
#train_x = np.asanyarray(train[['ENGINESIZE']])
#train_y = np.asanyarray(train[['CO2EMISSIONS']])
#test_x = np.asanyarray(test[['ENGINESIZE']])
#test_y = np.asanyarray(test[['CO2EMISSIONS']])
#regr.fit(train_x, train_y)
#predict_y = regr.predict(test_x)
#plt.scatter(train_x, train_y, color = 'blue')
#plt.scatter(test_x, predict_y, color = 'green')
#plt.scatter(test_x, test_y, color = "cyan")
#plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_, '-r')
#plt.show()

#Multiple Linear Regression
#multi_regr = linear_model.LinearRegression()
#first = np.asanyarray(train[['ENGINESIZE']])
#second = np.asanyarray(train[['CYLINDERS']])
#third = np.asanyarray(train[['FUELCONSUMPTION_COMB']])
#train_x = np.asanyarray(train[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
#train_y = np.asanyarray(train[['CO2EMISSIONS']])
#test_x = np.asanyarray(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
#test_y = np.asanyarray(test[['CO2EMISSIONS']])
#multi_regr.fit(train_x, train_y)
#print("Coefficient: ", multi_regr.coef_)
#print("Intercept: ", multi_regr.intercept_)
#multi_predict = multi_regr.predict(test_x)
#print(multi_predict.shape, test_x.shape)
#plt.plot(first, multi_regr.coef_[0][0] * first + multi_regr.intercept_, '-r')
#plt.plot(second, multi_regr.coef_[0][1] * second + multi_regr.intercept_, '-g')
#plt.plot(third, multi_regr.coef_[0][2] * third + multi_regr.intercept_, '-b')
#print("Residual sum of squares: %.2f"
      #% np.mean((multi_predict - test_y) ** 2))
#print('Variance score: %.2f' % r2_score(test_y, multi_predict))
#plt.show()


#For Polynomial Regression
#from sklearn.preprocessing import PolynomialFeatures

#train_x = np.asanyarray(train[['FUELCONSUMPTION_COMB']])
#train_y = np.asanyarray(train[['CO2EMISSIONS']])
#test_x = np.asanyarray(test[['FUELCONSUMPTION_COMB']])
#test_y = np.asanyarray(test[['CO2EMISSIONS']])
#poly = PolynomialFeatures(degree=2)

#train_x_poly = poly.fit_transform(train_x)
#poly_regr = linear_model.LinearRegression()
#poly_fit = poly_regr.fit(train_x_poly, train_y)
#print("Coefficient:", poly_regr.coef_)
#print("Intercept:", poly_regr.intercept_)
#X = np.arange(np.min(train_x),np.max(train_x), 0.1)
#plt.scatter(train_x, train_y, color = 'cyan')
#plt.plot(X, poly_fit.intercept_[0] + poly_fit.coef_[0][1] * X + poly_fit.coef_[0][2] * np.power(X, 2), '-r')
#plt.show()

#test_x_poly = poly.fit_transform(test_x)
#y_hat = poly_regr.predict(test_x_poly)
#plt.scatter(test_x, y_hat, color = 'green')
#plt.show()

#print("Mean absolute error: %.2f" % np.mean(np.absolute(y_hat - test_y)))
#print("Residual sum of squares (MSE): %.2f" % np.mean((y_hat - test_y) ** 2))
#print("R2-score: %.2f" % r2_score(y_hat , test_y) )


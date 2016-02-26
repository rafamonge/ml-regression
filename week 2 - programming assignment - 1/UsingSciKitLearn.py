'''
Created on Jan 29, 2016

@author: rmongemo
'''

import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
from pandas import DataFrame, Series
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error 


dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float, 'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float, 'sqft_living':float, 'floors':str, 'condition':int, 'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int}
train = pd.read_csv("kc_house_train_data.csv", dtype= dtype_dict )
test = pd.read_csv("kc_house_test_data.csv", dtype= dtype_dict)

train['bedrooms_squared']  = train['bedrooms'] *train['bedrooms']
train['bed_bath_rooms']  = train['bedrooms'] *train['bathrooms']
train['log_sqft_living']  = np.log(train['sqft_living'])
train['lat_plus_long']  = train['lat'] +train['long']


test['bedrooms_squared']  = test['bedrooms'] *test['bedrooms']
test['bed_bath_rooms']  = test['bedrooms'] *test['bathrooms']
test['log_sqft_living']  = np.log(test['sqft_living'])
test['lat_plus_long']  = test['lat'] +test['long']

print ('question 4')
print(round(test['bedrooms_squared'].mean(),2))
print(round(test['bed_bath_rooms'].mean(),2))
print(round(test['log_sqft_living'].mean(),2))
print(round(test['lat_plus_long'].mean(),2))



# create X and y
feature_cols_1 = ['sqft_living', 'bedrooms', 'bathrooms', 'lat', 'long']
feature_cols_2 = ['sqft_living', 'bedrooms', 'bathrooms', 'lat', 'long','bed_bath_rooms']
feature_cols_3 = ['sqft_living', 'bedrooms', 'bathrooms', 'lat', 'long', 'bed_bath_rooms', 'bedrooms_squared', 'log_sqft_living', 'lat_plus_long']


X1 = train[feature_cols_1]
X2 = train[feature_cols_2]
X3 = train[feature_cols_3]


X1Test = test[feature_cols_1]
X2Test = test[feature_cols_2]
X3Test = test[feature_cols_3]

y = train.price
yTest = test.price

# follow the usual sklearn pattern: import, instantiate, fit

lm1 = LinearRegression()
lm1.fit(X1, y)

# print intercept and coefficients
print('model 1')
print (lm1.intercept_)
coef1 =dict(zip(feature_cols_1,lm1.coef_)) 
print (coef1)


lm2 = LinearRegression()
lm2.fit(X2, y)

# print intercept and coefficients
print('model 2')
print (lm2.intercept_)
coef2 =dict(zip(feature_cols_2,lm2.coef_)) 
print (coef2)


lm3 = LinearRegression()
lm3.fit(X3, y)


# print intercept and coefficients
print('model 3')
print (lm3.intercept_)
coef3= dict(zip(feature_cols_3,lm3.coef_))
print (coef3)


print('question 6')
print(coef1['bathrooms'])

print('question 7')
print(coef2['bathrooms'])


print('question 10')
print(lm1.score(X1, y))
print(lm2.score(X2, y))
print(lm3.score(X3, y))

print('question 12')
print(lm1.score(X1Test, yTest))
print(lm2.score(X2Test, yTest))
print(lm3.score(X3Test, yTest))


print('question 10   -2 ')

print(mean_squared_error(y,lm1.predict(X1)))
print(mean_squared_error(y,lm2.predict(X2)))
print(mean_squared_error(y,lm3.predict(X3)))
print('model 3 is the best')

print('question 12   -2')
print(mean_squared_error(yTest,lm1.predict(X1Test)))
print(mean_squared_error(yTest,lm2.predict(X2Test)))
print(mean_squared_error(yTest,lm3.predict(X3Test)))
print('model 2 is the best')

print(' in R2 bigger is better. in RMSE/RSS lower is better.')

print('end')

if __name__ == '__main__':
    pass



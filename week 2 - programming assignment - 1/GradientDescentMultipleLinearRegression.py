'''
Created on Feb 5, 2016

@author: rmongemo
'''

import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import math
from pandas import DataFrame, Series
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error 
from builtins import round

dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float, 'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float, 'sqft_living':float, 'floors':str, 'condition':int, 'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int}
train_data = pd.read_csv("kc_house_train_data.csv", dtype= dtype_dict )
test_data = pd.read_csv("kc_house_test_data.csv", dtype= dtype_dict)

miniDataSet = pd.read_csv("MiniDataSet.csv" )
print(miniDataSet)


def get_numpy_data(data_sframe, features, output):
    data_sframe['constant'] = 1
    features = ['constant'] + features
    relevanteFeatures = data_sframe[features].values
    outputArray = np.array(data_sframe[output])
    return(relevanteFeatures, outputArray)

def predict_outcome(feature_matrix, weights):
    return(np.dot(feature_matrix, weights))

def feature_derivative(errors, feature):
    return(2*np.dot(feature, errors))

def get_residual_sum_of_squares(predictions, output):
    residuals = output - predictions
    residualsSquared = residuals * residuals 
    return(residualsSquared.sum()) 


def regression_gradient_descent(feature_matrix, output, initial_weights, step_size, tolerance):
    converged = False
    weights = np.array(initial_weights)
    counter = 0
    while not converged:
        # compute the predictions based on feature_matrix and weights:
        predictions = predict_outcome(feature_matrix, weights)
        # compute the errors as predictions - output:
        error =  predictions - output
        gradient_sum_squares = 0 # initialize the gradient
        # while not converged, update each weight individually:
        for i in range(len(weights)):
            # Recall that feature_matrix[:, i] is the feature column associated with weights[i]
            # compute the derivative for weight[i]:
            derivative = feature_derivative(error,feature_matrix[:, i] )
            # add the squared derivative to the gradient magnitude
            gradient_sum_squares = gradient_sum_squares + (derivative * derivative)
            # update the weight based on step size and derivative:
            weights[i] = weights[i]  -  step_size * derivative
        gradient_magnitude = math.sqrt(gradient_sum_squares)
        counter = counter + 1
        if gradient_magnitude < tolerance or counter > 10000:
            converged = True
    return(weights)

step_size = 7e-12
tolerance = 2.5e7      
simple_features = ['sqft_living']
my_output = 'price'
initial_weights = np.array([-47000., 1.])
(simple_feature_matrix, output) = get_numpy_data(train_data, simple_features, my_output)
   
simple_weights = regression_gradient_descent(simple_feature_matrix, output,initial_weights, step_size, tolerance)
print('question 1')
print(round(simple_weights[1],1))

print('question 2 - simple model predicting house 1 price')
(test_simple_feature_matrix, test_output) = get_numpy_data(test_data, simple_features, my_output)
simpleTestOutcomes = predict_outcome(test_simple_feature_matrix, simple_weights)
print(round(simpleTestOutcomes[0]))

print('question 3 - complex model predicting house 1 price' )


model_features = ['sqft_living', 'sqft_living15']
(feature_matrix, output) = get_numpy_data(train_data, model_features,my_output)
initial_weights = np.array([-100000., 1., 1.])
step_size = 4e-12
tolerance = 1e9
model_weights = regression_gradient_descent(feature_matrix, output,initial_weights, step_size, tolerance)

(test_model_feature_matrix, test_output) = get_numpy_data(test_data, model_features, my_output)
modelTestOutcomes = predict_outcome(test_model_feature_matrix, model_weights)
print(round(modelTestOutcomes[0]))

print('actual value of house ')
print(test_data['price'][0])
simpleRssTest = get_residual_sum_of_squares(simpleTestOutcomes, test_output)
modelRssTest=  get_residual_sum_of_squares(modelTestOutcomes, test_output)
print('last question')
print('simple model')
print(simpleRssTest )
print('complex model')
print(modelRssTest )
if __name__ == '__main__':
    pass
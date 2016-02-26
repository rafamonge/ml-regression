'''
Created on Jan 29, 2016

@author: rmongemo
'''

import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
from pandas import DataFrame, Series 

def simple_linear_regression(input_feature, output):
    x = input_feature
    y = output
    meanOfX = x.mean();
    meanOfY = y.mean();
    xy = x * y
    xx = x * x
    meanOfXy = xy.mean()
    meanOfXx = xx.mean()
    
    slopeNumerator = meanOfXy - meanOfX * meanOfY
    slopeDenominator = meanOfXx - meanOfX*meanOfX
    slope = slopeNumerator / slopeDenominator  
    
    intercept = meanOfY - slope * meanOfX
    return(intercept, slope)

def get_regression_predictions(input_feature, intercept, slope):
    x = input_feature
    predicted_output = x*slope + intercept;
    return(predicted_output)


def get_residual_sum_of_squares(input_feature, output, intercept,slope):
    results =  get_regression_predictions(input_feature, intercept, slope)
    residuals = output - results
    residualsSquared = residuals * residuals 
    return(residualsSquared.sum()) 




def inverse_regression_predictions(output, intercept, slope):
    return((output - intercept) / slope)

    '''[your code here]
return(RSS)'''



train = pd.read_csv("kc_house_train_data.csv")
test = pd.read_csv("kc_house_test_data.csv")
miniDataSet = pd.read_csv("miniDataSet.csv")

trainInput = train['sqft_living']
trainInputBedrooms   = train['bedrooms']
trainOutput = train['price']

testInput = test['sqft_living']
testInputBedrooms = test['bedrooms']
testOutput = test['price']

interceptAndSlope  = simple_linear_regression(trainInput, trainOutput)
intercept = interceptAndSlope[0]
slope  = interceptAndSlope[1]

print(interceptAndSlope)

question6 = intercept + slope * 2650
print('question 6')
print(question6)
print('700074.85')



print('question 8')
question8 = get_residual_sum_of_squares(trainInput, trainOutput, intercept, slope)
print(question8)


print('question 10')
question10 = inverse_regression_predictions(800000, intercept, slope)
print(question10)



print('question 11')


interceptAndSlopeBedrooms  = simple_linear_regression(trainInputBedrooms, trainOutput)
interceptBedrooms = interceptAndSlope[0]
slopeBedrooms  = interceptAndSlope[1]

rssTestSquareFeet = get_residual_sum_of_squares(testInput, testOutput, intercept, slope)
rssBedrooms = get_residual_sum_of_squares(testInputBedrooms, testOutput, interceptBedrooms, slopeBedrooms)  

print('square feet')
print(rssTestSquareFeet)
print ('bedrooms')
print(rssBedrooms)
print (rssTestSquareFeet <  rssBedrooms)

if __name__ == '__main__':
    pass


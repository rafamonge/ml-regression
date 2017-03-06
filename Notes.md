# notes 

Model: How we assume the world works
# Simple linear regresssion
yi =  W0 + W1Xi + Ei

W0 and W1 are regression coefficients
W0 is the intercept

Cost can be measure in:
RSS =  Sum ( yi - (w0 +w1xi)) ^ 2 

W0 = value of y when X  = 0
W1 = predicted change in the output per unit of change in the input x

## solving

- Gradient descent minimize the RSS function over all possible W0 and W1.
- Solve when the gradient = 0 . Also know as the closed form . Usually slower than gradient descent and in some cases can't be solved.
- Gradiend descent relies on stepsize and convergence criteria. 

#Error

Can use multiple loss/error functions

- Absolute Error = |y - f(x)|
- Squared  Error = (y - f(x)) ^ 2 

# on Model complexity

- Training error = avg. loss on houses in training  set
- Training error = avg. loss on houses in test set
- As model complexiy increases, training error decreases
- Small training error ≠> good predictions. Training error is overly optimistic because ŵ was fit to training data
- As model complexity increases, test error decreases  and then starts to climb back. Like an U
- Your job is to find the lowest part of the U.

# Overfitting

Formal definition
- there are 2 models, one called S and one called C. S is more simple than C (which is more complex)
- TrainingError(S) > TrainingError(C)
- TestingError(S)  > TestingError(S)
- Model C is overfitted.

# Training vs Test splits
- Too few data points in Training means the model will be poorly fitted.
- Too few data points in Testing means that we'll have a bad estimation of the generalization error (which the test error attempts to approach)
- Typically, you want just enought points in the test set to form a reasonable estimate of the generalization error.

# 3 sources of error 

1.  Noise: irreducible erorr. 
2.  Bias: Over all possible size N training sets, what do I expect my fit to be. Is our model potentiall flexible enough to capture the true prediction. E.g. a constant line won't predict a True Cubic function very well.
3.  Variance: How much do specific fits vary from the expected fit? 

- High Complexity -> High Variance  & Low Bias
- Low Complexity -> Low Variance & High bias
- find the sweetspot

# Adding validation set
- Split data in training, validation and test
- Select λ* such that ŵλ* minimizes error on
validation set
- Approximate generalization error of ŵλ* using
test set
- Training: fit ŵλ
- Validation: test performance of ŵλ to select λ
- Test: Assess generalization error.
- Typical splits: 80/10/10 or 50/25/25


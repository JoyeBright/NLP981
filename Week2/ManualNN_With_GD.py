import numpy as np

# Mini dataset
input_data = np.array([1, 2, 3])
weights = np.array([0, 2, 1])

learning_rate = 0.01
target = 0

predict = (input_data * weights).sum()

# print("predict: %s" % predict)

error = predict - target

# print("error: %s" % error)

# The derivation of loss function (MSE) would be the slope
slope = 2 * error * input_data

# print("Slope: %s" % slope)

# Gradient Descent:
# if slope is positive: Subtraction
# if slope is negative: Add
updatedWeights = weights - (slope * learning_rate)

# print("Updated weights: %s" % updatedWeights)

updatedPredicts = (input_data * updatedWeights).sum()

# print("Updated Predicts: %s" % updatedPredicts)

updatedError = updatedPredicts - target

# print("Update Error: %s" % updatedError)

print("The error without applying gradient descent: %f" % error)
print("the error with applying Gradient Descent/UpdatedError: %f" % updatedError)
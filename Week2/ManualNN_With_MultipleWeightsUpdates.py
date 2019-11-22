import numpy as np
import matplotlib.pyplot as plt

input_data = np.array([1, 2, 3])
target = 0
weights = np.array([0, 2, 1])
updateNum = 20
mse_hist = []


def get_error(inputd, actual_target, weight):
    predicts = (inputd * weight).sum()
    error = predicts - actual_target
    return error


def get_slope(inputd, actual_target, weight):
    error = get_error(inputd, actual_target, weight)
    out = 2 * error * inputd
    return out


def get_mse(inputd, actual_target, updated_weights):
    errors = get_error(inputd, actual_target, updated_weights)
    mse_result = np.mean(errors**2)
    return mse_result


for i in range(updateNum):
    slope = get_slope(input_data, target, weights)
    weights = weights - (slope * 0.01)
    mse = get_mse(input_data, target, weights)
    mse_hist.append(mse)


plt.plot(mse_hist)
plt.xlabel("Iterations")
plt.ylabel("Mean Squared Error")
plt.show()

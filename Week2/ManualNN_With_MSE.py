import numpy as np
from sklearn.metrics import mean_squared_error

# Mini Dataset

input_data = np.array([0, 3, 5, 7, 8, 2])

weights = {'node0': np.array([2, 1]),
           'node1': np.array([1, 2]),
           'output': np.array([1, 1])}

newWeights = {'node0': np.array([0, 1]),
              'node1': np.array([1, 0]),
              'output': np.array([1, 1])}

ActualTargets = [1, 2, 3, 3, 9, 3]


def ReLu(x):
    out = max(0, x)
    return out


def predict_with_network(inputd, weights):

    node0_input = (inputd * weights['node0']).sum()
    node0_output = ReLu(node0_input)

    node1_input = (inputd * weights['node1']).sum()
    node1_output = ReLu(node1_input)

    hidden_layer_values = np.array([node0_output, node1_output])

    output = (hidden_layer_values * weights['output']).sum()
    return output


# Create model_output_0
model_output_0 = []

# Create model_output_1
model_output_1 = []

for row in input_data:
    model_output_0.append(predict_with_network(row, weights))
    model_output_1.append((predict_with_network(row, newWeights)))

# Calculate the Mean Squared Error for model_output_0: mse0
mse0 = mean_squared_error(model_output_0, ActualTargets)

# Calculate the Mean Squared Error for model_output_1: mse1
mse1 = mean_squared_error(model_output_1, ActualTargets)


print("The predicted output for first model: %s" % model_output_0)
print("Mean Squared Error with first series of weights: %f" % mse0)
print("The predicted output for second model: %s" % model_output_1)
print("Mean Squared Error with new weights: %f" % mse1)

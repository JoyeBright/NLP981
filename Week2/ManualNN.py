import numpy as np

# Mini Dataset
input_data = np.array([5, 7, 8, 1])

weights = {'node0': np.array([1, 1]),
           'node1': np.array([-1, 1]),
           'output': np.array([2, -1])}


# Activation Function
def ReLu(x):
    out = max(0, x)
    return out


def predict_with_NN(input_data_row, weights):
    print("\nFirst element of input data: %s" % input_data)
    print("Weights for nodes in hidden layer 0: %s" % weights['node0'])
    print("Weights for nodes in hidden layer 1: %s" % weights['node1'])
    print("Weights for nodes in output: %s" % weights['output'])

    # Calculation for node 0 value
    node_0_input = (input_data_row * weights['node0']).sum()
    print("Node 0 in hidden layer before activation function: %d" % node_0_input)
    node_0_output = ReLu(node_0_input)
    print("Node 0 in hidden layer after activation function: %d" % node_0_output)

    # Calculation for node 1 value
    node_1_input = (input_data_row * weights['node1']).sum()
    print("Node 1 in hidden layer before activation function: %d" % node_1_input)
    node_1_output = ReLu(node_1_input)
    print("Node 1 in hidden layer after activation function: %d" % node_1_output)

    # put node values into array : hidden_layer_output
    hidden_layer_output = np.array([node_0_output, node_1_output])
    print("Hidden layer: %s" % hidden_layer_output)

    # Calculate model output
    input_to_final_layer = (hidden_layer_output * weights['output']).sum()
    print("Output layer before activation function: %d" % input_to_final_layer)
    model_output = ReLu(input_to_final_layer)
    print("Output layer after activation function: %d" % model_output)

    # Return model output
    return model_output


# Create Empty list to store prediction results
results = []
for input_data_row in input_data:
    # Append prediction to result
    results.append(predict_with_NN(input_data_row, weights))


print(results)



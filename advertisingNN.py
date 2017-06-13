'''
Created on Apr 29, 2017

@author: Robert
'''

from random import random
from math import exp
from math import log
from random import seed
from copy import deepcopy

def main():
    sowTrainData = [[97.329, 1, 29.1987],
            [104.267, 1, 31.2801],
            [104.079, 2, 62.4474],
            [108.986, 2, 65.3916],
            [114.758, 2, 68.8548],
            [117.522, 2, 70.5132],
            [131.998, 3, 118.7982]]
    
    sowTestData = [[99.856, 1, 29.9568],
           [107.891, 1, 32.3673],
           [106.136, 1, 31.8408],
           [110.794, 2, 66.4764],
           [123.794, 3, 111.4146],
           [124.338, 2, 74.6028],
           [140.874, 3, 126.7866]]
    
    #n_inputs = len(sowTrainData[0]) - 1
    #n_outputs = len(set([row[-1] for row in sowTrainData]))
    
    tacaTrainData = [[97.329, 1, 1],
            [104.267, 1, 1],
            [104.079, 2, 0],
            [108.986, 2, 0],
            [114.758, 2, 0],
            [117.522, 2, 0],
            [131.998, 3, 1]]
    
    tacaTestData = [[99.856, 1, 1],
            [107.891, 1, 1],
            [106.136, 1, 1],
            [110.794, 2, 0],
            [123.794, 3, 1],
            [124.338, 2, 1],
            [140.874, 3, 0]]
    #n_inputs = len(tacaTrainData[0]) - 1
    #n_outputs = len(set([row[-1] for row in tacaData]))
    
    seed(1)
    
    dataset = [[2.7810836,2.550537003,0],
    [1.465489372,2.362125076,0],
    [3.396561688,4.400293529,0],
    [1.38807019,1.850220317,0],
    [3.06407232,3.005305973,0],
    [7.627531214,2.759262235,1],
    [5.332441248,2.088626775,1],
    [6.922596716,1.77106367,1],
    [8.675418651,-0.242068655,1],
    [7.673756466,3.508563011,1]]
    #n_inputs = len(dataset[0]) - 1
    #n_outputs = len(set([row[-1] for row in dataset]))

    
    n_outputs = 1
    n_inputs = 2
    
    MetricType = enum(TACA='TACA', SOW='SOW')
    n_hidden_nodes = 4
    
    network = initialize_network(n_inputs, n_hidden_nodes, n_outputs)
    #train_network(network, sowTrainData, .01, 70, n_outputs, MetricType.SOW)
    train_network(network, tacaTrainData, .0001, 70, n_outputs, MetricType.TACA)
    for layer in network:
        print(layer)
    evaluate_network(network, tacaTestData, n_hidden_nodes, n_outputs, MetricType.TACA)
    return 1

def enum(**enums):
    return type('Enum', (), enums)

def initialize_network(n_inputs, n_hidden, n_outputs):
    network = list()
    hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
    network.append(hidden_layer)
    output_layer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
    network.append(output_layer)
    print(network)
    return network

def activate(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights)-1):
        activation += weights[i] * inputs[i]
    return activation

# Transfer neuron activation
def sigmoidActivation(activation):
    return 1.0 / (1.0 + exp(-activation))

# Transfer neuron activation
def rampActivation(activation):
    return log(1 + exp(activation))

def forward_propagate(network, row, metric_type):
    inputs = row
    layer_count = 0
    for layer in network:
        layer_count += 1
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            if(metric_type == 'SOW' and layer_count < len(network)):
                neuron['output'] = rampActivation(activation)
            else:
                neuron['output'] = sigmoidActivation(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs

# Calculate the derivative of an neuron output using the sigmoid activation function
def sigmoid_derivative(output):
    return output * (1.0 - output)

# Calculate the derivative of an neuron output using the ramp activation function
#def ramp_derivative(output):
#    return 1.0 - exp(-output)

# Backpropagate error and store in neurons
def backward_propagate_error(network, expected, metric_type):
    print('Backward Propagating...')
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network)-1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j] - neuron['output'])
        for j in range(len(layer)):
            neuron = layer[j]
            #if(metric_type == 'SOW'):
            #    neuron['delta'] = errors[j] * ramp_derivative(neuron['output'])
            #else:
            neuron['delta'] = errors[j] * sigmoid_derivative(neuron['output'])
    return network

def update_weights(network, row, eta):
    print('Updating Weights...')
    for i in range(len(network)):
        inputs = row[:-1]
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] += eta * neuron['delta'] * inputs[j]
            neuron['weights'][-1] += eta * neuron['delta']
    return network
            
def train_network(network, train, eta, n_epoch, n_outputs, metric_type):
    for epoch in range(n_epoch):
        sum_error = 0
        for row in train:
            outputs = forward_propagate(network, row, metric_type)
            expected = [0 for i in range(n_outputs)]
            expected[-1] = row[-1]

            sum_error += sum([(expected[i]-outputs[i]) ** 2 for i in range(len(expected))])
            network = backward_propagate_error(network, expected, metric_type)
            network = update_weights(network, row, eta)
        print('>epoch=%d, eta=%.3f, error=%.3f' % (epoch, eta, sum_error/len(train)))
        
def evaluate_network(network, test_data, hidden_nodes, output_nodes, metric_type):
    print('Evaluating...')
    print(network)
    mse = 0
    for row in test_data:
        hidden_node_values = [0 for node in range(hidden_nodes)]
        current_layer = 0
        for layer in network:
            if current_layer == 0:
                for node in range(len(hidden_node_values)):
                    activity_value = 0
                    for input in range(len(row)-1):
                        activity_value += row[input]*layer[node]['weights'][input]
                    activity_value += layer[node]['weights'][-1]
                    activation_value = sigmoidActivation(activity_value)
                    hidden_node_values[node] = activation_value
            else:
                output_value = 0
                for node in range(len(hidden_node_values)):
                    output_value += hidden_node_values[node] * layer[-1]['weights'][node]
                output_value += layer[-1]['weights'][-1]
                if metric_type == 'SOW':
                    output_value = rampActivation(output_value)
                else:
                    output_value = sigmoidActivation(output_value)
                    
            current_layer += 1
        mse += (row[-1] - output_value) ** 2
        print('Expected:')
        print(row)
        print('Predicted')
        print(output_value)
    mse = mse/len(test_data)
        
        
    if(metric_type == 'TACA'):
        false_positives = 0
        false_negatives = 0
        true_positives = 0
        true_negatives = 0
    
    else: 
        print(mse)
        
    
    return    
    
# Evaluate SOW using mean squared error 
def mean_squared_error(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0

# Evaluate the TACA using receiver operator characteristics


if __name__ == '__main__':
    main()
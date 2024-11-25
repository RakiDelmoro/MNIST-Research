import math
import torch
import cupy as cp
import numpy as np

def cupy_array(x):
    return cp.round(cp.array(x, dtype=cp.float32), 4)

def one_hot(x, number_of_classes):
    return cupy_array(cp.eye(number_of_classes)[x])

def axons_initialization(input_feature, output_feature):
    weights = torch.empty((input_feature, output_feature))
    bias = torch.empty(output_feature)
    torch.nn.init.kaiming_normal_(weights, a=math.sqrt(5))
    # weights = cp.random.normal(loc=0, scale=0.01, size=(input_feature, output_feature))
    return cupy_array(weights)

def resiudal_connections_initialization(network_feature_sizes):
    network_connections = []
    step_magnitude = 1
    step_back_size = 2**step_magnitude
    residual_neurons_size = 0
    for layer_idx in range(len(network_feature_sizes)-1):
        have_residual_neurons = layer_idx > step_back_size
        if have_residual_neurons:
            residual_neurons_size += (network_feature_sizes[layer_idx] // step_back_size)
            input_neurons_size = network_feature_sizes[layer_idx] + residual_neurons_size
            output_neurons_size = network_feature_sizes[layer_idx+1]
            step_magnitude += 1
            step_back_size = 2**step_magnitude
        else:
            input_neurons_size = network_feature_sizes[layer_idx] + residual_neurons_size
            output_neurons_size = network_feature_sizes[layer_idx+1]
        axons = axons_initialization(input_neurons_size, output_neurons_size)
        network_connections.append(axons)
    return network_connections

def count_parameters(network_connections):
    return sum([param.shape[0]*param.shape[1] for param in network_connections])


import math
import torch
import cupy as cp
import numpy as np

def cupy_array(x):
    return cp.round(cp.array(x, dtype=cp.float32), 4)

def one_hot(x, number_of_classes):
    return cupy_array(cp.eye(number_of_classes)[x])

def residual_axons_and_dentrites_initialization(network_feature_sizes):
    neurons_sizes_pulled = [2**i for i in range(8)]
    residual_connection_idx = 0
    network_axons_and_dentrites = []
    for idx in range(len(network_feature_sizes)-1):
        residual_idx = 0 if residual_connection_idx >= len(neurons_sizes_pulled) else residual_connection_idx
        first_layer = idx == 0
        if first_layer:
            input_nodes_size = network_feature_sizes[idx]
            output_nodes_size = network_feature_sizes[idx+1]
        else:
            pulled_neurons_sizes =  neurons_sizes_pulled[-(residual_idx+1):]
            neurons_pulled_size = pulled_neurons_sizes[-1] if len(pulled_neurons_sizes) == 1 else sum(pulled_neurons_sizes)
            input_nodes_size = network_feature_sizes[idx] + neurons_pulled_size
            output_nodes_size = network_feature_sizes[idx+1]
            if residual_connection_idx < len(neurons_sizes_pulled):
                residual_connection_idx += 1
            else:
                input_nodes_size = network_feature_sizes[idx]
                output_nodes_size = network_feature_sizes[idx+1]
                residual_connection_idx = 0
        axons, dentrites = axons_and_dentrites_initialization(input_nodes_size, output_nodes_size)
        network_axons_and_dentrites.append([axons, dentrites])
    return network_axons_and_dentrites

def axons_and_dentrites_initialization(input_feature, output_feature):
    weights = torch.empty((input_feature, output_feature))
    bias = torch.empty(output_feature)
    torch.nn.init.kaiming_normal_(weights, a=math.sqrt(5))
    fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(weights)
    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
    torch.nn.init.uniform_(bias, -bound, bound)

    # weights = cp.random.randn(input_feature, output_feature) * 0.01

    return cupy_array(weights), cupy_array(bias)

def axons_initialization(input_feature, output_feature):
    bound_w = cp.sqrt(5) / cp.sqrt(input_feature) if input_feature > 0 else 0
    weights = cp.random.uniform(-bound_w, bound_w, size=(input_feature, output_feature), dtype=cp.float32)
    # weights = cp.random.randn(input_feature, output_feature) * 0.01
    return weights

def dentrites_initialization(output_feature):
    bound_b = 1 / cp.sqrt(output_feature) if output_feature > 0 else 0
    bias = cp.random.uniform(-bound_b, bound_b, size=(output_feature,), dtype=cp.float32)
    # bias = cp.zeros(output_feature)
    return bias

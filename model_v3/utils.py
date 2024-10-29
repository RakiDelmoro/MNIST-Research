import cupy as cp
from cupy_utils.utils import one_hot
from nn_utils.activation_functions import leaky_relu, softmax
from nn_utils.loss_functions import cross_entropy_loss

def axon_and_dentrites_initialization(input_feature, output_feature):
    bound_w = cp.sqrt(5.0) / cp.sqrt(input_feature) if input_feature > 0 else 0
    axons = cp.random.uniform(-bound_w, bound_w, size=(input_feature, output_feature))
    bound_b = 1.0 / cp.sqrt(input_feature) if input_feature > 0 else 0
    dentrites = cp.random.uniform(-bound_b, bound_b, size=(output_feature,))
    return axons, dentrites

def network_axons_and_dentrites(model_feature_sizes):
    layers_axons_and_dentrites = []
    for connection_idx in range(len(model_feature_sizes)-1):
        axons, dentrites = axon_and_dentrites_initialization(model_feature_sizes[connection_idx], model_feature_sizes[connection_idx+1])
        layers_axons_and_dentrites.append([axons, dentrites])
    return layers_axons_and_dentrites

def get_forward_activations(input_neurons, layers_parameters):
    neurons = cp.array(input_neurons)
    total_connections = len(layers_parameters)
    neurons_activations = [neurons]
    for each_connection in range(total_connections):
        axons = layers_parameters[each_connection][0]
        dentrites = layers_parameters[each_connection][1]
        neurons = cp.dot(neurons, axons) + dentrites
        neurons_activations.append(neurons)
    return neurons_activations

def get_backward_activations(input_neurons, layers_parameters):
    neurons = cp.array(input_neurons)
    total_connections = len(layers_parameters)-1
    neurons_activations = [neurons]
    for each_connection in range(total_connections):
        axons = layers_parameters[-(each_connection+1)][0].transpose()
        dentrites = layers_parameters[-(each_connection+2)][1]
        neurons = cp.dot(neurons, axons) + dentrites
        neurons_activations.append(neurons)
    return neurons_activations

def calculate_network_stress(forward_activations, backward_activations):
    layers_stress = []
    total_activations = len(backward_activations)
    for each_activation in range(total_activations):
        stress = forward_activations[-(each_activation+1)] - backward_activations[each_activation]
        layers_stress.append(stress)
    return layers_stress
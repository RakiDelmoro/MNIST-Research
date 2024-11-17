import cupy as cp
from cupy_utils.utils import axons_and_dentrites_initialization


def update_axons(previous_activation, current_activation, axons):
    pass

def calculate_activation_error(current_activation, axons):
    reconstruct_previous_activation = cp.dot(current_activation, axons.transpose())
    reconctruct_current_activation = cp.dot(reconstruct_previous_activation, axons)
    activation_error = current_activation - reconctruct_current_activation
    avg_reconstructed_error = cp.sum(cp.linalg.norm(activation_error)**2) / current_activation.shape[0]
    return avg_reconstructed_error

def forward_once_for_next_layer(dataloader, axons):
    pass
# def forward_in_layer(dataloader, axons, learning_rate):
#     for each in range(10):
#         per_batch_stress = []
#         for i, (input_neurons, _) in enumerate(dataloader):
#             layer_activation = cp.dot(input_neurons, axons)
#             activation_stress = calculate_activation_error(layer_activation)
#             update_axons(input_neurons,layer_activation,axons)
    
#     return forward_once_for_next_layer(dataloader, axons)

def training_layers(dataloader, network_parameters, learning_rate):
    per_batch_stress = []
    total_layers = len(network_parameters-1)
    for each_layer in range(total_layers):
        layer_axons = network_parameters[each_layer]

    return

def custom_model_v1(network_architecture, training_dataloader, learning_rate, epochs):
    model_parameters = [axons_and_dentrites_initialization(network_architecture[feature_idx], network_architecture[feature_idx+1]) for feature_idx in range(len(network_architecture)-1)]

    training_layers(training_dataloader, model_parameters, learning_rate)
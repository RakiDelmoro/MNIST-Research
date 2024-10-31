import torch
import cupy as cp
from cupy_utils.utils import one_hot
from nn_utils.loss_functions import cross_entropy_loss
from nn_utils.activation_functions import leaky_relu, softmax
# from model_v2.utils_array_format import forward_pass_architecture, backward_pass_architecture, get_network_activations_array_format, layers_of_neurons_stress, nudge_axons_and_dentrites, test_run_result
from model_v2.utils_py_list_format import initialize_network_parameters, training_layers, test_layers

def neural_network(network_architecture: list, training_dataloader, validation_dataloader, learning_rate, epochs):
    # Return Tuple of axons and dentrites
    forward_pass_axons_and_dentrites = initialize_network_parameters(network_features_sizes=network_architecture)
    backward_pass_axons_and_dentrites = initialize_network_parameters(network_features_sizes=network_architecture[::-1])

    def training_run(dataloader):
        training_loss = training_layers(dataloader, forward_pass_axons_and_dentrites, backward_pass_axons_and_dentrites, learning_rate)
        return training_loss
    
    def test_run(dataloader):
        accuracy = test_layers(dataloader, forward_pass_axons_and_dentrites, backward_pass_axons_and_dentrites)
        return accuracy

    for epoch in range(epochs):
        training_loss = training_run(training_dataloader)
        accuracy = test_run(validation_dataloader)
        print(f"EPOCH: {epoch+1} Training Loss: {training_loss} Model Accuracy: {accuracy}")

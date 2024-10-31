import cupy as cp
from features import GREEN, RED, RESET
from cupy_utils.utils import one_hot
from nn_utils.activation_functions import leaky_relu, softmax
from model_v3.utils import network_axons_and_dentrites, get_forward_activations, get_backward_activations, calculate_network_stress, update_parameters, test_layers, training_layers

def neural_network(network_architecture, training_loader, validation_loader, learning_rate):
    parameters = network_axons_and_dentrites(network_architecture)

    def training_run():
        return training_layers(training_loader, parameters, learning_rate=learning_rate)

    def test_run():
        return test_layers(validation_loader, parameters)

    for epoch in range(10):
        print(f'EPOCHS: {epoch+1}')
        loss = training_run()
        accuracy = test_run()
        print(f'EPOCH {epoch+1} loss: {loss} accuracy: {accuracy}')

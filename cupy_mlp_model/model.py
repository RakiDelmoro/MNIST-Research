import cupy as cp
from features import GREEN, RED, RESET
from nn_utils.loss_functions import cross_entropy_loss
from cupy_utils.utils import axons_and_dentrites_initialization
from cupy_mlp_model.utils import training_layers, test_layers

def cupy_mlp_neural_network(network_feature_sizes, training_loader, validation_loader, learning_rate, epochs):
    network_parameters = [axons_and_dentrites_initialization(network_feature_sizes[feature_idx], network_feature_sizes[feature_idx+1])
                         for feature_idx in range(len(network_feature_sizes)-1)]

    def training_run():
        return training_layers(dataloader=training_loader, layers_parameters=network_parameters, learning_rate=learning_rate)

    def test_run():
        return test_layers(dataloader=validation_loader, layers_parameters=network_parameters)

    for epoch in range(epochs):
        print(f'EPOCH: {epoch+1}')
        model_stress = training_run()
        model_accuracy = test_run()
        print(f'Average loss per epoch: {model_stress} accuracy: {model_accuracy}')

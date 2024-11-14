import cupy as cp
from features import GREEN, RED, RESET
from nn_utils.loss_functions import cross_entropy_loss
from cupy_utils.utils import axons_and_dentrites_initialization, residual_axons_and_dentrites_initialization
from cupy_mlp_model.backprop_utils import training_layers, test_layers
# from cupy_mlp_model.residual_utils import residual_training_layers, residual_test_layers
# from cupy_mlp_model.backprop_combined_oja_utils import training_layers, test_layers
from cupy_mlp_model.residual_v2_utils import residual_test_layers, residual_training_layers

def cupy_mlp_neural_network(network_feature_sizes, training_loader, validation_loader, learning_rate, epochs, residual_use):
    # [1, 2, 4, 8, 16, 32, 64, 128]
    residual_neurons_sizes = [(2**n) for n in range(network_feature_sizes[1]) if (2**n) < network_feature_sizes[1]]
    # network_parameters = [axons_and_dentrites_initialization(network_feature_sizes[feature_idx], network_feature_sizes[feature_idx+1])
    #                      for feature_idx in range(len(network_feature_sizes)-1)]
    network_axons_and_dentrites = residual_axons_and_dentrites_initialization(network_feature_sizes, residual_neurons_sizes)
    def training_run():
        if residual_use:
            return residual_training_layers(dataloader=training_loader, layers_parameters=network_axons_and_dentrites, residual_neurons_sizes=residual_neurons_sizes, learning_rate=learning_rate)
        else:
            return training_layers(dataloader=training_loader, layers_parameters=network_axons_and_dentrites, learning_rate=learning_rate)

    def test_run():
        if residual_use:
            return residual_test_layers(dataloader=validation_loader, layers_parameters=network_axons_and_dentrites, residual_idx=residual_neurons_sizes)
        else:
            return test_layers(dataloader=validation_loader, layers_parameters=network_axons_and_dentrites)

    for epoch in range(epochs):
        print(f'EPOCH: {epoch+1}')
        model_stress = training_run()
        model_accuracy = test_run()
        # print(f'accuracy: {model_accuracy}')
        print(f'Average loss per epoch: {model_stress} accuracy: {model_accuracy}')

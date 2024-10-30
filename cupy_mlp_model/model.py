import cupy as cp
from torch.nn.functional import one_hot
from nn_utils.loss_functions import cross_entropy_loss
from cupy_utils.utils import axons_initialization, dentrites_initialization


def neural_network(network_feature_sizes):
    layers_parameters = [[axons_initialization(network_feature_sizes[feature_idx], network_feature_sizes[feature_idx+1]), dentrites_initialization(network_feature_sizes[feature_idx+1])]
                         for feature_idx in range(len(network_feature_sizes)-1)]

    def forward_pass(input_neurons):
        neurons = cp.array(input_neurons)
        total_activations = len(layers_parameters)
        neurons_activations = [neurons]
        for each in range(total_activations):
            axons = layers_parameters[each][0]
            dentrites = layers_parameters[each][1]
            neurons = cp.dot(neurons, axons) + dentrites
            neurons_activations.append(neurons)
        return neurons_activations

    def backward_pass(model_prediction, expected_model_prediction):
        expected_model_prediction = cp.array(expected_model_prediction)
        total_layers_loss = len(layers_parameters)-1
        network_error, layer_loss = cross_entropy_loss(model_prediction, expected_model_prediction)
        layers_losses = [layer_loss]
        for each_connection in range(total_layers_loss):
            axons = layers_parameters[-(each_connection+1)][0]
            layer_loss = cp.dot(layer_loss, axons.transpose())
            layers_losses.append(layer_loss)
        return network_error, layers_losses
    
    def update_layers_parameters(neurons_activations, layers_losses, learning_rate):
        total_parameters = len(layers_parameters)
        for layer_idx in range(total_parameters):
            axons = layers_parameters[-(layer_idx+1)][0]
            dentrites = layers_parameters[-(layer_idx+1)][1]
            neuron_activation = neurons_activations[-(layer_idx+2)]
            # This line fixed the Nan problem!
            loss = layers_losses[layer_idx] / 2098

            axons -= learning_rate * cp.dot(neuron_activation.transpose(), loss)
            dentrites -= learning_rate * cp.sum(loss, axis=0)

    def runner(training_loader, validation_loader):
        for epoch in range(10):
            print(f'EPOCH: {epoch+1}')
            per_batch_losses = []
            for input_batch, expected_batch in training_loader:
                neurons_activations = forward_pass(input_neurons=input_batch)
                error, layers_losses = backward_pass(neurons_activations[-1], expected_batch)
                per_batch_losses.append(error)
                print(error)
                update_layers_parameters(neurons_activations, layers_losses, 0.01)
            print(f'Average loss per epoch: {cp.mean(cp.array(per_batch_losses))}')

    return runner
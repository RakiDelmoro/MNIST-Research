import cupy as cp
from features import GREEN, RED, RESET
from nn_utils.loss_functions import cross_entropy_loss
from nn_utils.activation_functions import leaky_relu, softmax
from model_v3.utils import network_axons_and_dentrites, get_forward_activations, get_backward_activations, calculate_network_stress

def neural_network(network_architecture, training_loader, validation_loader, lr):
    parameters = network_axons_and_dentrites(network_architecture)

    def training_layers(dataloader):
        loss_per_batch = []
        for input_batch, expected_batch in dataloader:
            forward_activations = get_forward_activations(input_batch, parameters)
            backward_activations = get_backward_activations(expected_batch, parameters)
            network_stress = calculate_network_stress(forward_activations, backward_activations)
            #TODO: Create a function for parameters update
        return cp.mean(cp.array(loss_per_batch))

    for epoch in range(10):
        print(f'EPOCHS: {epoch+1}')
        loss = training_layers(dataloader=training_loader)
        print(loss)
import torch
import cupy as cp
from cupy_utils.utils import one_hot
from features import GREEN, RED, RESET
from nn_utils.loss_functions import cross_entropy_loss
from nn_utils.activation_functions import leaky_relu, softmax
# from model_v2.utils_array_format import forward_pass_architecture, backward_pass_architecture, get_network_activations_array_format, layers_of_neurons_stress, nudge_axons_and_dentrites, test_run_result
from model_v2.utils_py_list_format import forward_pass_architecture, backward_pass_architecture, get_network_activations, layers_of_neurons_stress, nudge_axons_and_dentrites, test_run_result

def neural_network(network_architecture: list, training_dataloader, validation_dataloader, learning_rate, epochs):
    # Return Tuple of axons and dentrites
    forward_pass_axons_and_dentrites = forward_pass_architecture(network_features_size=network_architecture)
    backward_pass_axons_and_dentrites = backward_pass_architecture(network_features_size=network_architecture[::-1])

    def forward_in_neurons(neurons):
        neurons_activations = get_network_activations(input_neurons=neurons,  network_architecture=network_architecture, network_axons_and_dentrites=forward_pass_axons_and_dentrites)
        return neurons_activations
    
    def backward_in_neurons(neurons):
        neurons_activations = get_network_activations(input_neurons=neurons, network_architecture=network_architecture, network_axons_and_dentrites=backward_pass_axons_and_dentrites)
        return neurons_activations

    def calculate_network_stress(forward_activations, backward_activations):
        layers_stress = layers_of_neurons_stress(network_architecture=network_architecture, forward_pass_activations=forward_activations, backward_pass_activations=backward_activations)
        return layers_stress

    def training_run(training_loader, learning_rate):
        each_batch_layers_stress = []
        for input_batch, expected_batch in training_loader:
            forward_activations = forward_in_neurons(input_batch)
            backward_activations = backward_in_neurons(expected_batch)
            neurons_stress = calculate_network_stress(forward_activations, backward_activations)
            # Update bidirectional passes parameters
            nudge_axons_and_dentrites(layers_stress=neurons_stress, neurons_activations=forward_activations, axons_and_dentrites=forward_pass_axons_and_dentrites, for_backward_pass=False, learning_rate=learning_rate)
            nudge_axons_and_dentrites(layers_stress=neurons_stress, neurons_activations=backward_activations, axons_and_dentrites=backward_pass_axons_and_dentrites, for_backward_pass=True, learning_rate=learning_rate)
            each_batch_layers_stress.append(neurons_stress)

        return cp.mean(cp.array([cp.mean(layer_stress) for layers_stress in each_batch_layers_stress for layer_stress in layers_stress]))

    def test_run(dataloader):
        accuracy, correct_samples, wrong_samples, model_prediction, model_expected_prediction = test_run_result(dataloader, forward_in_neurons)
        print(f"{GREEN}Model Correct Predictions{RESET}")
        for indices in correct_samples: print(f"Digit Image is: {GREEN}{model_expected_prediction[indices]}{RESET} Model Prediction: {GREEN}{model_prediction[indices]}{RESET}")
        print(f"{RED}Model Wrong Predictions{RESET}")
        for indices in wrong_samples: print(f"Digit Image is: {RED}{model_expected_prediction[indices]}{RESET} Model Predictions: {RED}{model_prediction[indices]}{RESET}")

        return accuracy

    for epoch in range(epochs):
        training_loss = training_run(training_dataloader, learning_rate)
        accuracy = test_run(validation_dataloader)
        print(f"EPOCH: {epoch+1} Training Loss: {training_loss} Model Accuracy: {accuracy}")

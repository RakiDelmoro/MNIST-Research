import torch
import cupy as cp
from cupy_utils.utils import one_hot
from features import GREEN, RED, RESET
from nn_utils.loss_functions import cross_entropy_loss
from nn_utils.activation_functions import leaky_relu, softmax
# from model_v2.utils_array_format import forward_pass_architecture, backward_pass_architecture, get_network_activations_array_format, layers_of_neurons_stress, nudge_axons_and_dentrites, test_run_result
from model_v2.utils_py_list_format import initialize_network_parameters, training_layers, test_layers

def neural_network(network_architecture: list, training_dataloader, validation_dataloader, learning_rate, epochs):
    # Return Tuple of axons and dentrites
    forward_pass_axons_and_dentrites = initialize_network_parameters(network_features_sizes=network_architecture)
    backward_pass_axons_and_dentrites = initialize_network_parameters(network_features_sizes=network_architecture[::-1])

    def test_run(forward_parameters, backward_paramters):
        accuracy, correct_samples, wrong_samples, model_prediction, model_expected_prediction = test_layers(dataloader=validation_dataloader, forward_pass_trained_parameters=forward_parameters, backward_pass_trained_parameters=backward_paramters)
        print(f"{GREEN}Model Correct Predictions{RESET}")
        for indices in correct_samples: print(f"Digit Image is: {GREEN}{model_expected_prediction[indices]}{RESET} Model Prediction: {GREEN}{model_prediction[indices]}{RESET}")
        print(f"{RED}Model Wrong Predictions{RESET}")
        for indices in wrong_samples: print(f"Digit Image is: {RED}{model_expected_prediction[indices]}{RESET} Model Predictions: {RED}{model_prediction[indices]}{RESET}")

        return accuracy

    for epoch in range(epochs):
        training_loss = training_layers(dataloader=training_dataloader, forward_pass_parameters=forward_pass_axons_and_dentrites, backward_pass_parameters=backward_pass_axons_and_dentrites, learning_rate=learning_rate)
        accuracy = test_run(forward_pass_axons_and_dentrites, backward_pass_axons_and_dentrites)
        print(f"EPOCH: {epoch+1} Training Loss: {training_loss} Model Accuracy: {accuracy}")

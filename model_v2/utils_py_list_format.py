import random
import math
import torch
import cupy as cp
from functools import reduce
from cupy_utils.utils import one_hot
from features import GREEN, RED, RESET

def axons_initialization(input_feature, output_feature):
    bound_w = cp.sqrt(5) / cp.sqrt(input_feature) if input_feature > 0 else 0
    weights = cp.random.uniform(-bound_w, bound_w, size=(input_feature, output_feature))
    return weights

def dentrites_initialization(output_feature):
    bound_b = 1 / cp.sqrt(output_feature) if output_feature > 0 else 0
    bias = cp.random.uniform(-bound_b, bound_b, size=(output_feature,))
    return bias

def initialize_network_parameters(network_features_sizes: list):
    layers_axons = []
    layers_dentrites = []
    for feature_size_idx in range(len(network_features_sizes)-1):
        input_feature = network_features_sizes[feature_size_idx]
        output_feature = network_features_sizes[feature_size_idx+1]
        axons = axons_initialization(input_feature, output_feature)
        dentrites = dentrites_initialization(output_feature)
        layers_axons.append(axons)
        layers_dentrites.append(dentrites)
    return layers_axons, layers_dentrites

def get_network_activations(input_neurons, total_connections, network_axons_and_dentrites):
    neurons = cp.array(input_neurons)
    neurons_activations = [neurons]
    for neurons_layer_idx in range(total_connections):
        axons = network_axons_and_dentrites[0][neurons_layer_idx]
        dentrites = network_axons_and_dentrites[-1][neurons_layer_idx]
        neurons = cp.dot(neurons, axons)
        neurons_activations.append(neurons)
    return neurons_activations

def layers_of_neurons_stress(total_activations, forward_pass_activations, backward_pass_activations):
    neurons_stress = []
    for activation_idx in range(total_activations):
        forward_activation = forward_pass_activations[activation_idx]
        backward_activation = backward_pass_activations[-(activation_idx+1)]
        stress = (forward_activation - backward_activation)
        neurons_stress.append(stress)
    return neurons_stress

def nudge_forwad_pass_parameters(layer_index, neurons_activations, layers_stress, layers_axons, layers_dentrites, learning_rate):
    previous_neurons_activation = neurons_activations[layer_index]
    neurons_stress = layers_stress[layer_index+1]
    axons = layers_axons[layer_index]
    dentrites = layers_dentrites[layer_index]
    
    indices_to_be_nudge_up = cp.dot(previous_neurons_activation.transpose(), neurons_stress) < 0
    indices_to_be_nudge_down = ~indices_to_be_nudge_up

    batch_size = previous_neurons_activation.shape[0]
    update_amount = (learning_rate * cp.dot(previous_neurons_activation.transpose(), neurons_stress) / batch_size)
    axons -= cp.where(indices_to_be_nudge_down, update_amount,0)
    axons -= cp.where(indices_to_be_nudge_up, update_amount,0)

def nudge_backward_pass_parameters(layer_index, neurons_activations, layers_stress, layers_axons, layers_dentrites, learning_rate):
    previous_neurons_activation = neurons_activations[layer_index]
    neurons_stress = layers_stress[-(layer_index+2)]
    axons = layers_axons[layer_index]
    dentrites = layers_dentrites[layer_index]
    
    indices_to_be_nudge_up = cp.dot(previous_neurons_activation.transpose(), neurons_stress) > 0
    indices_to_be_nudge_down = ~indices_to_be_nudge_up
    
    batch_size = previous_neurons_activation.shape[0]
    update_amount = (learning_rate * cp.dot(previous_neurons_activation.transpose(), neurons_stress) / batch_size)
    axons += cp.where(indices_to_be_nudge_up, update_amount,0)
    axons += cp.where(indices_to_be_nudge_down, update_amount,0)

def nudge_axons_and_dentrites(layers_stress, neurons_activations, axons_and_dentrites, for_backward_pass, learning_rate):
    layers_axons = axons_and_dentrites[0]
    layers_dentrites = axons_and_dentrites[-1]
    total_connections = len(layers_axons)
    for connection_idx in range(total_connections):
        if not for_backward_pass:
            nudge_forwad_pass_parameters(connection_idx, neurons_activations, layers_stress, layers_axons, layers_dentrites, learning_rate)
        else:
            nudge_backward_pass_parameters(connection_idx, neurons_activations, layers_stress, layers_axons, layers_dentrites, learning_rate)

def visualize_neurons_activity(forward_activations, backward_activations, neurons_stress):
    for layer_idx in range(len(forward_activations)):
        print(f"({forward_activations[layer_idx][0][1].tolist():10.6e} {backward_activations[-(layer_idx+1)][0][1].tolist():10.6e} {neurons_stress[layer_idx][0][1].tolist():10.6e})", end=" ")
    print("")

def training_layers(dataloader, forward_pass_parameters, backward_pass_parameters, learning_rate):
    each_batch_layers_stress = []
    for input_batch, expected_batch in dataloader:
        forward_pass_activations = get_network_activations(input_neurons=input_batch, total_connections=len(forward_pass_parameters[0]), network_axons_and_dentrites=forward_pass_parameters)
        backward_pass_activations = get_network_activations(input_neurons=expected_batch, total_connections=len(backward_pass_parameters[0]), network_axons_and_dentrites=backward_pass_parameters)
        neurons_activation_stress = layers_of_neurons_stress(total_activations=len(forward_pass_activations), forward_pass_activations=forward_pass_activations, backward_pass_activations=backward_pass_activations)
        # visualize_neurons_activity(forward_pass_activations, backward_pass_activations, neurons_activation_stress)
        nudge_axons_and_dentrites(layers_stress=neurons_activation_stress, neurons_activations=forward_pass_activations, axons_and_dentrites=forward_pass_parameters, for_backward_pass=False, learning_rate=learning_rate)
        nudge_axons_and_dentrites(layers_stress=neurons_activation_stress, neurons_activations=backward_pass_activations, axons_and_dentrites=backward_pass_parameters, for_backward_pass=True, learning_rate=learning_rate)
        each_batch_layers_stress.append(neurons_activation_stress)
    network_loss_avg = cp.mean(cp.array([cp.sum(layer_stress) for layers_stress in each_batch_layers_stress for layer_stress in layers_stress]))
    return network_loss_avg

def test_layers(dataloader, forward_pass_trained_parameters, backward_pass_trained_parameters):
    def forward_pass_inference_run(input_batch):
        neurons = cp.array(input_batch)
        total_connections = len(forward_pass_trained_parameters[0])
        for connection_idx in range(total_connections):
            axons = forward_pass_trained_parameters[0][connection_idx]
            dentrites = forward_pass_trained_parameters[-1][connection_idx]
            neurons = cp.dot(neurons, axons) + dentrites
        return neurons

    per_batch_accuracy = []
    wrong_samples_indices = []
    correct_samples_indices = []
    model_predictions = []
    expected_model_prediction = []
    for input_image_batch, expected_batch in dataloader:
        expected_batch = cp.array(expected_batch)
        model_output = forward_pass_inference_run(input_image_batch)
        batch_accuracy = cp.array(expected_batch.argmax(-1) == (model_output).argmax(-1)).mean()
        correct_indices_in_a_batch = cp.where(expected_batch.argmax(-1) == model_output.argmax(-1))[0]
        wrong_indices_in_a_batch = cp.where(~(expected_batch.argmax(-1) == model_output.argmax(-1)))[0]

        per_batch_accuracy.append(batch_accuracy.item())
        correct_samples_indices.append(correct_indices_in_a_batch)
        wrong_samples_indices.append(wrong_indices_in_a_batch)
        model_predictions.append(model_output.argmax(-1))
        expected_model_prediction.append(expected_batch.argmax(-1))

    model_accuracy = cp.mean(cp.array(per_batch_accuracy))
    correct_samples = cp.concatenate(correct_samples_indices)[list(range(0,len(correct_samples_indices)))]
    wrong_samples = cp.concatenate(wrong_samples_indices)[list(range(0,len(wrong_samples_indices)))]
    model_prediction = cp.concatenate(model_predictions)
    model_expected_prediction = cp.concatenate(expected_model_prediction)

    print(f"{GREEN}Model Correct Predictions{RESET}")
    for indices in correct_samples: print(f"Digit Image is: {GREEN}{model_expected_prediction[indices]}{RESET} Model Prediction: {GREEN}{model_prediction[indices]}{RESET}")
    print(f"{RED}Model Wrong Predictions{RESET}")
    for indices in wrong_samples: print(f"Digit Image is: {RED}{model_expected_prediction[indices]}{RESET} Model Predictions: {RED}{model_prediction[indices]}{RESET}")

    return model_accuracy

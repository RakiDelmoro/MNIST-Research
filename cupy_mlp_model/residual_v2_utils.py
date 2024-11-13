import random
import cupy as cp
from features import GREEN, RED, RESET
from cupy_utils.utils import cupy_array
from nn_utils.activation_functions import leaky_relu
from nn_utils.loss_functions import cross_entropy_loss

def apply_residual_connection(neurons_activations, axons, dentrites, residual_neurons_sizes):
    neurons_for_next_layer = []
    for activation_idx, neurons_activation in enumerate(neurons_activations):
        previous_layer_activation = activation_idx == 0
        if previous_layer_activation:
            neurons_for_next_layer.append(neurons_activation)
        else:
            neurons_size = residual_neurons_sizes[-activation_idx]
            neurons_pulled = neurons_activation[:, :neurons_size]
            neurons_for_next_layer.append(neurons_pulled)
    input_neurons =  cp.concatenate(neurons_for_next_layer, axis=-1)
    return leaky_relu((cp.dot(input_neurons, axons)) + dentrites)

def forward_pass_activations(neurons, layers_parameters, residual_neurons_sizes):
    input_neurons = cp.array(neurons)
    neurons_activations = [input_neurons]
    total_activations = len(layers_parameters)
    total_previous_activation_pulled = 2
    without_residual_layer_idx = [layer_idx for layer_idx in range(0, len(layers_parameters), len(residual_neurons_sizes)+1)]
    for layer_idx in range(total_activations):
        axons = layers_parameters[layer_idx][0]
        dentrites = layers_parameters[layer_idx][1]
        if layer_idx in without_residual_layer_idx:
            input_neurons = leaky_relu((cp.dot(input_neurons, axons)) + dentrites)
            neurons_activations.insert(0, input_neurons)
            total_previous_activation_pulled = 2
        else:
            input_neurons = apply_residual_connection(neurons_activations[:total_previous_activation_pulled], axons, dentrites, residual_neurons_sizes)
            neurons_activations.insert(0, input_neurons)
            total_previous_activation_pulled += 1
    return neurons_activations[::-1]

def calculate_layers_stress(neurons_stress, layers_parameters, residual_indexes):
    # TODO: Refactor this code!
    # indexes of forward pass activation that have residual connection
    idx_to_aggregate_stress = [(len(layers_parameters))-index for index in residual_indexes[::-1]]
    backprop_stress_to_aggregate = []
    layers_gradient = [neurons_stress]
    total_layers_stress = len(layers_parameters)-1
    for layer_idx in range(total_layers_stress):
        axons = layers_parameters[-(layer_idx+1)][0]
        if layer_idx in idx_to_aggregate_stress:
            backprop_stress_to_aggregate.append(neurons_stress)
            if len(backprop_stress_to_aggregate) == 1:
                neurons_stress = neurons_stress
            else:
                backprop_aggregated_stress = cp.sum(cp.stack(backprop_stress_to_aggregate), axis=0)
                neurons_stress = backprop_aggregated_stress
        else:
            neurons_stress = cp.dot(neurons_stress, axons.transpose())
        layers_gradient.append(neurons_stress)
    return layers_gradient

def update_layers_parameters(neurons_activations, layers_losses, layers_parameters, learning_rate):
    #TODO: Should aggregate the information of neurons loss for residual neurons
    total_parameters = len(layers_losses)
    for layer_idx in range(total_parameters):
        axons = layers_parameters[-(layer_idx+1)][0]
        dentrites = layers_parameters[-(layer_idx+1)][1]
        previous_activation = neurons_activations[-(layer_idx+2)]
        loss = layers_losses[layer_idx]
        backprop_parameters_nudge = learning_rate * cp.dot(previous_activation.transpose(), loss)
        axons -= (backprop_parameters_nudge / previous_activation.shape[0])
        dentrites -= ((learning_rate * cp.sum(loss, axis=0)) / previous_activation.shape[0])

def residual_training_layers(dataloader, layers_parameters, residual_neurons_sizes, learning_rate):
    per_batch_stress = []
    for i, (input_batch, expected_batch) in enumerate(dataloader):
        neurons_activations = forward_pass_activations(input_batch, layers_parameters, residual_neurons_sizes)
        avg_last_neurons_stress, neurons_stress_to_backpropagate = cross_entropy_loss(neurons_activations[-1], cp.array(expected_batch))
        layers_stress = calculate_layers_stress(neurons_stress_to_backpropagate, layers_parameters, residual_idx)
        update_layers_parameters(neurons_activations, layers_stress, layers_parameters, learning_rate)
        print(f"Loss each batch {i+1}: {avg_last_neurons_stress}\r", end="", flush=True)
        per_batch_stress.append(avg_last_neurons_stress)
    return cp.mean(cp.array(per_batch_stress))

def residual_test_layers(dataloader, layers_parameters, residual_idx):
    correct_predictions = []
    wrong_predictions = []
    model_predictions = []
    for i, (input_image_batch, expected_batch) in enumerate(dataloader):
        expected_batch = cupy_array(expected_batch)
        model_output = forward_pass_activations(input_image_batch, residual_idx, layers_parameters)[-1]
        batched_accuracy = cp.array(expected_batch.argmax(-1) == (model_output).argmax(-1)).astype(cp.float16).mean()
        for each in range(100):
            if model_output[each].argmax(-1) == expected_batch[each].argmax(-1):
                correct_predictions.append((model_output[each].argmax(-1).item(), expected_batch[each].argmax(-1).item()))
            else:
                wrong_predictions.append((model_output[each].argmax(-1).item(), expected_batch[each].argmax(-1).item()))
        print(f"Number of sample: {i+1}\r", end="", flush=True)
        model_predictions.append(batched_accuracy)
    random.shuffle(correct_predictions)
    random.shuffle(wrong_predictions)
    print(f"{GREEN}MODEL Correct Predictions{RESET}")
    [print(f"Digit Image is: {GREEN}{expected}{RESET} Model Prediction: {GREEN}{prediction}{RESET}") for i, (prediction, expected) in enumerate(correct_predictions) if i < 10]
    print(f"{RED}MODEL Wrong Predictions{RESET}")
    [print(f"Digit Image is: {RED}{expected}{RESET} Model Prediction: {RED}{prediction}{RESET}") for i, (prediction, expected) in enumerate(wrong_predictions) if i < 10]
    return cp.mean(cp.array(model_predictions)).item()

import random
import cupy as cp
from features import GREEN, RED, RESET
from cupy_utils.utils import cupy_array
from nn_utils.activation_functions import relu
from nn_utils.loss_functions import cross_entropy_loss

def apply_residual_connection(neurons_activations, axons, dentrites, layer_idx_to_pull_residual_neurons):
    neurons_for_next_layer = []
    previous_layer_idx_to_be_pulled = 1
    previous_neurons_activations = neurons_activations[::-1]
    for layer_activation_idx in range(len(previous_neurons_activations)):
        if layer_activation_idx not in layer_idx_to_pull_residual_neurons:
            layer_no_residual = layer_activation_idx > 0
            if layer_no_residual: continue
            residual_neurons = previous_neurons_activations[layer_activation_idx]
            neurons_for_next_layer.append(residual_neurons)
        else:
            previous_layer_idx_to_be_pulled = 1 if previous_layer_idx_to_be_pulled > len(layer_idx_to_pull_residual_neurons) else previous_layer_idx_to_be_pulled
            neurons_size_pulled = layer_idx_to_pull_residual_neurons[-previous_layer_idx_to_be_pulled]
            residual_neurons = previous_neurons_activations[layer_activation_idx][:, :neurons_size_pulled]
            neurons_for_next_layer.append(residual_neurons)
            previous_layer_idx_to_be_pulled += 1
    input_neurons = cp.concatenate(neurons_for_next_layer, axis=-1)
    return relu((cp.dot(input_neurons, axons)) + dentrites), input_neurons

def forward_pass_activations(neurons, layers_parameters, previous_layer_pulled):
    neurons_inputs = []
    activation = cp.array(neurons)
    neurons_activations = [activation]
    total_activations = len(layers_parameters)
    for layer_idx in range(total_activations):
        axons = layers_parameters[layer_idx][0]
        dentrites = layers_parameters[layer_idx][1]
        activation, input_neurons = apply_residual_connection(neurons_activations, axons, dentrites, previous_layer_pulled)
        neurons_activations.append(activation)
        neurons_inputs.append(input_neurons)
    return neurons_inputs, neurons_activations

def aggregate_residual_stress(layers_neurons_stress, activation, axons, residual_connections_idx):
    # aggregated_neurons_stress = []
    # neurons_stress_idx_pulled = 1
    # # FROM: Output layer to Hidden layer TO: Hidden layer to Output layer
    # previous_layers_neurons_stress = layers_neurons_stress[::-1]
    # for neurons_stress_idx in range(len(layers_neurons_stress)):
    #     if neurons_stress_idx not in residual_connections_idx:
    #         if neurons_stress_idx > 0: continue
    #         residual_stress = previous_layers_neurons_stress[neurons_stress_idx]
    #         aggregated_neurons_stress.append(residual_stress)
    #     else:
    #         neurons_stress_idx_pulled = 1 if neurons_stress_idx_pulled > len(residual_connections_idx) else neurons_stress_idx_pulled
    #         neurons_stress_size_pulled = residual_connections_idx[-neurons_stress_idx_pulled]
    #         residual_stress = previous_layers_neurons_stress[neurons_stress_idx][:, :neurons_stress_size_pulled]
    #         aggregated_neurons_stress.append(residual_stress)
    #         neurons_stress_idx_pulled += 1
    #TODO: Create a function that's return aggregated neurons stress for a given layer that has a residual connection coming from previous activations
    pass

def calculate_residual_layers_stress(last_layer_neurons_stress, input_neurons, layers_parameters, residual_connections):
    layers_neurons_size = layers_parameters[1][0].shape[1]
    neurons_stress = last_layer_neurons_stress
    layers_stress = [last_layer_neurons_stress]
    total_layers_stress = len(layers_parameters)-1
    for layer_idx in range(total_layers_stress):
        activation = input_neurons[-(layer_idx+1)]
        axons = layers_parameters[-(layer_idx+1)][0]
        # Get the neurons stress for each layer
        neurons_stress = (cp.dot(neurons_stress, axons.transpose()) * relu(activation, return_derivative=True))[:, :layers_neurons_size]
    return layers_stress

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
        pre_activations_neurons, post_activations_neurons = forward_pass_activations(input_batch, layers_parameters, residual_neurons_sizes)
        avg_last_neurons_stress, neurons_stress_to_backpropagate = cross_entropy_loss(post_activations_neurons[-1], cp.array(expected_batch))
        layers_stress = calculate_residual_layers_stress(neurons_stress_to_backpropagate, pre_activations_neurons, layers_parameters, residual_neurons_sizes)
        update_layers_parameters(post_activations_neurons, layers_stress, layers_parameters, learning_rate)
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

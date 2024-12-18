import random
import cupy as cp
from features import GREEN, RED, RESET
from cupy_utils.utils import cupy_array
from nn_utils.activation_functions import tanh
from nn_utils.loss_functions import cross_entropy_loss
from cupy_utils.utils import resiudal_connections_initialization, count_parameters

def apply_residual_neurons(layer_idx, last_layer_idx, neurons_activations, axons):
    step_magnitude = 1
    step_back_size = 2**step_magnitude
    input_neurons_for_next_layer = [neurons_activations[-1]]
    while True:
        if layer_idx <= step_back_size: break
        pulled_neurons_size = neurons_activations[-(step_back_size+1)].shape[-1] // step_back_size
        pulled_neurons_activation = neurons_activations[-(step_back_size+1)][:, :pulled_neurons_size]
        input_neurons_for_next_layer.append(pulled_neurons_activation)
        step_magnitude += 1
        step_back_size = 2**step_magnitude
    no_activation_function = layer_idx == last_layer_idx
    pre_neurons_activation = cp.concatenate(input_neurons_for_next_layer, axis=-1)
    if no_activation_function: return pre_neurons_activation, cp.dot(pre_neurons_activation, axons)
    else: return pre_neurons_activation, tanh(cp.dot(pre_neurons_activation, axons))

def forward_pass_activations(neurons, layers_parameters):
    last_layer_idx = len(layers_parameters)-1
    pre_activation_neurons = cp.array(neurons)
    pre_neurons_activations = []
    post_neurons_activations = [pre_activation_neurons]
    total_activations = len(layers_parameters)
    for layer_idx in range(total_activations):
        axons = layers_parameters[layer_idx]
        pre_activation_neurons, post_activation_neurons = apply_residual_neurons(layer_idx, last_layer_idx, post_neurons_activations, axons)
        pre_neurons_activations.append(pre_activation_neurons)
        post_neurons_activations.append(post_activation_neurons)
    return pre_neurons_activations, post_neurons_activations

def reconstructed_activation_error(activation, axons):
    # 𝐲ℓ−1(i)−𝑾ℓ−1,ℓT⁢σ(𝑾ℓ−1,ℓ⁢𝐲ℓ−1(i)
    reconstructed_previous_activation = tanh(cp.dot(activation, axons.transpose()))
    # reconstructed_previous_activation = reconstructed_previous_activation
    reconstructed_activation = cp.dot(reconstructed_previous_activation, axons)
    neurons_reconstructed_error = activation - reconstructed_activation
    # 𝒥=1T⁢∑i=1T‖𝐲ℓ−1(i)−𝑾ℓ−1,ℓT⁢σ⁢(𝑾ℓ−1,ℓ⁢𝐲ℓ−1(i))‖2
    avg_reconstructed_error = cp.sum(cp.linalg.norm(neurons_reconstructed_error)**2) / activation.shape[0]
    return avg_reconstructed_error

def apply_residual_neurons_stress(layer_loss_idx, layer_stress, layers_losses, axons, pre_acitvation_neurons, post_activation_neurons_size):
    step_magnitude = 1
    step_back_size = 2**step_magnitude
    layer_stress = (cp.dot(layer_stress, axons.transpose()) * tanh(pre_acitvation_neurons, return_derivative=True))[:, :358]
    neurons_stress_to_aggregate = [layer_stress]
    while True:
        if layer_loss_idx <= step_back_size: break
        neurons_stress_size = post_activation_neurons_size // step_back_size
        residual_neurons_stress = layers_losses[-(step_back_size+1)][:, :neurons_stress_size]
        pulled_neurons_stress = cp.full_like(neurons_stress_to_aggregate[0], 0)
        pulled_neurons_stress[:, :neurons_stress_size] = residual_neurons_stress
        neurons_stress = (cp.dot(pulled_neurons_stress, axons.transpose()) * tanh(pre_acitvation_neurons, True))[:, :358]
        neurons_stress_to_aggregate.append(neurons_stress)
        step_magnitude += 1
        step_back_size = 2**step_magnitude
    aggregated_stress = cp.sum(cp.array(neurons_stress_to_aggregate), axis=0)
    if len(neurons_stress_to_aggregate) > 1: return layer_stress, aggregated_stress
    else: return layer_stress, layer_stress

def calculate_residual_layers_stress(last_layer_neurons_stress, pre_activations_neurons, post_activations_neurons, layers_parameters):
    activation_reconstructed_stress = []
    layer_stress = last_layer_neurons_stress
    layers_stress = [last_layer_neurons_stress]
    total_layers_stress = len(layers_parameters)
    for layer_idx in range(total_layers_stress):
        axons = layers_parameters[-(layer_idx+1)]
        pre_activation_neurons = pre_activations_neurons[-(layer_idx+1)]
        post_activation_neurons = post_activations_neurons[-(layer_idx+1)]
        layer_stress, aggregated_layer_stress = apply_residual_neurons_stress(layer_idx, layer_stress, layers_stress, axons, pre_activation_neurons, post_activation_neurons.shape[-1])
        reconstructed_activation_avg_stress = reconstructed_activation_error(post_activation_neurons, axons)
        layers_stress.append(aggregated_layer_stress)
        activation_reconstructed_stress.append(reconstructed_activation_avg_stress)
    return layers_stress, cp.mean(cp.array(activation_reconstructed_stress))

def oja_rule_update(previous_activation, current_activation, axons, learning_rate=0.001):
    rule_1 = cp.dot(cp.dot(current_activation.transpose(), current_activation), axons.transpose()).transpose()
    rule_2 = cp.dot(previous_activation.transpose(), current_activation)
    return (learning_rate * (rule_2 - rule_1)) / current_activation.shape[0]

def backpropagation_rule_update(previous_activation, loss, learning_rate):
    return (learning_rate * cp.dot(previous_activation.transpose(), loss)) / previous_activation.shape[0]

def update_layers_parameters(pre_activations_neurons, post_activations_neurons, layers_losses, layers_parameters, learning_rate):
    total_parameters = len(layers_parameters)
    for layer_idx in range(total_parameters):
        axons = layers_parameters[-(layer_idx+1)]
        current_activation = post_activations_neurons[-(layer_idx+1)]
        previous_activation = pre_activations_neurons[-(layer_idx+1)]
        loss = layers_losses[layer_idx]
        backprop_parameters_nudge = backpropagation_rule_update(previous_activation, loss, learning_rate)
        oja_parameters_nudge = oja_rule_update(previous_activation, current_activation, axons)
        axons -= backprop_parameters_nudge
        axons += oja_parameters_nudge

def residual_training_layers(dataloader, layers_parameters, learning_rate):
    per_batch_stress = []
    for i, (input_batch, expected_batch) in enumerate(dataloader):
        pre_activations_neurons, post_activations_neurons = forward_pass_activations(input_batch, layers_parameters)
        avg_last_neurons_stress, neurons_stress_to_backpropagate = cross_entropy_loss(post_activations_neurons[-1], cp.array(expected_batch))
        layers_stress, activation_reconstructed_stress = calculate_residual_layers_stress(neurons_stress_to_backpropagate, pre_activations_neurons, post_activations_neurons, layers_parameters)
        update_layers_parameters(pre_activations_neurons, post_activations_neurons, layers_stress, layers_parameters, learning_rate)
        print(f"Loss each batch {i+1}: {avg_last_neurons_stress} Reconstructed activation error: {activation_reconstructed_stress}\r", end="", flush=True)
        per_batch_stress.append(avg_last_neurons_stress)
        if i == 1000:
            break
    return cp.mean(cp.array(per_batch_stress))

def residual_test_layers(dataloader, layers_parameters):
    correct_predictions = []
    wrong_predictions = []
    model_predictions = []
    for i, (input_image_batch, expected_batch) in enumerate(dataloader):
        expected_batch = cupy_array(expected_batch)
        model_output = forward_pass_activations(input_image_batch, layers_parameters)[-1][-1]
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

def model(network_architecture, training_loader, validation_loader, learning_rate, epochs):
    network_parameters = resiudal_connections_initialization(network_architecture)
    # print(count_parameters(network_parameters))
    for epoch in range(epochs):
        print(f'EPOCH: {epoch+1}')
        model_stress = residual_training_layers(dataloader=training_loader, layers_parameters=network_parameters, learning_rate=learning_rate)
        model_accuracy = residual_test_layers(dataloader=validation_loader, layers_parameters=network_parameters)
        # print(f'accuracy: {model_accuracy}')
        print(f'Average loss per epoch: {model_stress} accuracy: {model_accuracy}')

import random
import cupy as cp
from features import GREEN, RED, RESET
from cupy_utils.utils import cupy_array
from nn_utils.activation_functions import relu
from nn_utils.loss_functions import cross_entropy_loss
from cupy_utils.utils import axons_and_dentrites_initialization

def forward_pass_activations(input_feature, layers_parameters):
    neurons_activation = cp.array(input_feature)
    total_activations = len(layers_parameters)
    neurons_activations = [neurons_activation]
    for layer_idx in range(total_activations):
        axons = layers_parameters[layer_idx][0]
        neurons = cp.dot(neurons_activation, axons)
        neurons_activation = neurons if layer_idx == total_activations-1 else relu(neurons)
        neurons_activations.append(neurons)
    return neurons_activations

def reconstructed_activation_error(activation, axons):
    # 𝐲ℓ−1(i)−𝑾ℓ−1,ℓT⁢σ(𝑾ℓ−1,ℓ⁢𝐲ℓ−1(i)
    reconstructed_previous_activation = relu(cp.dot(activation, axons.transpose()))
    reconstructed_activation = cp.dot(reconstructed_previous_activation, axons)
    neurons_reconstructed_error = activation - reconstructed_activation
    # 𝒥=1T⁢∑i=1T‖𝐲ℓ−1(i)−𝑾ℓ−1,ℓT⁢σ⁢(𝑾ℓ−1,ℓ⁢𝐲ℓ−1(i))‖2
    avg_reconstructed_error = cp.sum(cp.linalg.norm(neurons_reconstructed_error)**2) / activation.shape[0]
    return avg_reconstructed_error

def calculate_layers_stress(neurons_stress, layers_activations, layers_parameters):
    reconstructed_errors = []
    layers_gradient = []
    total_layers_stress = len(layers_activations)-1
    for each_layer in range(total_layers_stress):
        axons = layers_parameters[-(each_layer+1)][0]
        activation = layers_activations[-(each_layer+1)]
        previous_activation = layers_activations[-(each_layer+2)]
        avg_error = reconstructed_activation_error(activation, axons)
        layer_gradient = neurons_stress 
        neurons_stress = (cp.dot(neurons_stress, axons.transpose())) * (relu(previous_activation, True))
        layers_gradient.append(layer_gradient)
        reconstructed_errors.append(avg_error)
    return layers_gradient, cp.mean(cp.array(reconstructed_errors))

def oja_rule_update(previous_activation, current_activation, axons):
    rule_1 = cp.dot(cp.dot(current_activation.transpose(), current_activation), axons.transpose()).transpose()
    rule_2 = cp.dot(previous_activation.transpose(), current_activation)
    return rule_1 - rule_2

def update_layers_parameters(neurons_activations, layers_losses, layers_parameters, learning_rate):
    total_parameters = len(layers_losses)
    for layer_idx in range(total_parameters):
        axons = layers_parameters[-(layer_idx+1)][0]
        current_activation = neurons_activations[-(layer_idx+1)]
        previous_activation = neurons_activations[-(layer_idx+2)]
        loss = layers_losses[layer_idx]
        backprop_parameters_nudge = learning_rate * cp.dot(previous_activation.transpose(), loss)
        oja_parameters_nudge = 0.01 * oja_rule_update(previous_activation, current_activation, axons)
        axons -= (backprop_parameters_nudge / current_activation.shape[0])
        axons += (oja_parameters_nudge / current_activation.shape[0])

def training_layers(dataloader, layers_parameters, learning_rate):
    per_batch_stress = []
    for i, (input_batch, expected_batch) in enumerate(dataloader):
        neurons_activations = forward_pass_activations(input_batch, layers_parameters)
        avg_last_neurons_stress, neurons_stress_to_backpropagate = cross_entropy_loss(neurons_activations[-1], cp.array(expected_batch))
        backprop_and_oja_combine_layers_stress, reconstructed_error_avg = calculate_layers_stress(neurons_stress_to_backpropagate, neurons_activations, layers_parameters)
        update_layers_parameters(neurons_activations, backprop_and_oja_combine_layers_stress, layers_parameters, learning_rate)
        print(f"Loss each batch {i+1}: {avg_last_neurons_stress} Reconstruct activation error: {reconstructed_error_avg}\r", end="", flush=True)
        per_batch_stress.append(avg_last_neurons_stress)
    return cp.mean(cp.array(per_batch_stress))

def test_layers(dataloader, layers_parameters):
    correct_predictions = []
    wrong_predictions = []
    model_predictions = []
    for i, (input_image_batch, expected_batch) in enumerate(dataloader):
        expected_batch = cupy_array(expected_batch)
        model_output = forward_pass_activations(input_image_batch, layers_parameters)[-1]
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
    network_parameters = [axons_and_dentrites_initialization(network_architecture[feature_idx], network_architecture[feature_idx+1]) for feature_idx in range(len(network_architecture)-1)]
    
    for epoch in range(epochs):
        print(f'EPOCH: {epoch+1}')
        model_stress = training_layers(dataloader=training_loader, layers_parameters=network_parameters, learning_rate=learning_rate)
        model_accuracy = test_layers(dataloader=validation_loader, layers_parameters=network_parameters)
        # print(f'accuracy: {model_accuracy}')
        print(f'Average loss per epoch: {model_stress} accuracy: {model_accuracy}')

import random
import cupy as cp
from features import GREEN, RED, RESET
from cupy_utils.utils import cupy_array
from nn_utils.loss_functions import cross_entropy_loss
from cupy_utils.utils import axons_and_dentrites_initialization

def hebbian_plasticity(previous_activation, current_activation, axons):
    oja_parameters_nudge = 0.01 * (cp.dot(previous_activation.transpose(), current_activation) - cp.dot(cp.dot(current_activation.transpose(), current_activation), axons.transpose()).transpose())
    axons += oja_parameters_nudge / current_activation.shape[0]

def calculate_activation_error(current_activation, axons):
    reconstruct_previous_activation = cp.dot(current_activation, axons.transpose())
    reconctruct_current_activation = cp.dot(reconstruct_previous_activation, axons)
    activation_error = current_activation - reconctruct_current_activation
    avg_reconstructed_error = cp.sum(cp.linalg.norm(activation_error)**2) / current_activation.shape[0]
    return avg_reconstructed_error

def forward_once_for_next_layer(dataloader, axons):
    new_dataloader = []
    for input_neurons, expected in dataloader:
        input_neurons = cp.array(input_neurons)
        layer_activation = cp.dot(input_neurons, axons)
        new_dataloader.append((layer_activation, expected))
    return new_dataloader

def train_each_layer(dataloader, layer_idx, axons, learning_rate):
    for each in range(1):
        per_batch_stress = []
        for i, (input_neurons, _) in enumerate(dataloader):
            input_neurons = cp.array(input_neurons)
            neurons_activation = cp.dot(input_neurons, axons)
            activation_stress = calculate_activation_error(neurons_activation, axons)
            hebbian_plasticity(input_neurons, neurons_activation, axons)
            per_batch_stress.append(activation_stress)
        # print(f'Layer: {layer_idx+1} avg loss per epoch: {cp.mean(cp.array(per_batch_stress))}')
    return forward_once_for_next_layer(dataloader, axons)

def training_middle_layers(dataloader, network_parameters, learning_rate):
    total_layers = len(network_parameters)
    for each_layer in range(total_layers):
        layer_axons = network_parameters[each_layer][0]
        if each_layer == total_layers-1:
            break
        dataloader = train_each_layer(dataloader, each_layer, layer_axons, learning_rate)

def forward_in_layers(neurons_activation, model_parameters):
    neurons_activations = []
    for each in range(len(model_parameters)):
        axons = model_parameters[each][0]
        neurons_activation = cp.dot(neurons_activation, axons)
        neurons_activations.append(neurons_activation)
    return neurons_activations

def backpropagation(neurons_stress, neurons_activation, axons):
    previous_activation = neurons_activation[-2]
    axons -= 0.001 * (cp.dot(previous_activation.transpose(), neurons_stress) / neurons_stress[0].shape[0])

def training_last_layer(dataloader, model_parameters, learning_rate, epochs):
    per_batch_stress = []
    for input_batch, expected_batch in dataloader:
        neurons_activations = forward_in_layers(cp.array(input_batch), model_parameters)
        avg_stress, neurons_stress = cross_entropy_loss(neurons_activations[-1], cp.array(expected_batch))
        backpropagation(neurons_stress, neurons_activations, model_parameters[-1][0])
        per_batch_stress.append(avg_stress)
    return cp.mean(cp.array(per_batch_stress))

def test_layers(dataloader, layers_parameters):
    correct_predictions = []
    wrong_predictions = []
    model_predictions = []
    for i, (input_image_batch, expected_batch) in enumerate(dataloader):
        expected_batch = cupy_array(expected_batch)
        model_output = forward_in_layers(cp.array(input_image_batch), layers_parameters)[-1]
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

def custom_model_v1(network_architecture, training_dataloader, validation_dataloader, learning_rate, epochs):
    model_parameters = [axons_and_dentrites_initialization(network_architecture[feature_idx], network_architecture[feature_idx+1]) for feature_idx in range(len(network_architecture)-1)]
    training_middle_layers(training_dataloader, model_parameters, learning_rate)
    for _ in range(10):
        avg_loss = training_last_layer(training_dataloader, model_parameters, learning_rate, 50)
        accuracy = test_layers(validation_dataloader, model_parameters)
        print(f"Loss: {avg_loss} Accuracy: {accuracy}")
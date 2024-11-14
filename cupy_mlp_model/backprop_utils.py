import random
import cupy as cp
from features import GREEN, RED, RESET
from cupy_utils.utils import cupy_array
from nn_utils.activation_functions import relu
from nn_utils.loss_functions import cross_entropy_loss

def forward_pass_activations(input_feature, layers_parameters):
    neurons = cp.array(input_feature)
    total_activations = len(layers_parameters)
    neurons_activations = [neurons]
    for layer_idx in range(total_activations):
        axons = layers_parameters[layer_idx][0]
        dentrites = layers_parameters[layer_idx][1]
        neurons = relu((cp.dot(neurons, axons)) + dentrites) 
        neurons_activations.append(neurons)
    return neurons_activations

def calculate_layers_stress(neurons_stress, neurons_activations, layers_parameters):
    layers_gradient = [neurons_stress]
    total_layers_stress = len(layers_parameters)-1
    for each_layer in range(total_layers_stress):
        activation = neurons_activations[-(each_layer+2)]
        axons = layers_parameters[-(each_layer+1)][0]
        neurons_stress = cp.dot(neurons_stress, axons.transpose()) * relu(input_data=activation, return_derivative=True)
        layers_gradient.append(neurons_stress)
    return layers_gradient

def update_layers_parameters(neurons_activations, layers_losses, layers_parameters, learning_rate):
    total_parameters = len(layers_losses)
    for layer_idx in range(total_parameters):
        axons = layers_parameters[-(layer_idx+1)][0]
        dentrites = layers_parameters[-(layer_idx+1)][1]
        current_activation = neurons_activations[-(layer_idx+1)]
        previous_activation = neurons_activations[-(layer_idx+2)]
        loss = layers_losses[layer_idx]
        backprop_parameters_nudge = learning_rate * cp.dot(previous_activation.transpose(), loss)
        axons -= (backprop_parameters_nudge / current_activation.shape[0])
        dentrites -= ((learning_rate * cp.sum(loss, axis=0)) / current_activation.shape[0])

def training_layers(dataloader, layers_parameters, learning_rate):
    per_batch_stress = []
    for i, (input_batch, expected_batch) in enumerate(dataloader):
        neurons_activations = forward_pass_activations(input_batch, layers_parameters)
        avg_last_neurons_stress, neurons_stress_to_backpropagate = cross_entropy_loss(neurons_activations[-1], cp.array(expected_batch))
        layers_stress = calculate_layers_stress(neurons_stress_to_backpropagate, neurons_activations, layers_parameters)
        update_layers_parameters(neurons_activations, layers_stress, layers_parameters, learning_rate)
        print(f"Loss each batch {i+1}: {avg_last_neurons_stress}\r", end="", flush=True)
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

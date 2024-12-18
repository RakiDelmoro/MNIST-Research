import random
import cupy as cp
from features import GREEN, RED, RESET
from cupy_utils.utils import cupy_array
from nn_utils.activation_functions import relu, softmax, log_softmax
from nn_utils.loss_functions import cross_entropy_loss
from cupy_utils.utils import cupy_axon_and_dentrites_init

def forward_pass_activations(input_neurons, layers_parameters):
    first_layer_output = cp.matmul(cp.array(input_neurons), layers_parameters[0][0])
    first_layer_activated = relu(first_layer_output)
    second_layer_output = cp.matmul(first_layer_activated, layers_parameters[-1][0])
    model_output = log_softmax(second_layer_output)
    return input_neurons, first_layer_output, first_layer_activated, second_layer_output, model_output

def network_loss_function(network_output, expected_output):
    # Compute probability distribution of model output
    correct_index_mask = cp.zeros(network_output.shape)
    correct_index_mask[cp.arange(network_output.shape[0]), cp.array(expected_output)] = 1.0
    mean_loss = (-correct_index_mask * network_output).mean(axis=1)
    network_output_stress = -correct_index_mask/network_output.shape[0]
    # LogSoftmax derivative
    stress_propagate_back_to_network = network_output_stress - cp.exp(network_output)*network_output_stress.sum(axis=1).reshape(-1, 1)
    return mean_loss, stress_propagate_back_to_network

def calculate_layers_stress(network_activations, network_stress_propagated, layers_parameters):
    second_layer_out_stress = cp.matmul(network_stress_propagated, layers_parameters[-1][0].transpose())
    first_layer_out_activated_stress = relu(network_activations[2], return_derivative=True) * second_layer_out_stress
    return network_stress_propagated, first_layer_out_activated_stress

def update_layers_parameters(network_activations, layers_stresses, layers_parameters, learning_rate):
    # Layer 2 axons update
    first_layer_activated = network_activations[2]
    second_layer_stress = layers_stresses[0]
    layers_parameters[-1][0] -= ((learning_rate * cp.matmul(first_layer_activated.transpose(), second_layer_stress)) /network_activations[0].shape[0])
    # Layer 1 axons update
    input_neurons = network_activations[0]
    first_layer_stress = layers_stresses[-1]
    layers_parameters[0][0] -= ((learning_rate * cp.matmul(cp.array(input_neurons).transpose(), first_layer_stress))/network_activations[0].shape[0])

def training_layers(dataloader, layers_parameters, learning_rate):
    per_batch_stress = []
    for i, (input_batch, expected_batch) in enumerate(dataloader):
        neurons_activations = forward_pass_activations(input_batch, layers_parameters)
        mean_stress, neurons_stress = network_loss_function(neurons_activations[-2], expected_batch)
        network_neurons_stresses = calculate_layers_stress(neurons_activations, neurons_stress, layers_parameters)
        update_layers_parameters(neurons_activations, network_neurons_stresses, layers_parameters, learning_rate)
        print(f"Loss each batch {i+1}: {mean_stress.mean()}\r", end="", flush=True)
        per_batch_stress.append(mean_stress.mean())
    return cp.mean(cp.array(per_batch_stress))

def test_layers(dataloader, layers_parameters):
    correct_predictions = []
    wrong_predictions = []
    model_predictions = []
    for i, (input_image_batch, expected_batch) in enumerate(dataloader):
        expected_batch = cupy_array(expected_batch)
        model_output = forward_pass_activations(input_image_batch, layers_parameters)[-2]
        batched_accuracy = cp.array(expected_batch == (model_output.argmax(-1))).mean()
        for each in range(100):
            if model_output[each].argmax(-1).item() == expected_batch[each].item():
                correct_predictions.append((model_output[each].argmax(-1).item(), expected_batch[each].item()))
            else:
                wrong_predictions.append((model_output[each].argmax(-1).item(), expected_batch[each].item()))
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
    network_parameters = [cupy_axon_and_dentrites_init(network_architecture[feature_idx], network_architecture[feature_idx+1]) for feature_idx in range(len(network_architecture)-1)]
    for epoch in range(epochs):
        print(f'EPOCH: {epoch+1}')
        model_stress = training_layers(dataloader=training_loader, layers_parameters=network_parameters, learning_rate=learning_rate)
        model_accuracy = test_layers(dataloader=validation_loader, layers_parameters=network_parameters)
        # print(f'accuracy: {model_accuracy}')
        print(f'Average loss per epoch: {model_stress} accuracy: {model_accuracy}')

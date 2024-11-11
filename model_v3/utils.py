import cupy as cp
from cupy_utils.utils import cupy_array
from features import GREEN, RED, RESET
from nn_utils.loss_functions import cross_entropy_loss
from nn_utils.activation_functions import leaky_relu, softmax
from cupy_utils.utils import axons_initialization, dentrites_initialization

def network_axons_and_dentrites(model_feature_sizes):
    layers_axons_and_dentrites = []
    for connection_idx in range(len(model_feature_sizes)-1):
        # axons, dentrites = axon_and_dentrites_initialization(model_feature_sizes[connection_idx], model_feature_sizes[connection_idx+1])
        axons = axons_initialization(model_feature_sizes[connection_idx], model_feature_sizes[connection_idx+1])
        dentrites = dentrites_initialization(model_feature_sizes[connection_idx+1])
        layers_axons_and_dentrites.append([axons, dentrites])
    return layers_axons_and_dentrites

def get_forward_activations(input_neurons, layers_parameters):
    neurons = cp.array(input_neurons)
    total_connections = len(layers_parameters)
    neurons_activations = [neurons]
    for each_connection in range(total_connections):
        axons = layers_parameters[each_connection][0]
        dentrites = layers_parameters[each_connection][1]
        neurons = (cp.dot(neurons, axons)) + dentrites
        neurons_activations.append(neurons)
    return neurons_activations

def get_backward_activations(input_neurons, layers_parameters):
    neurons = cp.array(input_neurons)
    total_connections = len(layers_parameters)-1
    neurons_activations = [neurons]
    for each_connection in range(total_connections):
        axons = layers_parameters[-(each_connection+1)][0].transpose()
        dentrites = layers_parameters[-(each_connection+2)][1]
        neurons = (cp.dot(neurons, axons)) + dentrites
        neurons_activations.append(neurons)
    return neurons_activations

def training_layers(dataloader, parameters, learning_rate):
    loss_per_batch = []
    for i, (input_batch, expected_batch) in enumerate(dataloader):
        forward_activations = get_forward_activations(input_batch, parameters)
        backward_activations = get_backward_activations(expected_batch, parameters)
        network_stress, loss = calculate_network_stress(forward_activations, backward_activations)
        loss_per_batch.append(loss)
        print(f"Loss each batch {i+1}: {loss}\r", end="", flush=True)
        update_parameters(network_stress, forward_activations, parameters, learning_rate)
    return cp.mean(cp.array(loss_per_batch))

def calculate_network_stress(forward_activations, backward_activations):
    layers_stress = []
    average_loss = []
    total_activations = len(backward_activations)
    for each_activation in range(total_activations):
        stress = (forward_activations[-(each_activation+1)] - backward_activations[each_activation])
        average_loss.append(cp.mean(stress))
        layers_stress.append(stress)
    loss = cp.sum(cp.array(average_loss))**2
    return layers_stress, loss

def update_parameters(network_stress, forward_activations, parameters, learning_rate):
    total_parameters = len(parameters)
    for layer_idx in range(total_parameters):
        layer_axons, layer_dentrites = parameters[-(layer_idx+1)]
        neurons_activation = forward_activations[-(layer_idx+2)]
        stress = network_stress[layer_idx]

        layer_axons -= (learning_rate * cp.dot(neurons_activation.transpose(), stress) / neurons_activation.shape[0])
        layer_dentrites -= (learning_rate * cp.sum(stress, axis=0) / neurons_activation.shape[0])

def test_layers(dataloader, parameters):
    correct_predictions = []
    wrong_predictions = []
    model_predictions = []
    for i, (input_image, expected) in enumerate(dataloader):
        expected_sample = cupy_array(expected)
        model_activations = get_forward_activations(input_image, parameters)
        model_prediction = cp.array(expected_sample.argmax(-1) == model_activations[-1].argmax(-1)).astype(cp.float16).item()
        model_predictions.append(model_prediction)
        if model_activations[-1].argmax(-1) == expected_sample.argmax(-1):
            correct_predictions.append((model_activations[-1].argmax(-1).item(), expected_sample.argmax(-1).item()))
        else:
            wrong_predictions.append((model_activations[-1].argmax(-1).item(), expected_sample.argmax(-1).item()))
        if i > 100:
            break
    print(f"{GREEN}MODEL Correct Predictions{RESET}")
    [print(f"Digit Image is: {GREEN}{expected}{RESET} Model Prediction: {GREEN}{prediction}{RESET}") for i, (prediction, expected) in enumerate(correct_predictions) if i < 10]
    print(f"{RED}MODEL Wrong Predictions{RESET}")
    [print(f"Digit Image is: {RED}{expected}{RESET} Model Prediction: {RED}{prediction}{RESET}") for i, (prediction, expected) in enumerate(correct_predictions) if i < 10]
    return cp.mean(cp.array(model_predictions)).item()

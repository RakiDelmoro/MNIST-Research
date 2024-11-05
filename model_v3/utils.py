import cupy as cp
from cupy_utils.utils import one_hot
from features import GREEN, RED, RESET
from nn_utils.activation_functions import leaky_relu, softmax
from nn_utils.loss_functions import cross_entropy_loss
from cupy_utils.utils import axons_initialization, dentrites_initialization


    # oja_term = theta2 * (cp.dot(current_output.transpose(), neurons_activation) - cp.dot(cp.dot(current_output.transpose(), current_output), layer_axons)
        

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
    for input_batch, expected_batch in dataloader:
        forward_activations = get_forward_activations(input_batch, parameters)
        backward_activations = get_backward_activations(expected_batch, parameters)
        network_stress, loss = calculate_network_stress(forward_activations, backward_activations)
        loss_per_batch.append(loss)
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
        stress = (network_stress[layer_idx]) / neurons_activation.shape[0]

        layer_axons -= learning_rate * cp.dot(neurons_activation.transpose(), stress)
        layer_dentrites -= learning_rate * cp.sum(stress, axis=0)


def test_layers(dataloader, parameters):
    per_batch_accuracy = []
    wrong_samples_indices = []
    correct_samples_indices = []
    model_predictions = []
    expected_model_prediction = []
    for input_image_batch, expected_batch in dataloader:
        expected_batch = cp.array(expected_batch)
        # get the last layer activation
        model_output = get_forward_activations(input_image_batch, parameters)[-1]
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
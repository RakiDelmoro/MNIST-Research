import torch
import cupy as cp
from features import GREEN, RED, RESET
from nn_utils.activation_functions import leaky_relu, softmax
from cupy_utils.utils import axons_and_dentrites_initialization, one_hot

def neural_network(network_architecture: list):
    forward_layers_axons = []
    forward_layers_dentrites = []
    for feature_size_idx in range(len(network_architecture)-1):
        forward_axons, forward_dentrites = axons_and_dentrites_initialization(network_architecture[feature_size_idx], network_architecture[feature_size_idx+1])
        forward_layers_axons.append(forward_axons)
        forward_layers_dentrites.append(forward_dentrites)

    backward_layers_axons = []
    backward_layers_dentrites = []
    reverse_network_architecture = network_architecture[::-1]
    for feature_size_idx in range(len(reverse_network_architecture)-1):
        backward_axons, backward_dentrites = axons_and_dentrites_initialization(reverse_network_architecture[feature_size_idx], reverse_network_architecture[feature_size_idx+1])
        backward_layers_axons.append(backward_axons)
        backward_layers_dentrites.append(backward_dentrites)

    def forward_in_neurons(neurons):
        neurons = cp.array(neurons.reshape(neurons.shape[0], -1))
        neurons_activations = [neurons]
        for neurons_layer_idx in range(len(network_architecture)-1):
            forward_last_layer = len(network_architecture)-2 == neurons_layer_idx
            if not forward_last_layer:
                neurons = leaky_relu(cp.dot(neurons, forward_layers_axons[neurons_layer_idx]) + forward_layers_dentrites[neurons_layer_idx])
            else:
                neurons = softmax(cp.dot(neurons, forward_layers_axons[neurons_layer_idx]) + forward_layers_dentrites[neurons_layer_idx])
            neurons_activations.append(neurons)
        return neurons_activations

    def backward_in_neurons(neurons):
        neurons = one_hot(cp.array(neurons), number_of_classes=10)
        neurons_activations = [neurons]
        for neurons_layer_idx in range(len(network_architecture)-1):
            backward_last_layer = len(reverse_network_architecture)-2 == neurons_layer_idx
            if not backward_last_layer:
                neurons = leaky_relu(cp.dot(neurons, backward_layers_axons[neurons_layer_idx]) + backward_dentrites[neurons_layer_idx])
            else:
                neurons = softmax(cp.dot(neurons, backward_layers_axons[neurons_layer_idx]) + backward_dentrites[neurons_layer_idx])
            neurons_activations.append(neurons)
        return neurons_activations

    def calculate_network_stress(forward_activations, backward_activations, loss_function):
        loss_func = torch.nn.MSELoss(reduction='none')
        layers_neurons_stress = []
        each_layer_stress = []
        for each_layer_neurons_activation in range(len(network_architecture)):
            neurons_stress = loss_func(torch.tensor(forward_activations[each_layer_neurons_activation]), torch.tensor(backward_activations[-(each_layer_neurons_activation+1)]))
            average_neurons_stress = loss_function(torch.tensor(forward_activations[each_layer_neurons_activation]), torch.tensor(backward_activations[-(each_layer_neurons_activation+1)]))
            layers_neurons_stress.append(cp.array(neurons_stress))
            each_layer_stress.append(average_neurons_stress)
        return layers_neurons_stress, cp.mean(cp.array(each_layer_stress)).item()

    def update_axons_and_dentrites_forward_layers(layers_stress, neurons_activations, learning_rate):
        for layer_neurons_idx, (layer_axons, layer_dentrites) in enumerate(zip(forward_layers_axons, forward_layers_dentrites)):
            layer_neurons_activation = neurons_activations[layer_neurons_idx]
            layer_stress = layers_stress[layer_neurons_idx+1]

            layer_axons -= learning_rate * cp.dot(layer_neurons_activation.transpose(), layer_stress)
            layer_dentrites -= learning_rate * cp.sum(layer_stress, axis=0)

    def update_axons_and_dentrites_backward_layers(layers_stress, neurons_activations, learning_rate):
        for layer_neurons_idx, (layer_axons, layer_dentrites) in enumerate(zip(backward_layers_axons, backward_layers_dentrites)):
            layer_neurons_activation = neurons_activations[layer_neurons_idx]
            layer_stress = layers_stress[-(layer_neurons_idx+2)]

            layer_axons += learning_rate * cp.dot(layer_neurons_activation.transpose(), layer_stress)
            layer_dentrites += learning_rate * cp.sum(layer_stress, axis=0)

    def training_run(training_loader, loss_function, learning_rate):
        per_batch_stress = []
        for input_batch, expected_batch in training_loader:
            forward_activations = forward_in_neurons(input_batch)
            backward_activations = backward_in_neurons(expected_batch)
            neurons_stress, average_layer_stress = calculate_network_stress(forward_activations, backward_activations, loss_function)
            # Update bidirectional passes parameters
            update_axons_and_dentrites_forward_layers(neurons_stress, forward_activations, learning_rate)
            update_axons_and_dentrites_backward_layers(neurons_stress, backward_activations, learning_rate)
            per_batch_stress.append(average_layer_stress)

    def test_run(dataloader):
        per_batch_accuracy = []
        wrong_samples_indices = []
        correct_samples_indices = []
        model_predictions = []
        expected_model_prediction = []
        for input_image_batch, expected_batch in dataloader:
            expected_batch = one_hot(expected_batch, number_of_classes=10)
            model_output = forward_in_neurons(input_image_batch)[-1]
            batch_accuracy = cp.array(expected_batch.argmax(-1) == (model_output).argmax(-1)).mean()
            correct_indices_in_a_batch = cp.where(expected_batch.argmax(-1) == model_output.argmax(-1))[0]
            wrong_indices_in_a_batch = cp.where(~(expected_batch.argmax(-1) == model_output.argmax(-1)))[0]

            per_batch_accuracy.append(batch_accuracy.item())
            correct_samples_indices.append(correct_indices_in_a_batch)
            wrong_samples_indices.append(wrong_indices_in_a_batch)
            model_predictions.append(model_output.argmax(-1))
            expected_model_prediction.append(expected_batch.argmax(-1))

        correct_samples = cp.concatenate(correct_samples_indices)[list(range(0,len(correct_samples_indices)))]
        wrong_samples = cp.concatenate(wrong_samples_indices)[list(range(0,len(wrong_samples_indices)))]
        model_prediction = cp.concatenate(model_predictions)
        model_expected_prediction = cp.concatenate(expected_model_prediction)

        print(f"{GREEN}Model Correct Predictions{RESET}")
        for indices in correct_samples: print(f"Digit Image is: {GREEN}{model_expected_prediction[indices]}{RESET} Model Prediction: {GREEN}{model_prediction[indices]}{RESET}")
        print(f"{RED}Model Wrong Predictions{RESET}")
        for indices in wrong_samples: print(f"Digit Image is: {RED}{model_expected_prediction[indices]}{RESET} Model Predictions: {RED}{model_prediction[indices]}{RESET}")

        return cp.mean(cp.array(per_batch_accuracy)).item()
    
    def runner(epochs, training_loader, validation_loader, loss_function, learning_rate):
        for epoch in range(epochs):
            # Training
            training_loss = training_run(training_loader, loss_function, learning_rate)
            # Test
            accuracy = test_run(validation_loader)
            print(f"EPOCH: {epoch+1} Training Loss: {training_loss} Model Accuracy: {accuracy}")

    return runner

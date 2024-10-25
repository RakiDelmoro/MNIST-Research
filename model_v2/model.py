import torch
import cupy as cp
from cupy_utils.utils import one_hot
from features import GREEN, RED, RESET
from nn_utils.loss_functions import cross_entropy_loss
from nn_utils.activation_functions import leaky_relu, softmax
from model_v2.utils import forward_pass_architecture, backward_pass_architecture, get_non_zero_value

def neural_network(network_architecture: list):
    #TODO: Create a function for network architecture layers
    # Return Tuple of axons and dentrites
    # axons is an array shape -> (total connections, filled value until to a certain feature size, filled value until to a certain feature size)
    # dentrites is an array shape -> (total connection, filled value until to a certain feature size)
    forward_pass_axons_and_dentrites = forward_pass_architecture(network_features_size=network_architecture)
    backward_pass_axons_and_dentrites = backward_pass_architecture(network_features_size=network_architecture[::-1])

    def forward_in_neurons(neurons):
        neurons = cp.array(neurons)
        # Shape -> (Total activations, batch_size, maximum value in network_architecture)
        neurons_activations = cp.zeros(shape=(len(network_architecture),  neurons.shape[0], max(network_architecture)))
        neurons_activations[0, :, :neurons.shape[-1]] = neurons
        for neurons_layer_idx in range(len(network_architecture)-1):
            axons = get_non_zero_value(forward_pass_axons_and_dentrites[0][neurons_layer_idx])
            neurons = cp.dot(neurons, axons)
            neurons_activations[neurons_layer_idx+1, :, :neurons.shape[-1]] = neurons
        return neurons_activations

    def backward_in_neurons(neurons):
        neurons = cp.array(neurons)
        neurons_activations = cp.zeros(shape=(len(network_architecture), neurons.shape[0], max(network_architecture)))
        neurons_activations[0, :, :neurons.shape[-1]] = neurons
        for neurons_layer_idx in range(len(network_architecture)-1):
            axons = get_non_zero_value(backward_pass_axons_and_dentrites[0][neurons_layer_idx])
            neurons = cp.dot(neurons, axons)
            neurons_activations[neurons_layer_idx+1, :, :neurons.shape[-1]] = neurons
        return neurons_activations

    def calculate_network_stress(forward_activations, backward_activations):
        #TODO: Do not derive average here do it in print statement
        layers_neurons_stress = []
        each_layer_avg_stress = []
        for activation_idx in range(len(network_architecture)):
            # Calculate stress start from forward pass output and backward pass input
            neurons_stress = forward_activations[-(activation_idx+1)] - backward_activations[activation_idx]
            layers_neurons_stress.append(neurons_stress)
            each_layer_avg_stress.append(cp.mean(neurons_stress))
        return layers_neurons_stress, cp.mean(cp.array(each_layer_avg_stress)).item()

    def update_axons_and_dentrites_forward_layers(layers_stress, neurons_activations, learning_rate):
        for layer_neurons_idx in range(len(forward_pass_axons_and_dentrites)):
            layer_neurons_activation = neurons_activations[-(layer_neurons_idx+2)]
            layer_stress = layers_stress[layer_neurons_idx]

            forward_pass_axons_and_dentrites[-(layer_neurons_idx+1)] -= learning_rate * cp.dot(layer_neurons_activation.transpose(), layer_stress)

    def update_axons_and_dentrites_backward_layers(layers_stress, neurons_activations, learning_rate):
        for layer_neurons_idx in range(len(backward_layers_axons)): #TODO: Fix
            # layer_axons = 
            layer_neurons_activation = neurons_activations[-(layer_neurons_idx+2)]
            layer_stress = layers_stress[-(layer_neurons_idx+1)]

            backward_layers_axons[-(layer_neurons_idx+1)] += learning_rate * cp.dot(layer_neurons_activation.transpose(), layer_stress) #TODO: FIx
        # for layer_neurons_idx, (layer_axons, layer_dentrites) in enumerate(zip(backward_layers_axons[::-1], backward_layers_dentrites)):
        #     layer_neurons_activation = neurons_activations[-(layer_neurons_idx+2)]
        #     layer_stress = layers_stress[-(layer_neurons_idx+1)]
            # layer_dentrites += learning_rate * cp.sum(layer_stress, axis=0)

    def training_run(training_loader, loss_function, learning_rate):
        per_batch_stress = []
        for input_batch, expected_batch in training_loader:
            forward_activations = forward_in_neurons(input_batch)
            backward_activations = backward_in_neurons(expected_batch)
            neurons_stress, average_layer_stress = calculate_network_stress(forward_activations, backward_activations, loss_function)
            print(average_layer_stress)
            # Update bidirectional passes parameters
            update_axons_and_dentrites_forward_layers(neurons_stress, forward_activations, learning_rate)
            update_axons_and_dentrites_backward_layers(neurons_stress, backward_activations, learning_rate)
            per_batch_stress.append(average_layer_stress)

        return cp.mean(cp.array(per_batch_stress))

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

    #TODO: This function will not return otherwise it will run as we call it
    return runner

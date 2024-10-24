import cupy as cp
from features import GREEN, RED, RESET
from nn_utils.loss_functions import cross_entropy_loss
from nn_utils.activation_functions import leaky_relu, softmax
from cupy_utils.utils import axons_initialization, dentrites_initialization, one_hot

def neural_network(network_architecture: list):
    # Initializae axons and dentrites for forward pass and backward pass
    forward_axons = [axons_initialization(network_architecture[size_idx], network_architecture[size_idx+1]) for size_idx in range(len(network_architecture)-1)]
    forward_dentrites = [dentrites_initialization(network_architecture[size_idx+1]) for size_idx in range(len(network_architecture)-1)]
    # Backward pass axons and dentrites same as forward axons and dentrties except for layer_neurons 2 to output neurons
    reverse_network_architecture = network_architecture[::-1]
    backward_axons = [forward_axons[-(size_idx+1)].transpose() if size_idx != len(reverse_network_architecture)-2 else axons_initialization(reverse_network_architecture[size_idx], reverse_network_architecture[size_idx+1])
                      for size_idx in range(len(reverse_network_architecture)-1)]
    backward_dentrites = [forward_dentrites[-(size_idx+2)] if size_idx != len(reverse_network_architecture)-2 else dentrites_initialization(reverse_network_architecture[size_idx+1])
                          for size_idx in range(len(reverse_network_architecture)-1)]

    def forward_in_neurons(neurons):
        neurons_activations = [neurons]
        for layer_idx in range(len(network_architecture)-1):
            forward_layer_idx  = len(network_architecture)-2 == layer_idx
            if not forward_layer_idx:
                neurons = leaky_relu(cp.dot(neurons, forward_axons[layer_idx]) + forward_dentrites[layer_idx])
            else:
                neurons = softmax(cp.dot(neurons, forward_axons[layer_idx]) + forward_dentrites[layer_idx])
            neurons_activations.append(neurons)
        return neurons_activations

    def backward_in_neurons(neurons):
        neurons_activations = [neurons]
        for layer_idx in range(len(network_architecture)-1):
            backward_layer_idx = len(network_architecture)-2 == layer_idx
            if not backward_layer_idx:
                neurons = leaky_relu(cp.dot(neurons, backward_axons[layer_idx]) + backward_dentrites[layer_idx])
            else:
                neurons = softmax(cp.dot(neurons, backward_axons[layer_idx]) + backward_dentrites[layer_idx])
            neurons_activations.append(neurons)
        return neurons_activations

    def calculate_network_stress(forward_activations, backward_activations):
        forward_pass_stress = []
        backward_pass_stress = []
        tuple_of_forward_neurond_activation_and_stress = []
        tuple_of_backward_neurons_activation_and_stress = []
        for activation_idx in range(len(network_architecture)-1):
            forward_activation = forward_activations[-(activation_idx+2)]
            if activation_idx == 0:
                forward_neurons_stress = cross_entropy_loss(forward_activations[-(activation_idx+1)],  backward_activations[activation_idx])[1]
            else:
                forward_neurons_stress = forward_activations[-(activation_idx+1)] - backward_activations[activation_idx]
            backward_activation = backward_activations[-(activation_idx+2)]
            backward_neurons_stress = backward_activations[-(activation_idx+1)] - forward_activations[activation_idx]

            tuple_of_forward_neurond_activation_and_stress.append((forward_activation, forward_neurons_stress))
            tuple_of_backward_neurons_activation_and_stress.append((backward_activation, backward_neurons_stress))
            forward_pass_stress.append(cp.mean(forward_neurons_stress))
            backward_pass_stress.append(cp.mean(backward_neurons_stress))
        return tuple_of_forward_neurond_activation_and_stress, tuple_of_backward_neurons_activation_and_stress, forward_pass_stress, backward_pass_stress

    def update_axons_and_dentrites(neurons_and_stress, axons_to_update: list, dentrites_to_update: list, learning_rate):
        # Update from output connection to input connection
        for layer_idx in range(len(network_architecture)-1):
            neurons_activation = neurons_and_stress[layer_idx][0]
            neurons_stress = neurons_and_stress[layer_idx][1]

            axons_to_update[-(layer_idx+1)] -= learning_rate * cp.dot(neurons_activation.transpose(), neurons_stress)
            dentrites_to_update[-(layer_idx+1)] -= learning_rate * cp.sum(neurons_stress, axis=0)

    def training_run(dataloader, learning_rate):
        per_batch_forward_stress = []
        per_batch_backward_stress = []
        for input_batch, expected_batch in dataloader:
            input_batch = cp.array(input_batch).reshape(input_batch.shape[0], -1)
            expected_batch = one_hot(cupy_array=expected_batch, number_of_classes=10)
            forward_activations = forward_in_neurons(input_batch)
            backward_activations = backward_in_neurons(expected_batch)
            forward_neurons_and_stress, backward_neurons_and_stress, forward_stress, backward_stress = calculate_network_stress(forward_activations, backward_activations)
            # Update bidirectional passes parameters
            update_axons_and_dentrites(forward_neurons_and_stress, forward_axons, forward_dentrites, learning_rate)
            # update_axons_and_dentrites(backward_neurons_and_stress, backward_axons, backward_dentrites, learning_rate)
            per_batch_forward_stress.append(forward_stress)
            per_batch_backward_stress.append(backward_stress)

        return cp.mean(cp.array(per_batch_forward_stress))

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
    
    def runner(epochs, training_loader, validation_loader, learning_rate):
        for epoch in range(epochs):
            # Training
            training_loss = training_run(training_loader, learning_rate)
            # Test
            accuracy = test_run(validation_loader)
            print(f"EPOCH: {epoch+1} Training Loss: {training_loss} Model Accuracy: {accuracy}")

    return runner

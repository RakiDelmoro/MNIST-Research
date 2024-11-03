import torch
import cupy as cp
from features import GREEN, RED, RESET
from cupy_utils.utils import cupy_array
from nn_utils.loss_functions import cross_entropy_loss

def forward_pass_activations(input_feature, layers_parameters):
    total_activations = len(layers_parameters)
    neurons = cp.array(input_feature)
    neurons_activations = [neurons]
    for each in range(total_activations):
        axons = layers_parameters[each][0]
        dentrites = layers_parameters[each][1] 
        neurons = cp.dot(neurons, axons)
        neurons_activations.append(neurons)
    return neurons_activations

def backward_pass_network_stress(backprop_stress, expected_activation, layers_activations, layers_parameters):
    total_layers_stress = len(layers_parameters)-1
    layers_stress = [backprop_stress]
    batch_size = layers_activations[0].shape[0]
    reconstruction = expected_activation
    reconstruction_layers = [reconstruction]
    for each in range(total_layers_stress):
        axons = layers_parameters[-(each+1)][0]
        backprop_stress = (cp.dot(backprop_stress, axons.transpose())) / batch_size

        previous_activation = layers_activations[-(each+2)]
        # Calculate ‖𝐲ℓ−1(i)−𝑾ℓ−1,ℓT⁢σ(𝑾ℓ−1,ℓ⁢𝐲ℓ−1(i))‖2
        reconstruction = cp.dot(reconstruction, axons.transpose())
        oja_stress = previous_activation - reconstruction
        loss = cp.sum(cp.linalg.norm(oja_stress, axis=1)**2) / batch_size

        layer_loss = backprop_stress + oja_stress
        layers_stress.append(layer_loss)
        reconstruction_layers.append(reconstruction)
    return layers_stress, reconstruction_layers

def update_layers_parameters(neurons_activations, reconstruct_layers, layers_losses, layers_parameters, learning_rate):
    total_parameters = len(layers_parameters)
    for layer_idx in range(total_parameters):
        axons = layers_parameters[-(layer_idx+1)][0]
        # dentrites = layers_parameters[-(layer_idx+1)][1]
        current_activation = reconstruct_layers[layer_idx]
        previous_activation = neurons_activations[-(layer_idx+2)]
        loss = layers_losses[layer_idx]

        backprop_parameters_nudge = learning_rate * cp.dot(previous_activation.transpose(), loss)
        oja_parameters_nudge = learning_rate * (cp.dot(previous_activation.transpose(), current_activation) - cp.dot(cp.dot(current_activation.transpose(), current_activation), axons.transpose()).transpose())        

        axons -= (backprop_parameters_nudge / current_activation.shape[0])
        axons += (oja_parameters_nudge / current_activation.shape[0])
        # dentrites -= learning_rate * cp.sum(loss, axis=0) / current_activation.shape[0]

def training_layers(dataloader, layers_parameters, learning_rate):
    per_batch_stress = []
    for i, (input_batch, expected_batch) in enumerate(dataloader):
        neurons_activations = forward_pass_activations(input_batch, layers_parameters)
        stress, stress_to_propagate = cross_entropy_loss(neurons_activations[-1], cp.array(expected_batch))
        network_layers_stress, reconstruct_layers = backward_pass_network_stress(cp.array(stress_to_propagate), cp.array(expected_batch), neurons_activations, layers_parameters)
        update_layers_parameters(neurons_activations, reconstruct_layers, network_layers_stress, layers_parameters, learning_rate)
        print(f"Loss each batch {i+1}: {stress}\r", end="", flush=True)
        per_batch_stress.append(stress)
    return cp.mean(cp.array(per_batch_stress))

def test_layers(dataloader, layers_parameters):
    per_batch_accuracy = []
    wrong_samples_indices = []
    correct_samples_indices = []
    model_predictions = []
    expected_model_prediction = []
    for input_image_batch, expected_batch in dataloader:
        # expected_batch = cp.array(one_hot(expected_batch, num_classes=10))
        expected_batch = cupy_array(expected_batch)
        model_output = forward_pass_activations(input_image_batch, layers_parameters)[-1]
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

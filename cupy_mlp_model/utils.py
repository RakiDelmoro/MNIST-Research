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

def oja_gradient(layers_activations, layers_parametrs):
    avg_reconsturcted_error = []
    layers_reconstruction_error = []
    total_reconstruction = len(layers_parametrs)
    for each in range(total_reconstruction):
        axons = layers_parametrs[-(each+1)][0]
        current_activation = layers_activations[-(each+1)]
        # 𝐲ℓ−1(i)−𝑾ℓ−1,ℓT⁢σ(𝑾ℓ−1,ℓ⁢𝐲ℓ−1(i)
        reconstruction = cp.dot(cp.dot(current_activation, axons.transpose()), axons)
        reconstruction_error = current_activation - reconstruction
        # Calculate ‖𝐲ℓ−1(i)−𝑾ℓ−1,ℓT⁢σ(𝑾ℓ−1,ℓ⁢𝐲ℓ−1(i))‖2
        stress = cp.sum(cp.linalg.norm(reconstruction)**2) / current_activation.shape[0]
        layers_reconstruction_error.append(reconstruction_error)
        avg_reconsturcted_error.append(stress)
    return avg_reconsturcted_error, layers_reconstruction_error

def backpropagation_gradient(expected_layer_output, layers_activations, layers_parameters):
    stress, stress_to_propagte = cross_entropy_loss(layers_activations[-1], cp.array(expected_layer_output))
    layers_activation_gradient = [stress_to_propagte]
    for each in range(len(layers_activations)-2):
        axons = layers_parameters[-(each+1)][0]
        stress_to_propagte = cp.dot(stress_to_propagte, axons.transpose())
        layers_activation_gradient.append(stress_to_propagte)
    return stress, layers_activation_gradient

def calculate_layers_stress(layers_backprop_grad, layers_oja_grad):
    layers_gradient = []
    total_layers_stress = len(layers_backprop_grad)
    for each_act in range(total_layers_stress):
        backprop_grad = layers_backprop_grad[each_act]
        oja_grad = layers_oja_grad[each_act]
        layer_gradient = backprop_grad + oja_grad
        layers_gradient.append(layer_gradient)
    return layers_gradient

def update_layers_parameters(neurons_activations, layers_losses, layers_parameters, learning_rate):
    total_parameters = len(layers_losses)
    for layer_idx in range(total_parameters):
        axons = layers_parameters[-(layer_idx+1)][0]
        # dentrites = layers_parameters[-(layer_idx+1)][1]
        current_activation = neurons_activations[-(layer_idx+1)]
        previous_activation = neurons_activations[-(layer_idx+2)]
        loss = layers_losses[layer_idx]

        backprop_parameters_nudge = learning_rate * cp.dot(previous_activation.transpose(), loss)
        oja_parameters_nudge = 0.01 * (cp.dot(previous_activation.transpose(), current_activation) - cp.dot(cp.dot(current_activation.transpose(), current_activation), axons.transpose()).transpose())        

        axons -= (backprop_parameters_nudge / current_activation.shape[0])
        axons += (oja_parameters_nudge / current_activation.shape[0])
        # dentrites -= learning_rate * cp.sum(loss, axis=0) / current_activation.shape[0]

def training_layers(dataloader, layers_parameters, learning_rate):
    per_batch_stress = []
    for i, (input_batch, expected_batch) in enumerate(dataloader):
        neurons_activations = forward_pass_activations(input_batch, layers_parameters)
        backprop_stress, layers_stress = backpropagation_gradient(expected_batch, neurons_activations, layers_parameters)
        _, reconstruction_errors = oja_gradient(neurons_activations, layers_parameters)
        backprop_and_oja_combine_layers_stress = calculate_layers_stress(layers_stress, reconstruction_errors)
        update_layers_parameters(neurons_activations, backprop_and_oja_combine_layers_stress, layers_parameters, learning_rate)
        print(f"Loss each batch {i+1}: {backprop_stress}\r", end="", flush=True)
        per_batch_stress.append(backprop_stress)
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

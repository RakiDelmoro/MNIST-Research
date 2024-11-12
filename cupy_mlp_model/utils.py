import random
import cupy as cp
from features import GREEN, RED, RESET
from cupy_utils.utils import cupy_array
from nn_utils.activation_functions import leaky_relu
from nn_utils.loss_functions import cross_entropy_loss

def forward_pass_activations(input_feature, idx_to_apply_residual, layers_parameters):
    neurons = cp.array(input_feature)
    neurons_activations = [neurons]
    idx_to_pulled = 0
    total_activations = len(layers_parameters)+1
    for layer_idx in range(1, total_activations):
        axons = layers_parameters[layer_idx-1][0]
        if layer_idx > 1 and layer_idx in idx_to_apply_residual:
            activation_idx_to_be_pulled = idx_to_apply_residual[idx_to_pulled]
            neurons = (leaky_relu(cp.dot(neurons, axons))) + neurons_activations[activation_idx_to_be_pulled]
            idx_to_pulled += 1
        else:
            neurons = leaky_relu(cp.dot(neurons, axons))
        neurons_activations.append(neurons)
    return neurons_activations

def reconstructed_activation_error(activation, axons):
    # 𝐲ℓ−1(i)−𝑾ℓ−1,ℓT⁢σ(𝑾ℓ−1,ℓ⁢𝐲ℓ−1(i)
    reconstructed_activation = cp.dot(leaky_relu(cp.dot(activation, axons.transpose())), axons)
    neurons_reconstructed_error = activation - reconstructed_activation
    # 𝒥=1T⁢∑i=1T‖𝐲ℓ−1(i)−𝑾ℓ−1,ℓT⁢σ⁢(𝑾ℓ−1,ℓ⁢𝐲ℓ−1(i))‖2
    avg_reconstructed_error = cp.sum(cp.linalg.norm(neurons_reconstructed_error)**2) / activation.shape[0]
    return avg_reconstructed_error, neurons_reconstructed_error

def calculate_layers_stress(neurons_stress, layers_activations, residual_indexes, layers_parameters):
    idx_to_aggregate_stress = [(len(layers_activations))-index for index in residual_indexes[::-1]]
    backprop_stress_to_aggregate = []
    oja_stress_to_aggregate = []
    backprop_and_oja_layers_gradient = []
    total_layers_stress = len(layers_activations)-1
    for each_layer in range(total_layers_stress):
        axons = layers_parameters[-(each_layer+1)][0]
        current_activation = layers_activations[-(each_layer+1)]
        _, neurons_reconstructed_error = reconstructed_activation_error(current_activation, axons)
        if each_layer in idx_to_aggregate_stress:
            if len(backprop_stress_to_aggregate) == 0:
                layer_gradient = neurons_stress
                backprop_stress_to_aggregate.append(neurons_stress)
                # oja_stress_to_aggregate.append(neurons_reconstructed_error)
            else:
                backprop_aggregated_stress = cp.sum(cp.stack(backprop_stress_to_aggregate), axis=0)
                layer_gradient = backprop_aggregated_stress
                backprop_stress_to_aggregate.append(neurons_stress)
                # oja_stress_to_aggregate.append(neurons_reconstructed_error)
        else:
            layer_gradient = neurons_stress
        neurons_stress = cp.dot(neurons_stress, axons.transpose())
        backprop_and_oja_layers_gradient.append(layer_gradient)
    return backprop_and_oja_layers_gradient

def update_layers_parameters(neurons_activations, layers_losses, layers_parameters, learning_rate):
    total_parameters = len(layers_losses)
    for layer_idx in range(total_parameters):
        axons = layers_parameters[-(layer_idx+1)][0]
        current_activation = neurons_activations[-(layer_idx+1)]
        previous_activation = neurons_activations[-(layer_idx+2)]
        loss = layers_losses[layer_idx]
        backprop_parameters_nudge = learning_rate * cp.dot(previous_activation.transpose(), loss)
        # oja_parameters_nudge = 0.01 * (cp.dot(previous_activation.transpose(), current_activation) - cp.dot(cp.dot(current_activation.transpose(), current_activation), axons.transpose()).transpose())
        axons -= (backprop_parameters_nudge / current_activation.shape[0])
        # axons += (oja_parameters_nudge / current_activation.shape[0])

def training_layers(dataloader, layers_parameters, learning_rate):
    per_batch_stress = []
    residual_idx = [(2**n) for n in range(len(layers_parameters)) if 2**n < len(layers_parameters)]
    for i, (input_batch, expected_batch) in enumerate(dataloader):
        neurons_activations = forward_pass_activations(input_batch, residual_idx, layers_parameters)
        avg_last_neurons_stress, neurons_stress_to_backpropagate = cross_entropy_loss(neurons_activations[-1], cp.array(expected_batch))
        backprop_and_oja_combine_layers_stress = calculate_layers_stress(neurons_stress_to_backpropagate, neurons_activations, residual_idx, layers_parameters)
        update_layers_parameters(neurons_activations, backprop_and_oja_combine_layers_stress, layers_parameters, learning_rate)
        print(f"Loss each batch {i+1}: {avg_last_neurons_stress}\r", end="", flush=True)
        per_batch_stress.append(avg_last_neurons_stress)
    return cp.mean(cp.array(per_batch_stress))

def test_layers(dataloader, layers_parameters, total_samples=100):
    residual_idx = [2**n for n in range(len(layers_parameters)) if 2**n < len(layers_parameters)]
    correct_predictions = []
    wrong_predictions = []
    model_predictions = []
    for i, (input_image_batch, expected_batch) in enumerate(dataloader):
        expected_batch = cupy_array(expected_batch)
        model_output = forward_pass_activations(input_image_batch, residual_idx, layers_parameters)[-1]
        model_prediction = cp.array(expected_batch.argmax(-1) == (model_output).argmax(-1)).astype(cp.float16).item()
        model_predictions.append(model_prediction)        
        if model_output.argmax(-1) == expected_batch.argmax(-1):
            correct_predictions.append((model_output.argmax(-1).item(), expected_batch.argmax(-1).item()))
        else:
            wrong_predictions.append((model_output.argmax(-1).item(), expected_batch.argmax(-1).item()))

        if i > total_samples:
            break
    random.shuffle(correct_predictions)
    random.shuffle(wrong_predictions)
    print(f"{GREEN}MODEL Correct Predictions{RESET}")
    [print(f"Digit Image is: {GREEN}{expected}{RESET} Model Prediction: {GREEN}{prediction}{RESET}") for i, (prediction, expected) in enumerate(correct_predictions) if i < 10]
    print(f"{RED}MODEL Wrong Predictions{RESET}")
    [print(f"Digit Image is: {RED}{expected}{RESET} Model Prediction: {RED}{prediction}{RESET}") for i, (prediction, expected) in enumerate(wrong_predictions) if i < 10]
    return cp.mean(cp.array(model_predictions)).item()

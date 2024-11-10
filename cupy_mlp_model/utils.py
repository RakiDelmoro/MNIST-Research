import cupy as cp
from features import GREEN, RED, RESET
from cupy_utils.utils import cupy_array
from nn_utils.activation_functions import leaky_relu
from nn_utils.loss_functions import cross_entropy_loss

def forward_pass_activations(input_feature, layers_parameters):
    total_activations = len(layers_parameters)
    neurons = cp.array(input_feature)
    neurons_activations = [neurons]
    layer_idx_to_apply_residual = 0
    for each in range(total_activations):
        axons = layers_parameters[each][0]
        dentrites = layers_parameters[each][1]
        if layer_idx_to_apply_residual == 8:
            neurons = neurons_activations[1]
            layer_idx_to_apply_residual = 0
        else:
            neurons = leaky_relu(cp.dot(neurons, axons))
            layer_idx_to_apply_residual += 1
        neurons_activations.append(neurons)
    return neurons_activations

def reconstructed_activation_error(activation, axons):
    # 𝐲ℓ−1(i)−𝑾ℓ−1,ℓT⁢σ(𝑾ℓ−1,ℓ⁢𝐲ℓ−1(i)
    reconstructed_activation = cp.dot(leaky_relu(cp.dot(activation, axons.transpose())), axons)
    neurons_reconstructed_error = activation - reconstructed_activation
    # 𝒥=1T⁢∑i=1T‖𝐲ℓ−1(i)−𝑾ℓ−1,ℓT⁢σ⁢(𝑾ℓ−1,ℓ⁢𝐲ℓ−1(i))‖2
    avg_reconstructed_error = cp.sum(cp.linalg.norm(neurons_reconstructed_error)**2) / activation.shape[0]
    return avg_reconstructed_error, neurons_reconstructed_error

def calculate_layers_stress(neurons_stress, layers_activations, layers_parameters):
    backprop_and_oja_layers_gradient = []
    total_layers_stress = len(layers_activations)-1
    for each_layer in range(total_layers_stress):
        axons = layers_parameters[-(each_layer+1)][0]
        current_activation = layers_activations[-(each_layer+1)]
        avg_error, neurons_reconstructed_error = reconstructed_activation_error(current_activation, axons)
        layer_gradient = neurons_stress - neurons_reconstructed_error
        neurons_stress = cp.dot(neurons_stress, axons.transpose())
        backprop_and_oja_layers_gradient.append(layer_gradient)
    return backprop_and_oja_layers_gradient

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

def training_layers(dataloader, layers_parameters, learning_rate, training_data_length=1000):
    per_batch_stress = []
    for i, (input_batch, expected_batch) in enumerate(dataloader):
        neurons_activations = forward_pass_activations(input_batch, layers_parameters)
        avg_last_neurons_stress, neurons_stress_to_backpropagate = cross_entropy_loss(neurons_activations[-1], cp.array(expected_batch))
        backprop_and_oja_combine_layers_stress = calculate_layers_stress(neurons_stress_to_backpropagate, neurons_activations, layers_parameters)
        update_layers_parameters(neurons_activations, backprop_and_oja_combine_layers_stress, layers_parameters, learning_rate)
        print(f"Loss each batch {i+1}: {avg_last_neurons_stress}\r", end="", flush=True)
        per_batch_stress.append(avg_last_neurons_stress)
        if i == training_data_length:
            break
    return cp.mean(cp.array(per_batch_stress))

def test_layers(dataloader, layers_parameters, total_samples=100):
    correct_predictions = []
    wrong_predictions = []
    model_predictions = []
    for i, (input_image_batch, expected_batch) in enumerate(dataloader):
        # expected_batch = cp.array(one_hot(expected_batch, num_classes=10))
        expected_batch = cupy_array(expected_batch)
        model_output = forward_pass_activations(input_image_batch, layers_parameters)[-1]
        model_prediction = cp.array(expected_batch.argmax(-1) == (model_output).argmax(-1)).astype(cp.float16).item()
        model_predictions.append(model_prediction)        
        if model_output.argmax(-1) == expected_batch.argmax(-1):
            correct_predictions.append((model_output.argmax(-1).item(), expected_batch.argmax(-1).item()))
        else:
            wrong_predictions.append((model_output.argmax(-1).item(), expected_batch.argmax(-1).item()))
        
        if i == total_samples:
            break
    
    print(f"{GREEN}Model Correct Predictions{RESET}")
    for i, (prediction, expected) in enumerate(correct_predictions):
        print(f"Digit Image is: {GREEN}{expected}{RESET} Model Predictions: {GREEN}{prediction}{RESET}")
        if i == 10:
            break
    print(f"{RED}Model Wrong Predictions{RESET}")
    for i, (prediction, expected) in enumerate(wrong_predictions):
        print(f"Digit Image is: {RED}{expected}{RESET} Model Predictions: {RED}{prediction}{RESET}")
        if i == 10:
            break
    return cp.mean(cp.array(model_predictions)).item()

# def test_layers(dataloader, layers_parameters):
#     per_batch_accuracy = []
#     wrong_samples_indices = []
#     correct_samples_indices = []
#     model_predictions = []
#     expected_model_prediction = []
#     for input_image_batch, expected_batch in dataloader:
#         # expected_batch = cp.array(one_hot(expected_batch, num_classes=10))
#         expected_batch = cupy_array(expected_batch)
#         model_output = forward_pass_activations(input_image_batch, layers_parameters)[-1]
#         batch_accuracy = cp.array(expected_batch.argmax(-1) == (model_output).argmax(-1)).mean()
#         correct_indices_in_a_batch = cp.where(expected_batch.argmax(-1) == model_output.argmax(-1))[0]
#         wrong_indices_in_a_batch = cp.where(~(expected_batch.argmax(-1) == model_output.argmax(-1)))[0]

#         per_batch_accuracy.append(batch_accuracy.item())
#         correct_samples_indices.append(correct_indices_in_a_batch)
#         wrong_samples_indices.append(wrong_indices_in_a_batch)
#         model_predictions.append(model_output.argmax(-1))
#         expected_model_prediction.append(expected_batch.argmax(-1))

#     model_accuracy = cp.mean(cp.array(per_batch_accuracy))
#     correct_samples = cp.concatenate(correct_samples_indices)[list(range(0,len(correct_samples_indices)))]
#     wrong_samples = cp.concatenate(wrong_samples_indices)[list(range(0,len(wrong_samples_indices)))]
#     model_prediction = cp.concatenate(model_predictions)
#     model_expected_prediction = cp.concatenate(expected_model_prediction)
    
#     print(f"{GREEN}Model Correct Predictions{RESET}")
#     for indices in correct_samples: print(f"Digit Image is: {GREEN}{model_expected_prediction[indices]}{RESET} Model Prediction: {GREEN}{model_prediction[indices]}{RESET}")
#     print(f"{RED}Model Wrong Predictions{RESET}")
#     for indices in wrong_samples: print(f"Digit Image is: {RED}{model_expected_prediction[indices]}{RESET} Model Predictions: {RED}{model_prediction[indices]}{RESET}")

    return model_accuracy

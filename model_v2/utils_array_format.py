import cupy as cp
from torch.nn.functional import one_hot

def axons_initialization(input_feature, output_feature):
    # weights = cp.random.randn(input_feature, output_feature)
    bound_w = cp.sqrt(3) * cp.sqrt(5) / cp.sqrt(input_feature) if input_feature > 0 else 0
    weights = cp.random.uniform(-bound_w, bound_w, size=(input_feature, output_feature))
    return weights

def dentrites_initialization(output_feature):
    # bias = cp.random.randn(output_feature)
    bound_b = 1 / cp.sqrt(output_feature) if output_feature > 0 else 0
    bias = cp.random.uniform(-bound_b, bound_b, size=(output_feature,))
    return bias

def forward_pass_architecture(network_features_size: list):
    # Shape -> (Total connection, maximum value in network_feature_size, maximum value in network_feature_size)
    layers_axons = cp.full(shape=(len(network_features_size)-1, max(network_features_size), max(network_features_size)), fill_value=cp.nan)
    # Shape -> (Total activation not include the input, maximum value in network_feature_size)
    layers_dentrites = cp.full(shape=(len(network_features_size)-1, max(network_features_size)), fill_value=cp.nan)
    for feature_size_idx in range(len(network_features_size)-1):
        # [connection index, fill the value until to a certain feature_size, fill the value until to a certain feature_size]
        layers_axons[feature_size_idx, :network_features_size[feature_size_idx], :network_features_size[feature_size_idx+1]] = axons_initialization(network_features_size[feature_size_idx], network_features_size[feature_size_idx+1])
        # [activation index, fill the value until to a certain feature_size]
        layers_dentrites[feature_size_idx, :network_features_size[feature_size_idx+1]] = dentrites_initialization(network_features_size[feature_size_idx+1])

    return layers_axons, layers_dentrites

def backward_pass_architecture(network_features_size: list):
    layers_axons = cp.full(shape=(len(network_features_size)-1, max(network_features_size), max(network_features_size)), fill_value=cp.nan)
    layers_dentrites = cp.full(shape=(len(network_features_size)-1, max(network_features_size)), fill_value=cp.nan)
    for feature_size_idx in range(len(network_features_size)-1):
        # [connection index, fill the value until to a certain feature_size, fill the value until to a certain feature_size]
        layers_axons[feature_size_idx, :network_features_size[feature_size_idx], :network_features_size[feature_size_idx+1]] = axons_initialization(network_features_size[feature_size_idx], network_features_size[feature_size_idx+1])
        # [activation index, fill the value until to a certain feature_size]
        layers_dentrites[feature_size_idx, :network_features_size[feature_size_idx+1]] = dentrites_initialization(network_features_size[feature_size_idx+1])
 
    return layers_axons, layers_dentrites

def get_non_nan_value_for_axons(array):
    # Use ~cp.isnan() to filter out NaN values
    valid_rows = cp.any(~cp.isnan(array), axis=1)
    valid_cols = cp.any(~cp.isnan(array), axis=0)
    return array[valid_rows][:, valid_cols]

def get_non_nan_value_for_dentrites(array):
    return array[~cp.isnan(array)]

def get_non_nan_value(array):
    valid_cols = cp.any(~cp.isnan(array), axis=0)
    # Filter out NaN values and return the shape that is not Nan value
    return array[:, valid_cols]

def get_network_activations_array_format(input_neurons, network_architecture, network_axons_and_dentrites):
    neurons = cp.array(input_neurons)
    # Shape -> (Total activations, batch_size, maximum value in network_architecture)
    neurons_activations = cp.full(shape=(len(network_architecture),  neurons.shape[0], max(network_architecture)), fill_value=cp.nan)
    neurons_activations[0, :, :neurons.shape[-1]] = neurons
    for neurons_layer_idx in range(len(network_architecture)-1):
        axons = get_non_nan_value_for_axons(network_axons_and_dentrites[0][neurons_layer_idx])
        neurons = cp.dot(neurons, axons)
        neurons_activations[neurons_layer_idx+1, :, :neurons.shape[-1]] = neurons
    return neurons_activations

def layers_of_neurons_stress(network_architecture, forward_pass_activations, backward_pass_activations):
    neurons_stress = []
    for activation_idx in range(len(network_architecture)):
        forward_activation = get_non_nan_value(forward_pass_activations[activation_idx])
        backward_activation = get_non_nan_value(backward_pass_activations[-(activation_idx+1)])
        stress = forward_activation - backward_activation
        neurons_stress.append(stress)
    return neurons_stress

def nudge_axons_and_dentrites(layers_stress, neurons_activations, axons_and_dentrites, for_backward_pass, learning_rate):
    axons = axons_and_dentrites[0]
    dentrites = axons_and_dentrites[-1]
    for layer_connection in range(len(axons)):
        if not for_backward_pass:
            layer_neuron_activation = get_non_nan_value(neurons_activations[layer_connection])
            layer_stress = layers_stress[layer_connection+1]
            # Update the Nan values
            layer_axons = get_non_nan_value_for_axons(axons[layer_connection])
            layer_axons -= learning_rate * cp.dot(layer_neuron_activation.transpose(), layer_stress)
            # dentrites[layer_connection] -= learning_rate * cp.sum(layer_stress, axis=0)
        else:
            layer_neuron_activation = get_non_nan_value(neurons_activations[layer_connection])
            layer_stress = layers_stress[-(layer_connection+2)]
            layer_axons = get_non_nan_value_for_axons(axons[layer_connection])
            layer_axons -= learning_rate * cp.dot(layer_neuron_activation.transpose(), layer_stress)
            # dentrites[layer_connection] -= learning_rate * cp.sum(layer_stress, axis=0)

def test_run_result(dataloader, forward_in_neurons):
    per_batch_accuracy = []
    wrong_samples_indices = []
    correct_samples_indices = []
    model_predictions = []
    expected_model_prediction = []
    for input_image_batch, expected_batch in dataloader:
        # expected_batch = cp.array(one_hot(expected_batch, num_classes=10))
        expected_batch = cp.array(expected_batch)
        model_output = get_non_nan_value(forward_in_neurons(input_image_batch)[-1])
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

    return model_accuracy, correct_samples, wrong_samples, model_prediction, model_expected_prediction

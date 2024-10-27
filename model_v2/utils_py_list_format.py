import cupy as cp
from torch.nn.functional import one_hot

def axons_initialization(input_feature, output_feature):
    bound_w = cp.sqrt(3) * cp.sqrt(5) / cp.sqrt(input_feature) if input_feature > 0 else 0
    weights = cp.random.uniform(-bound_w, bound_w, size=(input_feature, output_feature))
    # weights = cp.random.randn(input_feature, output_feature)
    return weights

def dentrites_initialization(output_feature):
    # bias = cp.random.randn(output_feature)
    bound_b = 1 / cp.sqrt(output_feature) if output_feature > 0 else 0
    bias = cp.random.uniform(-bound_b, bound_b, size=(output_feature,))
    return bias

def forward_pass_architecture(network_features_size: list):
    layers_axons = []
    layers_dentrites = []
    for feature_size_idx in range(len(network_features_size)-1):
        axons = axons_initialization(network_features_size[feature_size_idx], network_features_size[feature_size_idx+1])
        dentrites = dentrites_initialization(network_features_size[feature_size_idx+1])
        layers_axons.append(axons)
        layers_dentrites.append(dentrites)
    return layers_axons, layers_dentrites

def backward_pass_architecture(network_features_size: list):
    layers_axons = []
    layers_dentrites = []
    for feature_size_idx in range(len(network_features_size)-1):
        axons = axons_initialization(network_features_size[feature_size_idx], network_features_size[feature_size_idx+1])
        dentrites = dentrites_initialization(network_features_size[feature_size_idx+1])
        layers_axons.append(axons)
        layers_dentrites.append(dentrites)

    return layers_axons, layers_dentrites

def get_network_activations(input_neurons, network_architecture, network_axons_and_dentrites):
    neurons = cp.array(input_neurons)
    neurons_activations = [neurons]
    for neurons_layer_idx in range(len(network_architecture)-1):
        axons = network_axons_and_dentrites[0][neurons_layer_idx]
        dentrites = network_axons_and_dentrites[-1][neurons_layer_idx]
        neurons = cp.dot(neurons, axons) + dentrites
        neurons_activations.append(neurons)
    return neurons_activations

def layers_of_neurons_stress(network_architecture, forward_pass_activations, backward_pass_activations):
    neurons_stress = []
    for activation_idx in range(len(network_architecture)):
        forward_activation = forward_pass_activations[activation_idx]
        backward_activation = backward_pass_activations[-(activation_idx+1)]
        stress = forward_activation - backward_activation
        neurons_stress.append(stress)
    return neurons_stress

def nudge_axons_and_dentrites(layers_stress, neurons_activations, axons_and_dentrites, for_backward_pass, learning_rate):
    layers_axons = axons_and_dentrites[0]
    layers_dentrites = axons_and_dentrites[-1]
    for layer_connection_idx in range(len(layers_axons)):
        if not for_backward_pass:
            layer_neuron_activation = neurons_activations[layer_connection_idx]
            layer_stress = layers_stress[layer_connection_idx+1]
            layer_axons = layers_axons[layer_connection_idx]
            layer_dentrites = layers_dentrites[layer_connection_idx]
            # Problem shape mismatch:
            # layer_axons shape -> (input_feature, output_feature)
            # learning_rate * (cp.dot(layer_stress, layer_axons.transpose() / cp.sum(layer_axons))) -> (batch_size, input_feature)
            #TODO: create a function so that this will work -> layer_axons -= learning_rate * (cp.dot(layer_stress, layer_axons.transpose() / cp.sum(layer_axons))) -> (batch_size, input_feature)
            # layer_axons -= learning_rate * (cp.dot(layer_stress, layer_axons.transpose() / cp.sum(layer_axons)))
            layer_axons -= learning_rate * (cp.dot(layer_neuron_activation.transpose(), layer_stress) / cp.sum(layer_neuron_activation))
            layer_dentrites -= learning_rate * cp.sum(layer_stress, axis=0)
        else:
            layer_neuron_activation = neurons_activations[layer_connection_idx]
            layer_stress = layers_stress[-(layer_connection_idx+2)]
            layer_axons = layers_axons[layer_connection_idx]
            layer_dentrites = layers_dentrites[layer_connection_idx]
            layer_axons += learning_rate * (cp.dot(layer_neuron_activation.transpose(), layer_stress) / cp.sum(layer_neuron_activation))
            layer_dentrites += learning_rate * cp.sum(layer_stress, axis=0)

def test_run_result(dataloader, forward_in_neurons):
    per_batch_accuracy = []
    wrong_samples_indices = []
    correct_samples_indices = []
    model_predictions = []
    expected_model_prediction = []
    for input_image_batch, expected_batch in dataloader:
        # expected_batch = cp.array(one_hot(expected_batch, num_classes=10))
        expected_batch = cp.array(expected_batch)
        model_output = forward_in_neurons(input_image_batch)[-1]
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

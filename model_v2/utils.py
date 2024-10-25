import cupy as cp
# from cupy_utils.utils import axons_initialization, dentrites_initialization, one_hot

def axons_initialization(input_feature, output_feature):
    bound_w = cp.sqrt(3) * cp.sqrt(5) / cp.sqrt(input_feature) if input_feature > 0 else 0
    weights = cp.random.uniform(-bound_w, bound_w, size=(input_feature, output_feature))
    return weights

def dentrites_initialization(output_feature):
    bound_b = 1 / cp.sqrt(output_feature) if output_feature > 0 else 0
    bias = cp.random.uniform(-bound_b, bound_b, size=(output_feature,))
    return bias

def forward_pass_architecture(network_features_size: list):
    # Shape -> (Total connection, maximum value in network_feature_size, maximum value in network_feature_size)
    layers_axons = cp.zeros(shape=(len(network_features_size)-1, max(network_features_size), max(network_features_size)))
    # Shape -> (Total activation not include the input, maximum value in network_feature_size)
    layers_dentrites = cp.zeros(shape=(len(network_features_size)-1, max(network_features_size)))
    for feature_size_idx in range(len(network_features_size)-1):
        # [connection index, fill the value until to a certain feature_size, fill the value until to a certain feature_size]
        layers_axons[feature_size_idx, :network_features_size[feature_size_idx], :network_features_size[feature_size_idx+1]] = axons_initialization(network_features_size[feature_size_idx], network_features_size[feature_size_idx+1])
        # [activation index, fill the value until to a certain feature_size]
        layers_dentrites[feature_size_idx, :network_features_size[feature_size_idx+1]] = dentrites_initialization(network_features_size[feature_size_idx+1])

    return layers_axons, layers_dentrites

def backward_pass_architecture(network_features_size: list):
    layers_axons = cp.zeros(shape=(len(network_features_size)-1, max(network_features_size), max(network_features_size)))
    layers_dentrites = cp.zeros(shape=(len(network_features_size)-1, max(network_features_size)))
    for feature_size_idx in range(len(network_features_size)-1):
        # [connection index, fill the value until to a certain feature_size, fill the value until to a certain feature_size]
        layers_axons[feature_size_idx, :network_features_size[feature_size_idx], :network_features_size[feature_size_idx+1]] = axons_initialization(network_features_size[feature_size_idx], network_features_size[feature_size_idx+1])
        # [activation index, fill the value until to a certain feature_size]
        layers_dentrites[feature_size_idx, :network_features_size[feature_size_idx+1]] = dentrites_initialization(network_features_size[feature_size_idx+1])
 
    return layers_axons, layers_dentrites

def get_non_zero_value(array):
    return array[cp.where(cp.any(array != 0, axis=0))][cp.where(cp.any(array != 0, axis=1))]

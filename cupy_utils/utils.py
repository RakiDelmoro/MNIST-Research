import cupy as cp

def one_hot(cupy_array, number_of_classes):
    return cp.array(cp.eye(number_of_classes)[cupy_array], dtype=cp.float32)

def axons_initialization(input_feature, output_feature):
    bound_w = cp.sqrt(3) * cp.sqrt(5) / cp.sqrt(input_feature) if input_feature > 0 else 0
    weights = cp.random.uniform(-bound_w, bound_w, size=(input_feature, output_feature))
    return weights

def dentrites_initialization(output_feature):
    bound_b = 1 / cp.sqrt(output_feature) if output_feature > 0 else 0
    bias = cp.random.uniform(-bound_b, bound_b, size=(output_feature,))
    return bias
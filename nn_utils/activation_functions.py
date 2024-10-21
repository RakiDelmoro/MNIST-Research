import cupy as cp

def leaky_relu(input_data, return_derivative=False):
    if return_derivative:
        return cp.where(input_data > 0, 1, 0.05 * input_data)
    else:
        return cp.maximum(input_data * 0.05, input_data)

def relu(input_data, return_derivative=False):
    if return_derivative:
        return cp.where(input_data > 0, 1, 0)
    else:
        return cp.maximum(0, input_data)

def sigmoid(input_data, return_derivative=False):
    if return_derivative:
       return input_data * (1 - input_data)
    else:
        return 1.0 / (1.0+cp.exp(-input_data))

def softmax(input_data):
    # Subtract max value for numerical stability
    shifted_data = input_data - cp.max(input_data, axis=1, keepdims=True)
    # Calculate exp
    exp_data = cp.exp(shifted_data)
    # Sum along axis=1 and keep dimensions for broadcasting
    sum_exp_data = cp.sum(exp_data, axis=1, keepdims=True)

    return exp_data / sum_exp_data

import cupy as cp

def model_parameters_initialization(input_size: int, output_size: int):
    mean, std = 0, cp.sqrt(6 / (input_size + output_size))
    forward_weights = cp.random.normal(mean, std, size=(output_size, input_size), dtype=float)
    bias = cp.random.normal(mean, std, size=output_size, dtype=float)
    backward_weights = forward_weights.transpose()

    return forward_weights, bias, backward_weights

def initialize_layer(weights: cp.array, bias: cp.array, activation_function, is_output_layer=False):
    if is_output_layer:
        layer_calculation = lambda input_data: cp.dot(input_data, weights.transpose()) + bias
    else:
        layer_calculation = lambda input_data: activation_function(cp.dot(input_data, weights.transpose()) + bias)

    return layer_calculation

def initialize_learnable_parameters(model_feature_sizes: list):
    forward_parameters = []
    list_of_biases = []
    list_of_backward_weights = []
    for idx in range(len(model_feature_sizes)-1):
        forward_weights, bias, transposed_weights = model_parameters_initialization(model_feature_sizes[idx], model_feature_sizes[idx+1])
        forward_parameters.append([forward_weights, bias])
        
        list_of_backward_weights.insert(0, transposed_weights)
        list_of_biases.insert(0, bias)

    backward_parameters = []
    for idx in range(len(model_feature_sizes)-2):
        biases_for_backward_parameters = list_of_biases[1:]
        weights_for_backward_parameters = list_of_backward_weights[:-1]
        backward_weight = weights_for_backward_parameters[idx]
        backward_bias = biases_for_backward_parameters[idx]

        backward_parameters.append([backward_weight, backward_bias])

    return forward_parameters, backward_parameters

def initialize_model_layers_and_parameters(model_feature_sizes: list, activation_function, parameters: list):
    forward_layers = []
    backward_layers = []
    for idx in range(len(model_feature_sizes)-1):
        is_last_layer = idx == len(model_feature_sizes) - 1
        if is_last_layer:
            layer = initialize_layer(parameters[idx][0], parameters[idx][1], activation_function, is_last_layer)
            forward_layers.append(layer)
            backward_layers.append(layer)
        else:
            layer = initialize_layer(parameters[idx][0], parameters[idx][1], activation_function, is_last_layer)
            forward_layers.append(layer)
            backward_layers.append(layer)

    return forward_layers, backward_layers

def neural_network(model_architecture: list, activation_function):
    model_architecture_forward = model_architecture
    model_architecture_backward = model_architecture[1:]
    model_learnable_parameters = initialize_learnable_parameters(model_feature_sizes=model_architecture)

    def forward(input_batch, model_trainable_parameters):
        forward_layers, _ = initialize_model_layers_and_parameters(model_architecture_forward, activation_function, model_trainable_parameters)
        neurons_activations = [input_batch]
        previous_layer_output = input_batch
        for layer in forward_layers:
            previous_layer_output = layer(previous_layer_output)
            neurons_activations.append(previous_layer_output)

        return previous_layer_output, neurons_activations

    def backward(expected_batch, model_trainable_parameters):
        _, backward_layers = initialize_model_layers_and_parameters(model_architecture_backward, activation_function, model_trainable_parameters)
        neurons_activations = []
        previous_layer_output = expected_batch
        for layer in backward_layers:
            previous_layer_output = layer(previous_layer_output)
            neurons_activations.append(previous_layer_output)
        
        return neurons_activations

    def update_parameters(batch_output_error, model_parameters, learning_rate, forward_activations, backward_activations):
        batch_size = batch_output_error.shape[0]

        hidden_2_error = (forward_activations[-2] - backward_activations[0]) / batch_size
        hidden_1_error = (forward_activations[-3] - backward_activations[1]) / batch_size

        # Weight update
        model_parameters[-1][0] -= learning_rate * cp.dot(batch_output_error.transpose(), forward_activations[-2])
        model_parameters[-2][0] -= learning_rate * cp.dot(hidden_2_error.transpose(), forward_activations[-3])
        model_parameters[-3][0] -= learning_rate * cp.dot(hidden_1_error.transpose(), forward_activations[-4])
        # Bias update
        model_parameters[-1][1] -= learning_rate * cp.sum(batch_output_error, axis=0)
        model_parameters[-2][1] -= learning_rate * cp.sum(hidden_2_error, axis=0)
        model_parameters[-3][1] -= learning_rate * cp.sum(hidden_1_error, axis=0)

        forward_pass_parameters = [[model_parameters[0][0], model_parameters[0][1]], [model_parameters[1][0], model_parameters[1][1]], [model_parameters[2][0], model_parameters[2][1]]]
        backward_pass_parameters = [[model_parameters[-1][0].transpose(), model_parameters[-2][1]], [model_parameters[-2][0].transpose(), model_parameters[-3][1]]]

        return forward_pass_parameters, backward_pass_parameters

    def backpropagation(model_error, model_outputs, model_parameters, learning_rate):
        batch_size = model_error.shape[0]

        # hidden2 to output gradient
        hidden2_to_output_weights_gradient = cp.zeros_like(model_parameters[-1][0])
        output_bias_gradient = cp.zeros_like(model_parameters[-1][1])
        # hidden1 to hidden2 gradient 
        hidden1_to_hidden2_weights_gradient = cp.zeros_like(model_parameters[-2][0])
        hidden2_bias_gradient = cp.zeros_like(model_parameters[-2][1])
        # input to hidden1 gradient
        input_to_hidden1_weights_gradient = cp.zeros_like(model_parameters[-3][0])
        hidden1_bias_gradient = cp.zeros_like(model_parameters[-3][1])
        # Neurons gradient
        hidden2_neuron_gradient = cp.zeros_like(model_outputs[-2])
        hidden1_neuron_gradient = cp.zeros_like(model_outputs[-3])

        # Hidden 2 to output weights and bias gradients
        hidden2_to_output_weights_gradient += cp.dot(model_error.transpose(), model_outputs[-2]) 
        output_bias_gradient += cp.sum(model_error, axis=0) 
        # Hidden 2 neuron gradient
        not_activated_hidden2_neuron_gradient = cp.dot(model_error, model_parameters[-1][0]) 
        hidden2_neuron_gradient += not_activated_hidden2_neuron_gradient * activation_function(model_outputs[-2], return_derivative=True)
        # Hidden 1 to hidden 2 weights and bias gradients
        hidden1_to_hidden2_weights_gradient += cp.dot(hidden2_neuron_gradient.transpose(), model_outputs[-3]) 
        hidden2_bias_gradient += cp.sum(hidden2_neuron_gradient, axis=0) 
        # Hidden 1 neuron gradient
        not_activated_hidden1_neuron_gradient = cp.dot(hidden2_neuron_gradient, model_parameters[-2][0])
        hidden1_neuron_gradient += not_activated_hidden1_neuron_gradient * activation_function(model_outputs[-3], return_derivative=True)
        # Input to hidden 1 weights and bias gradients
        input_to_hidden1_weights_gradient += cp.dot(hidden1_neuron_gradient.transpose(), model_outputs[-4]) 
        hidden1_bias_gradient += cp.sum(hidden1_neuron_gradient, axis=0) 

        # Update parameters
        hidden2_to_output_weights_gradient /= batch_size
        output_bias_gradient /= batch_size
        hidden1_to_hidden2_weights_gradient /= batch_size
        hidden2_bias_gradient /= batch_size
        input_to_hidden1_weights_gradient /= batch_size
        hidden1_bias_gradient /= batch_size

        model_parameters[-1][0] -= learning_rate * hidden2_to_output_weights_gradient
        model_parameters[-1][1] -= learning_rate * output_bias_gradient

        model_parameters[-2][0] -= learning_rate * hidden1_to_hidden2_weights_gradient
        model_parameters[-2][1] -= learning_rate * hidden2_bias_gradient

        model_parameters[-3][0] -= learning_rate * input_to_hidden1_weights_gradient
        model_parameters[-3][1] -= learning_rate * hidden1_bias_gradient

        return model_parameters

    return forward, backward, update_parameters, model_learnable_parameters

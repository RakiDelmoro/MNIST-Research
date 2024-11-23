import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from nn_utils.activation_functions import relu, leaky_relu, sigmoid, tanh

RED = '\033[31m'
RESET = '\033[0m'
GREEN = '\033[32m'

def activation_visualizer(activation_function: str):
    x_data = np.linspace(-6, 6, 100)
    if activation_function == 'relu':
        y_data = cp.asnumpy(relu(cp.array(x_data)))
        derivative_data = cp.asnumpy(relu(cp.array(x_data), return_derivative=True))
    elif activation_function == 'leaky_relu':
        y_data = cp.asnumpy(leaky_relu(cp.array(x_data)))
        derivative_data = cp.asnumpy(leaky_relu(cp.array(x_data), return_derivative=True))
    elif activation_function == 'sigmoid':
        y_data = cp.asnumpy(sigmoid(cp.array(x_data)))
        derivative_data = cp.asnumpy(sigmoid(cp.array(x_data), return_derivative=True))
    elif activation_function == 'tanh':
        y_data = cp.asnumpy(tanh(cp.array(x_data)))
        derivative_data = cp.asnumpy(tanh(cp.array(x_data), return_derivative=True))

    # Draw graph
    plt.plot(x_data, y_data, x_data, derivative_data)
    plt.title(f'{activation_function} activation and derivative')
    plt.legend([f'{activation_function} activation', f'{activation_function } derivative'])
    plt.grid()
    plt.savefig('activation.png')

activation_visualizer('tanh')

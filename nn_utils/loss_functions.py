import cupy as cp
from cupy_utils.utils import cupy_array
from nn_utils.activation_functions import softmax

def cross_entropy_loss(model_prediction, expected):
    # Compute probability distribution of model output
    probability_distribution = softmax(model_prediction)
    loss = cp.mean(-cp.sum(expected * cp.log(probability_distribution), axis=1))
    loss_for_backprop = (probability_distribution - expected)

    return loss, cupy_array(loss_for_backprop)

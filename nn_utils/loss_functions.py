import cupy as cp
from nn_utils.activation_functions import softmax

def cross_entropy_loss(model_prediction, expected):
    # Compute probability distribution of model output
    probability_distribution = softmax(model_prediction)
    loss = (-cp.sum(expected * cp.log(probability_distribution))) / model_prediction.shape[0]
    loss_for_backprop = probability_distribution - expected

    return loss, loss_for_backprop

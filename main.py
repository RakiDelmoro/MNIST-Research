from runner import runner
from model import neural_network
from utils import load_data_to_memory
from activation_functions import relu

def main():
    EPOCHS = 50
    BATCH_SIZE = 2098
    IMAGE_WIDTH = 28
    IMAGE_HEIGHT = 28
    LEARNING_RATE = 0.001
    INPUT_DATA_FEATURE_SIZE = IMAGE_HEIGHT*IMAGE_WIDTH

    forward_pass, backward_pass, update_parameters, model_parameters = neural_network([INPUT_DATA_FEATURE_SIZE, 1000, 1000, 10], relu)
    training_tuple_of_arrays, validation_tuple_of_arrays = load_data_to_memory('./training-data/mnist.pkl.gz')

    runner(EPOCHS, training_tuple_of_arrays, validation_tuple_of_arrays, True, BATCH_SIZE, update_parameters, forward_pass, backward_pass, model_parameters, learning_rate=LEARNING_RATE)

main()
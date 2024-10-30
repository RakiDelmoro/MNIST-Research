import torch
from cupy_utils.utils import one_hot
# from torch.nn.functional import one_hot
from model_v2.model import neural_network
# from cupy_mlp_model.model import neural_network
from mlp_torch_model.model import MlpNetwork
from torch.utils.data import DataLoader
from utils import load_data_to_memory
from torchvision import transforms, datasets 
from nn_utils.activation_functions import leaky_relu

import matplotlib.pyplot as plt

def main():
    EPOCHS = 10
    BATCH_SIZE = 2098
    IMAGE_WIDTH = 28
    IMAGE_HEIGHT = 28
    LEARNING_RATE = 0.001
    NUMBER_OF_CLASSES = 10
    INPUT_DATA_FEATURE_SIZE = IMAGE_HEIGHT*IMAGE_WIDTH
    NETWORK_ARCHITECTURE = [INPUT_DATA_FEATURE_SIZE, 2000, 2000, NUMBER_OF_CLASSES]
    LOSS_FUNCTION = torch.nn.CrossEntropyLoss()
    TRANSFORM = lambda x: torch.flatten(transforms.ToTensor()(x)).type(dtype=torch.float32) #  transforms.Compose([transforms.ToTensor()]) 
    TARGET_TRANSFORM = lambda x: torch.tensor(one_hot(x, number_of_classes=NUMBER_OF_CLASSES), dtype=torch.float32)
    training_dataset = datasets.MNIST('./training-data', download=True, train=True, transform=TRANSFORM, target_transform=TARGET_TRANSFORM)
    validation_dataset = datasets.MNIST('./training-data', download=True, train=False, transform=TRANSFORM, target_transform=TARGET_TRANSFORM)
    training_dataloader = DataLoader(training_dataset, batch_size=BATCH_SIZE, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # TORCH Model
    # MlpNetwork(network_architecture=NETWORK_ARCHITECTURE).runner(epochs=EPOCHS, training_loader=training_dataloader, validation_loader=validation_dataloader, loss_function=LOSS_FUNCTION,
            #    learning_rate=LEARNING_RATE)

    # runner = neural_network(NETWORK_ARCHITECTURE)
    # runner(training_dataloader, validation_dataloader)
    # CUPY Model
    neural_network(network_architecture=NETWORK_ARCHITECTURE, training_dataloader=training_dataloader, validation_dataloader=validation_dataloader, learning_rate=LEARNING_RATE, epochs=EPOCHS)
main()
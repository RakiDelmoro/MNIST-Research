import torch
from utils import digit_generator
from cupy_utils.utils import one_hot
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from model_v2.model import neural_network_v2
from model_v3.model import neural_network_v3
from cupy_mlp_model.model import cupy_mlp_neural_network

def main():
    RESIDUAL_USE = True
    EPOCHS = 100
    BATCH_SIZE = 2048
    IMAGE_WIDTH = 28
    IMAGE_HEIGHT = 28
    LEARNING_RATE = 0.001
    NUMBER_OF_CLASSES = 10
    INPUT_DATA_FEATURE_SIZE = IMAGE_HEIGHT*IMAGE_WIDTH
    INPUT_AND_OUTPUT_LAYERS = [INPUT_DATA_FEATURE_SIZE, NUMBER_OF_CLASSES]
    MIDDLE_LAYERS = [200] * 201
    NETWORK_ARCHITECTURE = INPUT_AND_OUTPUT_LAYERS[:1] + MIDDLE_LAYERS + INPUT_AND_OUTPUT_LAYERS[1:]
    TRANSFORM = lambda x: torch.flatten(transforms.ToTensor()(x)).type(dtype=torch.float32)
    TARGET_TRANSFORM = lambda x: torch.tensor(one_hot(x, number_of_classes=NUMBER_OF_CLASSES), dtype=torch.float32)
    
    training_dataset = datasets.MNIST('./training-data', download=True, train=True, transform=TRANSFORM, target_transform=TARGET_TRANSFORM)
    validation_dataset = datasets.MNIST('./training-data', download=True, train=False, transform=TRANSFORM, target_transform=TARGET_TRANSFORM)
    training_dataloader = DataLoader(training_dataset, batch_size=BATCH_SIZE, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # neural_network_v3(NETWORK_ARCHITECTURE, training_dataloader, validation_dataloader, LEARNING_RATE)
    # neural_network_v2(NETWORK_ARCHITECTURE, training_dataloader, validation_dataloader, LEARNING_RATE, EPOCHS)
    cupy_mlp_neural_network(NETWORK_ARCHITECTURE, training_dataloader, validation_dataloader, LEARNING_RATE, EPOCHS, RESIDUAL_USE)

main()

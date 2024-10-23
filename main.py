import torch
from model_v2.model import neural_network
from mlp_torch_model.model import MlpNetwork
from torch.utils.data import DataLoader
from utils import load_data_to_memory
from torchvision import transforms, datasets
from nn_utils.activation_functions import leaky_relu

def main():
    EPOCHS = 10
    BATCH_SIZE = 2098
    IMAGE_WIDTH = 28
    IMAGE_HEIGHT = 28
    LEARNING_RATE = 0.01
    INPUT_DATA_FEATURE_SIZE = IMAGE_HEIGHT*IMAGE_WIDTH
    NETWORK_ARCHITECTURE = [INPUT_DATA_FEATURE_SIZE, 2000, 2000, 10]
    LOSS_FUNCTION = torch.nn.MSELoss()
    TRANSFORM = transforms.Compose([transforms.ToTensor()])

    training_dataset = datasets.MNIST('./training-data', download=True, train=True, transform=TRANSFORM)
    validation_dataset = datasets.MNIST('./training-data', download=True, train=False, transform=TRANSFORM)
    training_dataloader = DataLoader(training_dataset, batch_size=BATCH_SIZE, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # # # # TORCH Model
    # MlpNetwork(network_architecture=NETWORK_ARCH).runner(epochs=EPOCHS, training_loader=training_dataloader, validation_loader=validation_dataloader, loss_function=LOSS_FUNCTION,
    #                   learning_rate=LEARNING_RATE)

    # CUPY Model
    runner = neural_network(network_architecture=NETWORK_ARCHITECTURE)
    runner(EPOCHS, training_dataloader, validation_dataloader, LOSS_FUNCTION, LEARNING_RATE)
main()
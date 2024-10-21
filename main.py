import torch
from model_v1.model import neural_network
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
    LEARNING_RATE = 0.001
    INPUT_DATA_FEATURE_SIZE = IMAGE_HEIGHT*IMAGE_WIDTH
    LOSS_FUNCTION = torch.nn.CrossEntropyLoss()
    
    # Use to transform image for TORCH based model
    torch_based_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307, ), (0.3081,))])
    # Use to transfrom image for CUPY based model
    cupy_based_transform = transforms.Compose([transforms.Normalize((0.1307, ), (0.3081,))])

    training_dataset = datasets.MNIST('./training-data', download=True, train=True, transform=torch_based_transform)
    validation_dataset = datasets.MNIST('./training-data', download=True, train=False, transform=torch_based_transform)
    training_dataloader = DataLoader(training_dataset, batch_size=BATCH_SIZE, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # # TORCH Model
    MlpNetwork().runner(epochs=EPOCHS, training_loader=training_dataloader, validation_loader=validation_dataloader, loss_function=LOSS_FUNCTION,
                      learning_rate=LEARNING_RATE)

    # CUPY Model
    runner = neural_network([784, 2000, 2000, 10])
    runner(EPOCHS, training_dataloader, validation_dataloader, LEARNING_RATE)
main()
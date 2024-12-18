import torch
from torch.utils.data import DataLoader
from cupy_utils.utils import one_hot
from mlp_torch_model.model import MlpNetwork
from torchvision import transforms, datasets

def main():
    EPOCHS = 1000
    BATCH_SIZE = 2098
    IMAGE_WIDTH = 28
    IMAGE_HEIGHT = 28
    LEARNING_RATE = 0.001
    NUMBER_OF_CLASSES = 10
    LOSS_FUNCTION = torch.nn.CrossEntropyLoss()
    INPUT_DATA_FEATURE_SIZE = IMAGE_HEIGHT*IMAGE_WIDTH
    NETWORK_ARCHITECTURE = [INPUT_DATA_FEATURE_SIZE, 128, NUMBER_OF_CLASSES]
    TRANSFORM = lambda x: torch.flatten(transforms.ToTensor()(x)).type(dtype=torch.float32)
    TARGET_TRANSFORM = lambda x: torch.tensor(x)

    training_dataset = datasets.MNIST('./training-data', download=True, train=True, transform=TRANSFORM, target_transform=TARGET_TRANSFORM)
    validation_dataset = datasets.MNIST('./training-data', download=True, train=False, transform=TRANSFORM, target_transform=TARGET_TRANSFORM)
    training_dataloader = DataLoader(training_dataset, batch_size=BATCH_SIZE, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=True)

    MlpNetwork().runner(EPOCHS, training_dataloader, validation_dataloader, LOSS_FUNCTION, LEARNING_RATE)

main()

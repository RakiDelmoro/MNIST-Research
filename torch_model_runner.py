import torch
from torch.utils.data import DataLoader
from cupy_utils.utils import one_hot
from mlp_torch_model.model import MlpNetwork
from torchvision import transforms, datasets

def main():
    EPOCHS = 10
    BATCH_SIZE = 2098
    IMAGE_WIDTH = 28
    IMAGE_HEIGHT = 28
    LEARNING_RATE = 0.001
    NUMBER_OF_CLASSES = 10
    LOSS_FUNCTION = torch.nn.CrossEntropyLoss()
    INPUT_DATA_FEATURE_SIZE = IMAGE_HEIGHT*IMAGE_WIDTH
    NETWORK_ARCHITECTURE = [INPUT_DATA_FEATURE_SIZE,2000, 2000, NUMBER_OF_CLASSES]
    TRANSFORM = lambda x: torch.flatten(transforms.ToTensor()(x)).type(dtype=torch.float32)
    TARGET_TRANSFORM = lambda x: torch.tensor(one_hot(x, number_of_classes=NUMBER_OF_CLASSES), dtype=torch.float32)

    training_dataset = datasets.MNIST('./training-data', download=True, train=True, transform=TRANSFORM, target_transform=TARGET_TRANSFORM)
    validation_dataset = datasets.MNIST('./training-data', download=True, train=False, transform=TRANSFORM, target_transform=TARGET_TRANSFORM)
    training_dataloader = DataLoader(training_dataset, batch_size=BATCH_SIZE, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=True)

    MlpNetwork(NETWORK_ARCHITECTURE).runner(EPOCHS, training_dataloader, validation_dataloader, LOSS_FUNCTION, LEARNING_RATE)

main()

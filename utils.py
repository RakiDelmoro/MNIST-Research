import gzip
import pickle
import cupy as cp
import torch

def training(forward_pass, backward_pass, learning_rate):
    pass

def one_hot_encoded(expected_array):
    number_of_classes = cp.unique(expected_array).size
    one_hot_encoded = cp.eye(number_of_classes)[expected_array]

    return torch.tensor(one_hot_encoded, dtype=torch.float32)

def load_data_to_memory(file_name: str):
    with (gzip.open(file_name, 'rb')) as file:
        ((training_image_array, training_label_array), (validation_image_array, validation_label_array), _) = pickle.load(file, encoding='latin-1')

    return (torch.tensor(training_image_array, dtype=torch.float32), one_hot_encoded(cp.array(training_label_array))), (cp.array(validation_image_array, dtype=torch.float32), one_hot_encoded(torch.tensor(validation_label_array)))

def cupy_dataloader(samples, batch_size, shuffle):
    training_samples = samples[0]
    expected_samples = samples[1]

    total_samples = training_samples.shape[0]
    number_of_batched_samples = int(cp.ceil(total_samples / batch_size))

    indices = cp.arange(total_samples)

    if shuffle:
        cp.random.shuffle(indices)

    for index in range(number_of_batched_samples):
        start_index = index * batch_size
        end_index = min((index + 1) * batch_size, total_samples)
        batch_indices = indices[start_index:end_index]

        batched_images = training_samples[batch_indices]
        batched_expected = expected_samples[batch_indices]

        yield batched_images, batched_expected

import cupy as cp
from cupy_utils.utils import one_hot
from nn_utils.activation_functions import leaky_relu, softmax

def neural_network(network_size: list):
    axons = [cp.random.randn(network_size[size], network_size[size+1]) * cp.sqrt(2/network_size[size]) for size in range(len(network_size)-1)]
    dentrites = [cp.zeros(network_size[size+1]) for size in range(len(network_size)-1)]

    def forward_in_neurons(input_neurons):
        previous_neurons = input_neurons
        neurons_activations = [input_neurons]
        for neurons_layer_index in range(len(network_size)-1):
            last_layer_idx = len(network_size)-2
            if neurons_layer_index != last_layer_idx:
                neurons_activation = leaky_relu(cp.dot(previous_neurons, axons[neurons_layer_index]) + dentrites[neurons_layer_index])
            else:
                neurons_activation = softmax(cp.dot(previous_neurons, axons[neurons_layer_index]) + dentrites[neurons_layer_index])

            neurons_activations.append(neurons_activation)    
            previous_neurons = neurons_activation

        return neurons_activations
    
    def backward_in_neurons(input_neurons):
        previous_neurons = input_neurons
        neurons_activations = [input_neurons]
        for neurons_layer_index in range(len(network_size)-2):
            neurons_activation = leaky_relu(cp.dot(previous_neurons, axons[-(neurons_layer_index+1)].transpose()) + dentrites[-(neurons_layer_index+2)])
            neurons_activations.append(neurons_activation)
            previous_neurons = neurons_activation

        return neurons_activations

    def calculate_network_stress(forward_pass_activations, backward_pass_activations):
        neurons_loss_and_previous_neurons = []
        for each_neurons_activation in range(len(network_size)-1):
            neurons_loss = forward_pass_activations[each_neurons_activation+1] - backward_pass_activations[-(each_neurons_activation+1)]
            previous_neurons = forward_pass_activations[each_neurons_activation]
            neurons_loss_and_previous_neurons.append((previous_neurons, neurons_loss))
        return neurons_loss_and_previous_neurons, neurons_loss
    
    def update_axons_and_dentrites(neurons_loss_and_previous_neurons, learning_rate):
        for neurons_index, (each_neurons_axons, each_neurons_dentrites) in enumerate(zip(axons, dentrites)):
            previous_neurons = neurons_loss_and_previous_neurons[neurons_index][0]
            neurons_loss = neurons_loss_and_previous_neurons[neurons_index][1]

            each_neurons_axons -= learning_rate * cp.dot(previous_neurons.transpose(), neurons_loss)
            each_neurons_dentrites -= learning_rate * cp.sum(neurons_loss, axis=0)
    
    def training_run(training_loader, learning_rate):
        batched_network_stress = []
        for input_batch, expected_batch in training_loader:
            input_batch = cp.array(input_batch).reshape(input_batch.shape[0], -1)
            expected_batch = one_hot(x=expected_batch, number_of_classes=10)
            forward_neurons_activations = forward_in_neurons(input_batch)
            backward_neurons_activations = backward_in_neurons(expected_batch)

            # Update parameters
            neurons_loss_correspond_to_previous_neurons, network_stress = calculate_network_stress(forward_neurons_activations, backward_neurons_activations)
            update_axons_and_dentrites(neurons_loss_correspond_to_previous_neurons, learning_rate)
            batched_network_stress.append(network_stress)

        return batched_network_stress

    def test_run(validation_dataloader):
        test_loss = 0
        correct = 0
        for input_batch, expected_batch in validation_dataloader:
            input_batch = cp.array(input_batch).reshape(input_batch.shape[0], -1)
            expected_batch = one_hot(x=expected_batch, number_of_classes=10)
            model_output = forward_in_neurons(input_batch)[-1]
            test_loss += (model_output - expected_batch).item()
            model_prediction = model_output.argmax(dim=1, keepdim=True)
            correct += model_prediction.eq(expected_batch.view_as(model_prediction)).sum().item()

        test_loss /= len(validation_dataloader.dataset)
        model_accuracy = 100. * correct / len(validation_dataloader.dataset)
        print(f"Test set: Average loss: {test_loss:.4f}, Accuracy: {model_accuracy}")

    def runner(epochs, training_loader, validation_loader, learning_rate):
        for epoch in range(epochs):
            print(f'EPOCHS: {epoch+1}')
            training_run(training_loader, learning_rate)
            test_run(validation_loader)

    return runner
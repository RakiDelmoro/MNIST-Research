import cupy as cp
from features import GREEN, RED, RESET
from torch.nn.functional import one_hot
from nn_utils.loss_functions import cross_entropy_loss
from cupy_utils.utils import axons_initialization, dentrites_initialization

def neural_network(network_feature_sizes):
    layers_parameters = [[axons_initialization(network_feature_sizes[feature_idx], network_feature_sizes[feature_idx+1]), dentrites_initialization(network_feature_sizes[feature_idx+1])]
                         for feature_idx in range(len(network_feature_sizes)-1)]

    def forward_pass(input_neurons):
        neurons = cp.array(input_neurons)
        total_activations = len(layers_parameters)
        neurons_activations = [neurons]
        for each in range(total_activations):
            axons = layers_parameters[each][0]
            dentrites = layers_parameters[each][1]
            neurons = cp.dot(neurons, axons) + dentrites
            neurons_activations.append(neurons)
        return neurons_activations

    def backward_pass(model_prediction, expected_model_prediction):
        expected_model_prediction = cp.array(expected_model_prediction)
        total_layers_loss = len(layers_parameters)-1
        network_error, layer_loss = cross_entropy_loss(model_prediction, expected_model_prediction)
        layers_losses = [layer_loss]
        for each_connection in range(total_layers_loss):
            axons = layers_parameters[-(each_connection+1)][0]
            layer_loss = cp.dot(layer_loss, axons.transpose())
            layers_losses.append(layer_loss)
        return network_error, layers_losses
    
    def update_layers_parameters(neurons_activations, layers_losses, learning_rate):
        total_parameters = len(layers_parameters)
        for layer_idx in range(total_parameters):
            axons = layers_parameters[-(layer_idx+1)][0]
            dentrites = layers_parameters[-(layer_idx+1)][1]
            neuron_activation = neurons_activations[-(layer_idx+2)]
            # This line fixed the Nan problem!
            loss = layers_losses[layer_idx] / neuron_activation.shape[0]

            axons -= learning_rate * cp.dot(neuron_activation.transpose(), loss)
            dentrites -= learning_rate * cp.sum(loss, axis=0)

    def test_run_result(dataloader):
        per_batch_accuracy = []
        wrong_samples_indices = []
        correct_samples_indices = []
        model_predictions = []
        expected_model_prediction = []
        for input_image_batch, expected_batch in dataloader:
            # expected_batch = cp.array(one_hot(expected_batch, num_classes=10))
            expected_batch = cp.array(expected_batch)
            model_output = forward_pass(input_image_batch)[-1]
            batch_accuracy = cp.array(expected_batch.argmax(-1) == (model_output).argmax(-1)).mean()
            correct_indices_in_a_batch = cp.where(expected_batch.argmax(-1) == model_output.argmax(-1))[0]
            wrong_indices_in_a_batch = cp.where(~(expected_batch.argmax(-1) == model_output.argmax(-1)))[0]

            per_batch_accuracy.append(batch_accuracy.item())
            correct_samples_indices.append(correct_indices_in_a_batch)
            wrong_samples_indices.append(wrong_indices_in_a_batch)
            model_predictions.append(model_output.argmax(-1))
            expected_model_prediction.append(expected_batch.argmax(-1))

        model_accuracy = cp.mean(cp.array(per_batch_accuracy))
        correct_samples = cp.concatenate(correct_samples_indices)[list(range(0,len(correct_samples_indices)))]
        wrong_samples = cp.concatenate(wrong_samples_indices)[list(range(0,len(wrong_samples_indices)))]
        model_prediction = cp.concatenate(model_predictions)
        model_expected_prediction = cp.concatenate(expected_model_prediction)
        
        print(f"{GREEN}Model Correct Predictions{RESET}")
        for indices in correct_samples: print(f"Digit Image is: {GREEN}{model_expected_prediction[indices]}{RESET} Model Prediction: {GREEN}{model_prediction[indices]}{RESET}")
        print(f"{RED}Model Wrong Predictions{RESET}")
        for indices in wrong_samples: print(f"Digit Image is: {RED}{model_expected_prediction[indices]}{RESET} Model Predictions: {RED}{model_prediction[indices]}{RESET}")

        return model_accuracy

    def runner(training_loader, validation_loader):
        for epoch in range(10):
            print(f'EPOCH: {epoch+1}')
            per_batch_losses = []
            for input_batch, expected_batch in training_loader:
                neurons_activations = forward_pass(input_neurons=input_batch)
                error, layers_losses = backward_pass(neurons_activations[-1], expected_batch)
                per_batch_losses.append(error)
                update_layers_parameters(neurons_activations, layers_losses, 0.01)
            model_accuracy = test_run_result(validation_loader)
            print(f'Average loss per epoch: {cp.mean(cp.array(per_batch_losses))} accuracy: {model_accuracy}')

    return runner

import cupy as cp
from utils import dataloader
from features import GREEN, RED, RESET
from activation_functions import softmax
from loss_functions import cross_entropy_loss

def training(dataloader, forward_pass, backward_pass, model_parameters, update_parameters, learning_rate):
    batched_losses = []
    forward_parameters, backward_parameters = model_parameters
    for input_image_batch, expected_batch in dataloader:
        # forward and backward parameters are same value it only differ is the shape for backpropagate
        model_prediction, forward_neuron_activations = forward_pass(input_image_batch, forward_parameters)
        backward_activations = backward_pass(expected_batch, backward_parameters)
        per_batch_loss, loss_for_backprop = cross_entropy_loss(model_prediction, expected_batch)
        forward_parameters, backward_parameters = update_parameters(loss_for_backprop, model_parameters[0], learning_rate, forward_neuron_activations, backward_activations)
        batched_losses.append(per_batch_loss.item())

    # Same value the only differs is the shape
    updated_model_parameters = forward_parameters, backward_parameters

    return cp.mean(cp.array(batched_losses)).item(), updated_model_parameters

def validate(dataloader, forward_pass, model_parameters):
    per_batches_accuracy = []
    
    wrong_samples_index = []
    correct_samples_index = []

    model_predictions = []
    expected_model_prediction = []

    for input_image_batch, expected_batch in dataloader:
        model_prediction, _ = forward_pass(input_image_batch, model_parameters)
        batch_accuracy = cp.mean(expected_batch.argmax(-1) == softmax(model_prediction).argmax(-1))
        per_batch_correct_index = cp.where(expected_batch.argmax(-1) == softmax(model_prediction).argmax(-1))[0]
        per_batch_wrong_index = cp.where(~(expected_batch.argmax(-1) == softmax(model_prediction).argmax(-1)))[0]
        
        per_batches_accuracy.append(batch_accuracy.item())
        
        correct_samples_index.append(per_batch_correct_index)
        wrong_samples_index.append(per_batch_wrong_index)
        
        expected_model_prediction.append(expected_batch.argmax(-1))
        model_predictions.append(softmax(model_prediction).argmax(-1))

    correct_samples_index = cp.concatenate(correct_samples_index)[:5].tolist()
    wrong_samples_index = cp.concatenate(wrong_samples_index)[:5].tolist()
    model_expected_prediction = cp.concatenate(expected_model_prediction)
    model_prediction_index = cp.concatenate(model_predictions)  

    print(f"{GREEN}Model Correct Predictions{RESET}")
    for index in correct_samples_index:
        print(f"Digit image is: {GREEN}{model_expected_prediction[index]}{RESET} Model prediction: {GREEN}{model_prediction_index[index]}{RESET}")

    print(f"{RED}Model Wrong Predictions{RESET}")
    for index in wrong_samples_index:
        print(f"Digit image is: {RED}{model_expected_prediction[index]}{RESET} Model prediction: {RED}{model_prediction_index[index]}{RESET}")

    return cp.mean(cp.array(per_batches_accuracy)).item()

def runner(epochs, for_training_array_of_tuple, for_validation_array_of_tuple, shuffle, batch_size, update_parameters, forward_pass, backward_pass, model_parameters, learning_rate):
    for epoch in range(epochs):
        training_dataloader = dataloader(for_training_array_of_tuple, batch_size, shuffle)
        validation_dataloader = dataloader(for_validation_array_of_tuple, batch_size, shuffle)
        average_loss_for_all_batch, model_parameters = training(training_dataloader, forward_pass, backward_pass, model_parameters, update_parameters, learning_rate)
        average_accuracy_for_all_batch = validate(validation_dataloader, forward_pass, model_parameters[0])
        print(f"EPOCH: {epoch+1} Training loss: {average_loss_for_all_batch} Model Accuracy: {average_accuracy_for_all_batch}")

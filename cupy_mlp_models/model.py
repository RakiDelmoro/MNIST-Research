
def cupy_mlp_neural_network(network_parameters, residual_neurons_sizes, training_loader, validation_loader, training_layers, test_layers, learning_rate, epochs):
    residual_use = residual_neurons_sizes is not None
    def training_run():
        if residual_use:
            return training_layers(dataloader=training_loader, layers_parameters=network_parameters, residual_neurons_sizes=residual_neurons_sizes, learning_rate=learning_rate)
        else:
            return training_layers(dataloader=training_loader, layers_parameters=network_parameters, learning_rate=learning_rate)

    def test_run():
        if residual_use:
            return test_layers(dataloader=validation_loader, layers_parameters=network_parameters, residual_idx=residual_neurons_sizes)
        else:
            return test_layers(dataloader=validation_loader, layers_parameters=network_parameters)

    for epoch in range(epochs):
        print(f'EPOCH: {epoch+1}')
        model_stress = training_run()
        model_accuracy = test_run()
        # print(f'accuracy: {model_accuracy}')
        print(f'Average loss per epoch: {model_stress} accuracy: {model_accuracy}')

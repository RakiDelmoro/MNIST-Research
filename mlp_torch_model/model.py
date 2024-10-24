import torch
from torch import nn
from torch.nn import functional
from torch.nn.functional import softmax
from features import GREEN, RED, RESET

class MlpNetwork(torch.nn.Module):
    def __init__(self, network_architecture: list):
        super().__init__()
        self.device = "cuda"
        self.network_layers = nn.ModuleList(
            nn.Sequential(
                nn.Linear(network_architecture[idx], network_architecture[idx+1], bias=False, device='cuda'),
                # nn.LeakyReLU() if idx != len(network_architecture)-2 else nn.Softmax(dim=-1)
            )
            for idx in range(len(network_architecture)-1)
        )

    def forward(self, batch_data):
        previous_neurons = batch_data.flatten(1, -1)
        for layer in self.network_layers:
            previous_neurons = layer(previous_neurons)

        return previous_neurons

    def training_run(self, training_loader, loss_function, learning_rate):
        optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)
        per_batch_loss = []
        for input_batch, expected_batch in training_loader:
            input_batch = input_batch.to(self.device)
            expected_batch = functional.one_hot(expected_batch, num_classes=10).float().to(self.device)
            model_prediction = self.forward(input_batch)
            loss = loss_function(model_prediction, expected_batch)
            per_batch_loss.append(loss.item())
            # Update Parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return torch.mean(torch.tensor(per_batch_loss)).item()

    def test_run(self, dataloader):
        per_batch_accuracy = []
        wrong_samples_indices = []
        correct_samples_indices = []
        model_predictions = []
        expected_model_prediction = []
        for input_image_batch, expected_batch in dataloader:
            input_image_batch = input_image_batch.to(self.device)
            expected_batch = functional.one_hot(expected_batch, num_classes=10).float().to(self.device)
            model_output = self.forward(input_image_batch)
            batch_accuracy = (expected_batch.argmax(-1) == (model_output).argmax(-1)).float().mean()
            correct_indices_in_a_batch = torch.where(expected_batch.argmax(-1) == model_output.argmax(-1))[0]
            wrong_indices_in_a_batch = torch.where(~(expected_batch.argmax(-1) == model_output.argmax(-1)))[0]

            per_batch_accuracy.append(batch_accuracy.item())
            correct_samples_indices.append(correct_indices_in_a_batch)
            wrong_samples_indices.append(wrong_indices_in_a_batch)
            model_predictions.append(model_output.argmax(-1))
            expected_model_prediction.append(expected_batch.argmax(-1))

        correct_samples = torch.concatenate(correct_samples_indices)[list(range(0,len(correct_samples_indices)))]
        wrong_samples = torch.concatenate(wrong_samples_indices)[list(range(0,len(wrong_samples_indices)))]
        model_prediction = torch.concatenate(model_predictions)
        model_expected_prediction = torch.concatenate(expected_model_prediction)

        print(f"{GREEN}Model Correct Predictions{RESET}")
        for indices in correct_samples: print(f"Digit Image is: {GREEN}{model_expected_prediction[indices]}{RESET} Model Prediction: {GREEN}{model_prediction[indices]}{RESET}")
        print(f"{RED}Model Wrong Predictions{RESET}")
        for indices in wrong_samples: print(f"Digit Image is: {RED}{model_expected_prediction[indices]}{RESET} Model Predictions: {RED}{model_prediction[indices]}{RESET}")

        return torch.mean(torch.tensor(per_batch_accuracy)).item()

    def runner(self, epochs, training_loader, validation_loader, loss_function, learning_rate):
        for epoch in range(epochs):
            # Training
            training_loss = self.training_run(training_loader, loss_function, learning_rate)
            # Test
            accuracy = self.test_run(validation_loader)
            print(f"EPOCH: {epoch+1} Training Loss: {training_loss} Model Accuracy: {accuracy}")
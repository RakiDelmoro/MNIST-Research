import torch
from torch.nn import functional

class MlpNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.device = "cuda"

        self.input_to_hidden_1 = torch.nn.Linear(784, 2000, device=self.device)
        self.hidden_1_to_hidden_2 = torch.nn.Linear(2000, 2000, device=self.device)
        self.hidden_2_to_output = torch.nn.Linear(2000, 10, device=self.device)
        self.activation_function = torch.nn.LeakyReLU()

    def forward(self, batch_data):
        hidden_1 = self.input_to_hidden_1(batch_data.flatten(1, -1))
        hidden_1_activated = self.activation_function(hidden_1)
        hidden_2 = self.activation_function(self.hidden_1_to_hidden_2(hidden_1_activated))
        output = self.hidden_2_to_output(hidden_2)

        return output
    
    def training_run(self, training_loader, loss_function, learning_rate):
        optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)
        for input_batch, expected_batch in training_loader:
            input_batch = input_batch.to(self.device)
            expected_batch = expected_batch.to(self.device)
            model_prediction = self.forward(input_batch)
            loss = loss_function(model_prediction, expected_batch)
            # Update Parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def test_run(self, test_loader):
        self.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.forward(data)
                test_loss += functional.nll_loss(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        model_accuracy = 100. * correct / len(test_loader.dataset)
        print(f"Test set: Average loss: {test_loss:.4f}, Accuracy: {model_accuracy}")

    def runner(self, epochs, training_loader, validation_loader, loss_function, learning_rate):
        for epoch in range(epochs):
            print(f'EPOCHS: {epoch+1}')
            # Training
            self.training_run(training_loader, loss_function, learning_rate)
            # Test
            self.test_run(validation_loader)

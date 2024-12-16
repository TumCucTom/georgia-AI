import torch

class WordPredictor(nn.Module):
    def __init__(self):
        super(WordPredictor, self).__init__()

        # Define your hidden layers
        self.fc1 = nn.Linear(300, 128)  # First hidden layer (300 input -> 512 output)
        self.fc2 = nn.Linear(128, 64)  # Second hidden layer (512 input -> 256 output)
        self.fc3 = nn.Linear(64, 130)  # Output layer (256 input -> 130 output)

        # Activation functions
        self.relu = nn.ReLU()

    def forward(self, x):
        # Pass through hidden layers with ReLU activation
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))

        # Output layer: Linear activation (softmax will be applied in loss function)
        x = self.fc3(x)

        # Reshape output to 5x26 (5 positions, 26 possible letters)
        x = x.view(-1, 5, 26)  # This will give you the shape [batch_size, 5, 26]

        return x

model = torch.jit.load('model_scripted.pt')
model.eval()
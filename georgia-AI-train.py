import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Load the dataset, split into input (X) and output (y) variables
dataset = np.loadtxt('data/NN-training-data.csv', delimiter=',')
X = dataset[:,0:299]  # 300 inputs
y = dataset[:,299:429]  # 130 output (should correspond to 5x26 letters)

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)  # Use long type for class indices (for CrossEntropyLoss)

class WordPredictor(nn.Module):
    def __init__(self):
        super(WordPredictor, self).__init__()

        # Define your hidden layers
        self.fc1 = nn.Linear(300, 512)  # First hidden layer (300 input -> 512 output)
        self.fc2 = nn.Linear(512, 256)  # Second hidden layer (512 input -> 256 output)
        self.fc3 = nn.Linear(256, 130)  # Output layer (256 input -> 130 output)

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

# Create an instance of the model
model = WordPredictor()
print(model)

# Train the model
loss_fn = nn.CrossEntropyLoss()  # Cross entropy loss expects raw logits
optimizer = optim.Adam(model.parameters(), lr=0.001)

n_epochs = 100
batch_size = 10

for epoch in range(n_epochs):
    for i in range(0, len(X), batch_size):
        Xbatch = X[i:i+batch_size]
        ybatch = y[i:i+batch_size]
        optimizer.zero_grad()

        # Forward pass
        y_pred = model(Xbatch)

        # Reshape target to match output: [batch_size, 5] of indices (not one-hot encoded)
        ybatch_reshaped = ybatch.view(-1, 5)  # Reshape to 5 (letters per word)

        # Compute loss (CrossEntropyLoss will apply softmax internally)
        loss = loss_fn(y_pred.view(-1, 26), ybatch_reshaped.view(-1))  # Flatten the output for CrossEntropyLoss
        loss.backward()
        optimizer.step()

    print(f'Finished epoch {epoch}, latest loss {loss}')

# Compute accuracy (no_grad is optional)
with torch.no_grad():
    y_pred = model(X)
    y_pred_labels = y_pred.argmax(dim=-1)  # Get the predicted letter index (5 letters per word)
    accuracy = (y_pred_labels == y.view(-1, 5)).float().mean()  # Compare predicted vs actual labels
print(f"Accuracy {accuracy.item()}")

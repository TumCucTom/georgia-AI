import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Load the dataset, split into input (X) and output (y) variables
dataset = np.loadtxt('data/NN-training-data.csv', delimiter=',')
X = dataset[:,0:300]  # 300 inputs
y = dataset[:,300:430]  # 130 output (should correspond to 5x26 letters)

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
        y_pred = model(Xbatch)  # Shape: [batch_size, 5, 26]

        # Reshape ybatch to represent 5 positions per word (batch_size, 5, num_classes)
        # Here ybatch originally has shape [10, 130] (1 letter per word),
        # but we need it in a shape of [10, 5, 26] (5 letters per word, 26 possible letters)

        ybatch_reshaped = ybatch.view(-1, 5, 26)  # Reshape to [batch_size, 5, num_classes]

        # Now apply argmax along the last dimension (num_classes) to get the indices of predicted letters
        ybatch_indices = ybatch_reshaped.argmax(dim=-1)  # Shape: [batch_size, 5]

        # Flatten ybatch_indices and y_pred for CrossEntropyLoss
        ybatch_reshaped = ybatch_indices.view(-1)  # Flatten to [batch_size * 5] (e.g., [50])

        # Reshape predictions as well
        y_pred_reshaped = y_pred.view(-1, 26)  # Flatten y_pred to [batch_size * 5, num_classes]

        # Compute loss (CrossEntropyLoss will apply softmax internally)
        loss = loss_fn(y_pred_reshaped, ybatch_reshaped)  # CrossEntropyLoss expects flattened predictions and targets
        loss.backward()
        optimizer.step()

    print(f'Finished epoch {epoch}, latest loss {loss}')

# Compute accuracy (no_grad is optional)
with torch.no_grad():
    y_pred = model(X)
    y_pred_labels = y_pred.argmax(dim=-1)  # Get the predicted letter index (5 letters per word)

    # Reshape y to [batch_size, 5, 26] and then apply argmax to get the indices
    y_reshaped = y.view(-1, 5, 26)  # Reshape y to [batch_size, 5, 26]
    y_indices = y_reshaped.argmax(dim=-1)  # Get indices (shape: [batch_size, 5])

    # Compare predicted vs actual labels
    accuracy = (y_pred_labels == y_indices).float().mean()  # Compare predicted vs actual labels

print(f"Accuracy {accuracy.item()}")



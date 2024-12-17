import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Load the dataset, split into input (X) and output (y) variables
dataset = np.loadtxt('../data/expanded/NN-training-data.csv', delimiter=',')
X = dataset[:,0:900]  # 300 inputs
y = dataset[:,900:1030]  # 130 output (should correspond to 5x26 letters)

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)  # Use long type for class indices (for CrossEntropyLoss)

class ImprovedWordPredictor(nn.Module):
    def __init__(self):
        super(ImprovedWordPredictor, self).__init__()

        # Define layers
        self.fc1 = nn.Linear(900, 1024)  # Increased the number of neurons in the first layer
        self.fc2 = nn.Linear(1024, 512)  # Increased the number of neurons in the second layer
        self.fc3 = nn.Linear(512, 256)   # Adding another layer for more complexity
        self.fc4 = nn.Linear(256, 130)   # Output layer

        # Batch normalization
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(256)

        # Dropout layers
        self.dropout = nn.Dropout(p=0.3)

        # Activation functions
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)  # Dropout for regularization
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.relu(self.bn3(self.fc3(x)))
        x = self.fc4(x)  # Final output without activation (softmax will be in the loss)

        # Reshape output to 5x26 (for classification of 5 letters with 26 possibilities)
        x = x.view(-1, 5, 26)

        return x


# Create an instance of the model
model = ImprovedWordPredictor()
print(model)

# Train the model
loss_fn = nn.CrossEntropyLoss()  # Cross entropy loss expects raw logits
optimizer = optim.Adam(model.parameters(), lr=0.001)

n_epochs = 100
batch_size = 10

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

for epoch in range(n_epochs):
    for i in range(0, len(X), batch_size):
        Xbatch = X[i:i+batch_size]
        ybatch = y[i:i+batch_size]
        optimizer.zero_grad()

        # Forward pass
        y_pred = model(Xbatch)

        # Reshape ybatch to represent 5 positions per word (batch_size, 5, num_classes)
        ybatch_reshaped = ybatch.view(-1, 5, 26)
        ybatch_indices = ybatch_reshaped.argmax(dim=-1)
        ybatch_reshaped = ybatch_indices.view(-1)
        y_pred_reshaped = y_pred.view(-1, 26)

        loss = loss_fn(y_pred_reshaped, ybatch_reshaped)
        loss.backward()
        optimizer.step()

    # Step the learning rate scheduler
    scheduler.step(loss)
    print(f'Finished epoch {epoch}, latest loss {loss}')



# Parameters
input_size = 900  # Number of input features
output_size = 130  # Number of output features (5 positions x 26 letters)
num_positions = 5  # Number of letter positions in the output
num_classes = 26  # Number of possible letters per position
csv_file = "../data/expanded/MINI-TEST.csv"  # Path to your CSV file

# Load dataset from CSV
data = np.genfromtxt(csv_file, delimiter=',')
if data.ndim == 1:  # If the file has a single row, reshape it
    data = data.reshape(1, -1)

# Split into input and output
X = data[:, :input_size]  # First `input_size` columns for input features
y = data[:, input_size:]  # Remaining columns for output labels

# Convert to PyTorch tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)  # Keep float for one-hot encoding

# Compute accuracy (no_grad is optional)
with torch.no_grad():
    y_pred = model(X)
    y_pred_labels = y_pred.argmax(dim=-1)  # Get the predicted letter index (5 letters per word)

    # Reshape y to [batch_size, 5, 26] and then apply argmax to get the indices
    y_reshaped = y.view(-1, 5, 26)  # Reshape y to [batch_size, 5, 26]
    y_indices = y_reshaped.argmax(dim=-1)  # Get indices (shape: [batch_size, 5])

    ind_string = [[]]
    pred_string = [[]]

    # compare
    for each in y_indices:
        letters = []
        for letter in each:
            letters.append(chr(97+letter))
        ind_string.append(letters)

    for each in y_pred_labels:
        letters = []
        for letter in each:
            letters.append(chr(97+letter))
        pred_string.append(letters)

    # Compare predicted vs actual labels
    accuracy = (y_pred_labels == y_indices).float().mean()  # Compare predicted vs actual labels

'''
for i in range(len(ind_string)):
    print(ind_string[i])
    print(pred_string[i])
    print("\n")

'''

print("Accuracy: ", accuracy)

model_scripted = torch.jit.script(model) # Export to TorchScript
model_scripted.save('model_scripted.pt') # Save
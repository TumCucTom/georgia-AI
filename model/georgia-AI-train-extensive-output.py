import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Load the dataset, split into input (X) and output (y) variables
dataset = np.loadtxt('../data/expanded/NN-training-data-not-one-hot.csv', delimiter=',')
X = dataset[:, 0:900]  # 900 inputs
y = dataset[:, 900]  # Single integer class index per sample (5757 classes)

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)  # Use long type for class indices (for CrossEntropyLoss)

# Define the WordPredictor model
class WordPredictor(nn.Module):
    def __init__(self):
        super(WordPredictor, self).__init__()

        # Layers
        self.fc1 = nn.Linear(900, 1024)
        self.bn1 = nn.BatchNorm1d(1024)  # Batch normalization after first layer

        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)  # Batch normalization after second layer

        self.fc3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)  # Batch normalization after third layer

        self.fc4 = nn.Linear(256, 5757)  # Output layer (5757 classes)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)

        # Activation function
        self.relu = nn.ReLU()

        # Weight initialization
        self._initialize_weights()

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.relu(self.bn3(self.fc3(x)))
        x = self.fc4(x)  # Logits (no activation; CrossEntropyLoss applies softmax)
        return x

    def _initialize_weights(self):
        # Apply Xavier initialization to all linear layers
        for layer in [self.fc1, self.fc2, self.fc3, self.fc4]:
            torch.nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                torch.nn.init.zeros_(layer.bias)

# Create an instance of the model
model = WordPredictor()
print(model)

# Training setup
loss_fn = nn.CrossEntropyLoss()  # Cross entropy loss expects raw logits
optimizer = optim.Adam(model.parameters(), lr=0.001)

n_epochs = 20  # Reduced epochs for batch size 10
batch_size = 10  # Batch size of 10

# Training loop
for epoch in range(n_epochs):
    for i in range(0, len(X), batch_size):
        Xbatch = X[i:i+batch_size]
        ybatch = y[i:i+batch_size]

        optimizer.zero_grad()

        # Forward pass
        y_pred = model(Xbatch)  # Shape: [batch_size, 5757]

        # Compute loss (ybatch should contain class indices)
        loss = loss_fn(y_pred, ybatch)
        loss.backward()
        optimizer.step()

    print(f'Finished epoch {epoch+1}/{n_epochs}, latest loss {loss.item()}')

# Parameters
input_size = 900
output_size = 5757
csv_file = "../data/expanded/MINI-TEST-extensive.csv"

# Load dataset for testing
data = np.genfromtxt(csv_file, delimiter=',')
if data.ndim == 1:  # If the file has a single row, reshape it
    data = data.reshape(1, -1)

# Split into input and output
X_test = data[:, :input_size]
y_test = data[:, input_size]  # Target class indices (5757 words)

# Convert to PyTorch tensors
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

# Load the list of words from the file
with open('../data/5-letter-words.txt', 'r') as f:
    word_list = [line.strip() for line in f]  # Strip newline characters

# Ensure the word list size matches the output size
assert len(word_list) == 5757, "Word list size does not match the number of output classes!"

# Compute accuracy and print predictions vs actual words
with torch.no_grad():
    y_pred = model(X_test)
    y_pred_indices = y_pred.argmax(dim=1)  # Predicted class indices

    total_letter_matches = 0
    total_letters = 5 * len(y_test)  # Total number of letters (5 letters per word)

    print("Predicted Word vs Actual Word (with letter accuracy):")
    for i in range(len(y_test)):
        predicted_word = word_list[y_pred_indices[i].item()]  # Get the predicted word
        actual_word = word_list[y_test[i].item()]            # Get the actual word

        # Count letter matches
        letter_matches = sum(p == a for p, a in zip(predicted_word, actual_word))
        total_letter_matches += letter_matches

        # Print details
        print(f"Sample {i+1}: Predicted = {predicted_word}, Actual = {actual_word}, Matching Letters = {letter_matches}/5")

    # Calculate letter-level accuracy
    letter_accuracy = (total_letter_matches / total_letters) * 100
    print(f"\nLetter-Level Accuracy: {letter_accuracy:.2f}%")

# Save the model
model_scripted = torch.jit.script(model)
model_scripted.save('model_scripted_e.pt')

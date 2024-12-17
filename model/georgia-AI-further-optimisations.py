import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Load the dataset
dataset = np.loadtxt('../data/expanded/NN-training-data-not-one-hot.csv', delimiter=',')
X = dataset[:, 0:900]  # 900 inputs
y = dataset[:, 900]    # Single integer class index per sample (5757 classes)

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

# Define the improved WordPredictor model
class WordPredictor(nn.Module):
    def __init__(self):
        super(WordPredictor, self).__init__()

        # First block: input to wide hidden layer
        self.fc1 = nn.Linear(900, 2048)
        self.ln1 = nn.LayerNorm(2048)

        # Second block: hidden layer with skip connection
        self.fc2 = nn.Linear(2048, 1024)
        self.ln2 = nn.LayerNorm(1024)

        # Third block: another hidden layer
        self.fc3 = nn.Linear(1024, 512)
        self.ln3 = nn.LayerNorm(512)

        # Fourth block: deeper hidden layer with residual connection
        self.fc4 = nn.Linear(512, 256)
        self.ln4 = nn.LayerNorm(256)

        # Output layer
        self.fc_out = nn.Linear(256, 5757)

        # Dropout
        self.dropout = nn.Dropout(0.2)

        # Activation
        self.swish = nn.SiLU()  # Swish activation for smoother gradients

        self.fc_proj = nn.Linear(2048, 1024)  # Projection layer for the first residual connection
        self.fc_proj2 = nn.Linear(512, 256)   # Projection layer for the second residual connection


# Initialize weights
        self._initialize_weights()

    def forward(self, x):
        # Block 1
        x1 = self.swish(self.ln1(self.fc1(x)))
        x1 = self.dropout(x1)

        # Block 2 with skip connection
        x2 = self.swish(self.ln2(self.fc2(x1)))
        x2 = self.dropout(x2)

        # Projection layer to match dimensions of x2
        x1_proj = self.fc_proj(x1)  # Project x1 to 1024 units
        x2 = x2 + x1_proj           # Residual connection

        # Block 3
        x3 = self.swish(self.ln3(self.fc3(x2)))
        x3 = self.dropout(x3)

        # Block 4 with another residual connection
        x4 = self.swish(self.ln4(self.fc4(x3)))
        x4 = self.dropout(x4)

        # Projection layer to match dimensions of x4
        x3_proj = self.fc_proj2(x3)  # Project x3 to 256 units
        x4 = x4 + x3_proj            # Residual connection

        # Output
        logits = self.fc_out(x4)
        return logits


    def _initialize_weights(self):
        # Use He initialization for better weight initialization
        for layer in [self.fc1, self.fc2, self.fc3, self.fc4, self.fc_out]:
            nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

# Create an instance of the model
model = WordPredictor()
print(model)

# Loss and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)  # Reduced learning rate for stability

n_epochs = 50  # Reduced epochs
batch_size = 10

# Training loop
for epoch in range(n_epochs):
    for i in range(0, len(X), batch_size):
        Xbatch = X[i:i+batch_size]
        ybatch = y[i:i+batch_size]

        optimizer.zero_grad()
        y_pred = model(Xbatch)
        loss = loss_fn(y_pred, ybatch)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss.item():.4f}")

# Load word list
with open('../data/5-letter-words.txt', 'r') as f:
    word_list = [line.strip() for line in f]

# Testing accuracy (letter-level)
data = np.genfromtxt("../data/expanded/MINI-TEST-extensive.csv", delimiter=',')
if data.ndim == 1:  # Reshape single row
    data = data.reshape(1, -1)

X_test = torch.tensor(data[:, :900], dtype=torch.float32)
y_test = torch.tensor(data[:, 900], dtype=torch.long)

with torch.no_grad():
    y_pred = model(X_test).argmax(dim=1)

    total_letters = 0
    correct_letters = 0

    print("\nPredicted vs Actual Words (letter accuracy):")
    for i in range(len(y_test)):
        predicted_word = word_list[y_pred[i].item()]
        actual_word = word_list[y_test[i].item()]
        matches = sum(1 for p, a in zip(predicted_word, actual_word) if p == a)

        correct_letters += matches
        total_letters += len(actual_word)

        print(f"Sample {i+1}: Predicted = {predicted_word}, Actual = {actual_word}, Matching Letters = {matches}/5")

    letter_accuracy = (correct_letters / total_letters) * 100
    print(f"\nLetter-Level Accuracy: {letter_accuracy:.2f}%")

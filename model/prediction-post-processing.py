import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ImprovedWordPredictor(nn.Module):
    def __init__(self):
        super(ImprovedWordPredictor, self).__init__()

        # Define layers
        self.fc1 = nn.Linear(900, 2048)  # Increased the number of neurons in the first layer
        self.fc2 = nn.Linear(2048, 1024)  # Increased the number of neurons in the second layer
        self.fc3 = nn.Linear(1024, 512)   # Adding another layer for more complexity
        self.fc4 = nn.Linear(512, 130)    # Output layer

        # Batch normalization
        self.bn1 = nn.BatchNorm1d(2048)
        self.bn2 = nn.BatchNorm1d(1024)
        self.bn3 = nn.BatchNorm1d(512)

        # Dropout layers
        self.dropout = nn.Dropout(p=0.4)

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

def load_dictionary(file_path):
    with open(file_path, 'r') as f:
        words = set(f.read().splitlines())  # Read each word in the file and store it in a set for quick lookup
    return words

def beam_search_with_sampling(model, input_data, dictionary, beam_width=5, max_sequence_length=5, temperature=1.0, epsilon=1e-10):
    # Generate initial probability distributions from the model
    input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        probabilities = model(input_tensor)  # Shape: [1, 5, 26]

    # Beam search initialization: (probability, sequence)
    sequences = [([], 0)]  # Start with an empty sequence and zero probability

    # Perform beam search
    for i in range(max_sequence_length):  # 5 positions
        all_candidates = []
        for seq, score in sequences:
            for j in range(26):  # Loop over all possible letters (26 possibilities)
                candidate = seq + [j]  # Add the letter index to the sequence
                candidate_prob = probabilities[0, i, j].item()

                # Apply temperature scaling to the probabilities
                adjusted_prob = candidate_prob ** (1.0 / temperature)

                # Avoid log(0) by ensuring the adjusted probability is never zero or negative
                adjusted_prob = max(adjusted_prob, epsilon)

                # Calculate cumulative score using log-probabilities
                candidate_score = score + np.log(adjusted_prob)

                all_candidates.append((candidate, candidate_score))

        # Sort all candidates based on their score and select top `beam_width`
        sequences = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_width]

        # Prune sequences that form invalid words based on the dictionary
        valid_sequences = []
        for seq, score in sequences:
            word = ''.join(chr(65 + letter) for letter in seq)  # Convert indices to characters
            if word in dictionary:  # Check if the word is valid
                valid_sequences.append((seq, score))

        # If no valid sequence, fall back to the top sequences
        if not valid_sequences:
            sequences = sequences[:beam_width]
        else:
            sequences = valid_sequences

    # Get the best sequence (the one with the highest probability score)
    best_sequence = sequences[0][0]  # Choose the sequence with the highest score

    return sequences[:beam_width], best_sequence

# Load the model
model = torch.jit.load('model_scripted.pt')
model.eval()

# Load the dictionary of valid 5-letter words
dictionary = load_dictionary('../data/5-letter-words.txt')

# Read the input data from the CSV file
input_data = np.loadtxt('../data/input.csv', delimiter=',')  # Multiple rows, 900 columns each
input_data = input_data.astype(np.float32)  # Convert to float32 if needed

# Open the output file for writing the results
output_file_path = '../data/results/beam-search-dict.txt'
with open(output_file_path, 'w') as output_file:

    # Iterate over each input row in the file and run beam search
    for idx, row in enumerate(input_data):
        print(f"\nProcessing input row {idx + 1}:")
        output_file.write(f"\nProcessing input row {idx + 1}:\n")  # Write to file

        # Run beam search to get the most probable letter sequence with sampling
        beam_output, best_beam_word = beam_search_with_sampling(model, row, dictionary, beam_width=10, max_sequence_length=5, temperature=1.0)

        # Convert the output letter indices to actual characters (A=0, B=1, ..., Z=25)
        predicted_word = ''.join(chr(65 + letter) for letter in best_beam_word)

        # Get the network's argmax-based prediction
        with torch.no_grad():
            input_tensor = torch.tensor(row, dtype=torch.float32).unsqueeze(0)
            probabilities = model(input_tensor)  # Shape: [1, 5, 26]
            predicted_indices = probabilities.argmax(dim=-1)  # Get the predicted letter index (5 letters per word)
            predicted_word_argmax = ''.join(chr(65 + idx.item()) for idx in predicted_indices[0])

        # Print out the 10 best words from beam search
        print("Top 10 words from beam search:")
        output_file.write("Top 10 words from beam search:\n")  # Write to file
        for idx, (seq, score) in enumerate(beam_output):
            word = ''.join(chr(65 + letter) for letter in seq)
            print(f"{idx + 1}: {word} (Score: {score})")
            output_file.write(f"{idx + 1}: {word} (Score: {score})\n")  # Write to file

        # Print the word predicted by the network using argmax
        print(f"\nNetwork's prediction (argmax): {predicted_word_argmax}")
        output_file.write(f"\nNetwork's prediction (argmax): {predicted_word_argmax}\n")  # Write to file

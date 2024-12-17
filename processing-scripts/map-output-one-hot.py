import csv
import re

def letter_to_one_hot(letter):
    """Converts a letter to its one-hot encoded representation (26 length)"""
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    index = alphabet.index(letter.lower())
    one_hot = [0] * 26
    one_hot[index] = 1
    return one_hot

def word_to_one_hot(word):
    """Converts a 5-letter word to a list of 130 comma-separated variables"""
    if len(word) != 5:
        raise ValueError(f"Expected a 5-letter word, got {len(word)} letters instead.")

    one_hot_representation = []

    for char in word:
        one_hot_representation.extend(letter_to_one_hot(char))

    return one_hot_representation

def process_csv(input_file, output_file):
    """Reads the input CSV file, processes the words, and writes the output CSV file"""
    with open(input_file, mode='r', newline='', encoding='utf-8') as infile, \
            open(output_file, mode='w', newline='', encoding='utf-8') as outfile:

        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        for row in reader:
            new_row = []
            for item in row:
                # Check if the item is a 5-letter word
                if len(item) == 5 and item.isalpha():
                    # Replace the 5-letter word with its one-hot encoded representation
                    new_row.append(','.join(map(str, word_to_one_hot(item))))
                else:
                    # Keep the other content unchanged
                    new_row.append(item)
            writer.writerow(new_row)

# Usage
input_file = '../data/message-data/NN-training-data-no-output.csv'  # Replace with your input file name
output_file = '../data/training-data/train/NN-training-data.csv'  # Replace with your desired output file name
process_csv(input_file, output_file)

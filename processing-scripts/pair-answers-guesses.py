import re

# File paths
word_list_file = "../data/answers.txt"  # First file with words and Wordle numbers
wordle_data_file = "../data/pre-processed-messages.txt"  # Second file with Wordle numbers and grids
output_file = "../data/pre-processed-data.txt"  # Output file

# Dictionary to map Wordle numbers to their words
wordle_words = {}

# Load the word list and populate the dictionary
with open(word_list_file, "r", encoding="utf-8") as file:
    for line in file:
        match = re.match(r"(\d+)\.\s+(\w+)", line)
        if match:
            wordle_number = int(match.group(1))
            word = match.group(2)
            wordle_words[wordle_number] = word

# Process the second file and add the word after the Wordle number
with open(wordle_data_file, "r", encoding="utf-8") as input_file, open(output_file, "w", encoding="utf-8") as output_file:
    for line in input_file:
        match = re.match(r"Wordle Number: (\d+)", line)
        if match:
            wordle_number = int(match.group(1))
            word = wordle_words.get(wordle_number, "UNKNOWN")
            output_file.write(f"Wordle Number: {wordle_number} ({word})\n")
        else:
            output_file.write(line)

print(f"Processed data saved to {output_file}")

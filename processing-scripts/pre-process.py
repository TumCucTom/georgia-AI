import re
from collections import defaultdict

# Input and output file paths
input_file = "../data/message-data/wordle-messages.txt"
output_file = "../data/message-data/pre-processed-messages.txt"

# Regular expressions
wordle_number_pattern = re.compile(r"Wordle (\d{1,3}(,\d{3})*?) \d/6")  # Extract Wordle number
grid_pattern = re.compile(r"[⬛⬜🟨🟩]")  # Matches Wordle squares

# Mapping Wordle squares to numeric values
square_mapping = {
    "🟩": "2",  # Green
    "🟨": "1",  # Yellow
    "⬛": "0",  # Black
    "⬜": "0",  # White
}

# Dictionary to group grids by Wordle number
wordle_by_number = defaultdict(list)

# Read the input file
with open(input_file, "r", encoding="utf-8") as file:
    current_wordle_number = None
    current_message = []

    for line in file:
        wordle_match = wordle_number_pattern.search(line)
        if wordle_match:
            # Extract Wordle number
            current_wordle_number = wordle_match.group(1).replace(",", "")  # Remove commas from number
            if current_message and current_wordle_number:
                wordle_by_number[current_wordle_number].append("".join(current_message).strip())
                current_message = []
        if current_wordle_number:
            current_message.append(line)

    # Add the last message if it exists
    if current_message and current_wordle_number:
        wordle_by_number[current_wordle_number].append("".join(current_message).strip())

# Transform squares to numeric values and prepare output
with open(output_file, "w", encoding="utf-8") as file:
    for wordle_number, messages in sorted(wordle_by_number.items(), key=lambda x: int(x[0])):
        file.write(f"Wordle Number: {wordle_number}\n")
        transformed_messages = []
        for message in messages:
            lines = message.splitlines()
            transformed_message = []
            for line in lines:
                if grid_pattern.search(line):
                    # Convert grid line to numeric values
                    transformed_line = "".join(square_mapping[char] for char in grid_pattern.findall(line))
                    transformed_message.append(transformed_line)
            # Join the transformed grid lines
            transformed_messages.append("\n".join(transformed_message))
        # Separate messages within the same Wordle number group with a blank line before and after "/"
        file.write("\n\n".join(transformed_messages) + "\n\n")

print(f"Processed Wordle data saved to {output_file}")

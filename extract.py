import re

# File path to the exported WhatsApp chat
file_path = "all-messages.txt"

# Regular expression to match "Wordle y x/6" where y can have commas
pattern = re.compile(r"Wordle (\d{1,3}(,\d{3})*?) (\d)/6")

# List to store matching messages
wordle_messages = []

# Read the chat file
with open(file_path, "r", encoding="utf-8") as file:
    for line in file:
        if pattern.search(line):
            wordle_messages.append(line.strip())

# Save the extracted messages to a new file
output_file = "wordle_messages.txt"
with open(output_file, "w", encoding="utf-8") as file:
    file.write("\n".join(wordle_messages))

print(f"Extracted Wordle messages saved to {output_file}")

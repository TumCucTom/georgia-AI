import re

# File path to the exported WhatsApp chat
file_path = "all-messages.txt"

# Regular expression to match the start of a new message with a timestamp
new_message_pattern = re.compile(r"^\[\d{2}/\d{2}/\d{4}, \d{2}:\d{2}:\d{2}\]")

# Regular expression to match the "Wordle y x/6" pattern
wordle_pattern = re.compile(r"Wordle (\d{1,3}(,\d{3})*?) (\d)/6")

# List to store matching messages
wordle_messages = []

# Read the chat file
with open(file_path, "r", encoding="utf-8") as file:
    lines = file.readlines()

# Extract multi-line Wordle messages
current_message = []
is_wordle_message = False

for line in lines:
    if new_message_pattern.match(line):
        # New message starts, check if the previous message is a Wordle message
        if is_wordle_message and current_message:
            wordle_messages.append("".join(current_message).strip())
        # Reset for the new message
        current_message = [line]
        is_wordle_message = bool(wordle_pattern.search(line))
    elif current_message:
        # Continue collecting lines if part of the same message
        current_message.append(line)

# Add the last collected message if it matches
if is_wordle_message and current_message:
    wordle_messages.append("".join(current_message).strip())

# Save the extracted messages to a new file
output_file = "wordle_messages.txt"
with open(output_file, "w", encoding="utf-8") as file:
    file.write("\n\n".join(wordle_messages))

print(f"Extracted Wordle messages saved to {output_file}")

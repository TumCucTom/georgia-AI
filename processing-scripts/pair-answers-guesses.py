import re

# File paths
word_list_file = "../data/resources/answers.txt"  # First file with words and Wordle numbers
wordle_data_file = "../data/message-data/pre-processed-messages.txt"  # Second file with Wordle numbers and grids
output_file = "../data/message-data/pre-processed-data.txt"  # Output file

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
    current_sections = []  # To hold all sections for the current Wordle number
    current_section = []  # To hold lines of the current section
    wordle_number = None  # To track the Wordle number for the current sections

    def process_section(section):
        """
        Ensure a section has exactly 6 rows, padding with '-1,-1,-1,-1,-1' if necessary.
        """
        while len(section) < 6:
            section.append("-1,-1,-1,-1,-1")
        return section

    def process_sections(wordle_number, sections):
        """
        Process sections for a Wordle number, ensure each has 6 rows, pad to 10 sections, and write to the output file.
        """
        word = wordle_words.get(wordle_number, "UNKNOWN")
        output_file.write(f"Wordle Number: {wordle_number} ({word})\n")

        # Ensure each section has 6 rows
        sections = [process_section(section) for section in sections]

        # Pad to 10 sections if fewer, or trim if more
        while len(sections) < 10:
            sections.append(["-1,-1,-1,-1,-1"] * 6)
        if len(sections) > 10:
            sections = sections[:10]

        # Write each section to the file
        for section in sections:
            output_file.write("\n".join(section) + "\n\n")

    for line in input_file:
        line = line.strip()

        # Check for Wordle Number
        match = re.match(r"Wordle Number: (\d+)", line)
        if match:
            # Process the previous Wordle number if it exists
            if current_sections:
                process_sections(wordle_number, current_sections)
                current_sections = []  # Reset for the next Wordle number

            # Start a new Wordle number
            wordle_number = int(match.group(1))
        elif line:
            # Process grid lines by separating numbers with commas
            formatted_line = ",".join(line)
            current_section.append(formatted_line)
        else:
            # Empty line indicates the end of a section
            if current_section:
                current_sections.append(current_section)
                current_section = []  # Reset for the next section

    # Handle the last Wordle number if the file does not end with an empty line
    if current_sections:
        process_sections(wordle_number, current_sections)

print(f"Processed data saved to {output_file.name}")

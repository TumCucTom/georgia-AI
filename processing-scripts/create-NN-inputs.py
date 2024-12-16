import csv

# File paths
processed_data_file = "../data/pre-processed-data.txt"  # Input file from the old script
output_csv_file = "../data/NN-training-data-no-output.csv"  # Output CSV file

# Open the input and output files
with open(processed_data_file, "r", encoding="utf-8") as input_file, open(output_csv_file, "w", encoding="utf-8", newline="") as csv_file:
    csv_writer = csv.writer(csv_file)

    current_rows = []  # To hold all rows for the current Wordle number
    word = None  # To track the 5-letter word for the current Wordle number

    for line in input_file:
        line = line.strip()

        # Check for a Wordle Number with the word (e.g., "Wordle Number: 277 (apple)")
        if line.startswith("Wordle Number:"):
            # If there's existing data, write it to the CSV file
            if current_rows:
                flat_row = [item for sublist in current_rows for item in sublist]  # Flatten all rows
                flat_row.append(word)  # Add the 5-letter word at the end
                csv_writer.writerow(flat_row)
                current_rows = []  # Reset for the next Wordle number

            # Extract the 5-letter word
            word = line.split("(")[-1].strip(")")

        elif line:
            # Process grid lines into lists of integers
            current_rows.append(line.split(","))

    # Write the last Wordle number's data
    if current_rows:
        flat_row = [item for sublist in current_rows for item in sublist]  # Flatten all rows
        flat_row.append(word)  # Add the 5-letter word at the end
        csv_writer.writerow(flat_row)

print(f"CSV file saved to {output_csv_file}")

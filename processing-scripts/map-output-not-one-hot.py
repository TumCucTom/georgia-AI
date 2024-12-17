import csv

def process_csv_files(input_csv, input_word_file, output_csv):
    # Step 1: Read the second file and build a list of words
    with open(input_word_file, 'r') as f2:
        words_list = f2.read().split()

    # Step 2: Create a dictionary to map 5-letter words to their index in the second file
    word_to_index = {word: idx for idx, word in enumerate(words_list)}

    # Step 3: Read the first CSV file and process it
    updated_rows = []
    with open(input_csv, 'r') as csv_in:
        csv_reader = csv.reader(csv_in)
        for row in csv_reader:
            updated_row = []
            row_changed = False  # Track if any word in the row was updated

            for cell in row:
                words = cell.split()  # Split cell into words
                updated_words = []
                for word in words:
                    # Check if the word is exactly 5 letters and exists in the second file
                    if len(word) == 5 and word in word_to_index:
                        updated_words.append(str(word_to_index[word]))
                        row_changed = True  # Mark that at least one change happened
                    else:
                        updated_words.append(word)
                updated_row.append(" ".join(updated_words))  # Join back the words

            # Add the updated row only if at least one word was replaced
            if row_changed:
                updated_rows.append(updated_row)

    # Step 4: Write the updated content to the output CSV file
    with open(output_csv, 'w', newline='') as csv_out:
        csv_writer = csv.writer(csv_out)
        csv_writer.writerows(updated_rows)

    print(f"Updated CSV file saved to {output_csv}")

# Replace these file paths with your actual file names
input_csv = '../data/NN-training-data-no-output.csv'        # Original CSV file
input_word_file = '../data/resources/5-letter-words.txt'  # File with word list
output_csv = '../data/NN-training-data-not-one-hot-expanded.csv'  # Output CSV file

process_csv_files(input_csv, input_word_file, output_csv)

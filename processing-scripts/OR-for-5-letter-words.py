def extract_five_letter_words(file_path):
    """Extract 5-letter words from a given file."""
    try:
        with open(file_path, 'r') as file:
            # Read all lines and split into words
            words = file.read().split()

        # Filter out words that are not 5 letters long
        five_letter_words = {word.strip().lower() for word in words if len(word.strip()) == 5}
        return five_letter_words
    except FileNotFoundError:
        print(f"The file at {file_path} was not found.")
        return set()
    except Exception as e:
        print(f"An error occurred while processing {file_path}: {e}")
        return set()

def merge_and_write_five_letter_words(file1_path, file2_path, output_path):
    """Merge 5-letter words from two files and write them to a new file."""
    # Extract 5-letter words from both files
    words_from_file1 = extract_five_letter_words(file1_path)
    words_from_file2 = extract_five_letter_words(file2_path)

    # Combine the words from both files (union of sets)
    all_five_letter_words = words_from_file1 | words_from_file2

    # Write the unique 5-letter words to the output file
    try:
        with open(output_path, 'w') as file:
            for word in sorted(all_five_letter_words):
                file.write(word + '\n')
        print(f"Unique 5-letter words have been written to {output_path}.")
    except Exception as e:
        print(f"An error occurred while writing to {output_path}: {e}")

# Example usage
file1_path = '../data/5-letter-words.txt'  # Replace with your first file path
file2_path = '../data/pre-processed-data.txt'  # Replace with your second file path
output_path = '../data/5-letter-words.txt'  # Replace with your desired output file path

merge_and_write_five_letter_words(file1_path, file2_path, output_path)

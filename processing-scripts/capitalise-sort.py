# Read words from a file, capitalize them, and sort alphabetically

def write_array_to_file(file_path, array):
    try:
        # Open the file in write mode ('w'), which will overwrite the existing file
        with open(file_path, 'w') as file:
            # Write each element from the array to the file on a new line
            for item in array:
                file.write(str(item) + '\n')  # Convert item to string if necessary
        print(f"Data has been written to {file_path}.")
    except Exception as e:
        print(f"An error occurred: {e}")

def process_words(file_path):
    try:
        # Open the file and read all lines
        with open(file_path, 'r') as file:
            words = file.readlines()

        # Strip any extra spaces or newline characters and capitalize each word
        capitalized_words = [word.strip().upper() for word in words]

        # Sort the words alphabetically
        capitalized_words.sort()

        # Output the result
        return capitalized_words

    except FileNotFoundError:
        print(f"The file at {file_path} was not found.")
        return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

# Example usage
file_path = '../data/5-letter-words.txt'
sorted_words = process_words(file_path)
write_array_to_file(file_path,sorted_words)

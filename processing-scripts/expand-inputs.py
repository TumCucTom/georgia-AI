import csv

# Function to process the rows
def process_row(row):
    # Create a new row that will hold the transformed data
    new_row = []

    # Process the first 300 entries
    for i in range(min(300, len(row))):
        if row[i] == '0':
            new_row.extend([1, 0, 0])  # Replace 0 with 1,0,0
        elif row[i] == '1':
            new_row.extend([0, 1, 0])  # Replace 1 with 0,1,0
        elif row[i] == '2':
            new_row.extend([0, 0, 1])  # Replace 2 with 0,0,1
        elif row[i] == '-1':
            new_row.extend([-1, -1, -1])  # Replace -1 with -1,-1,-1
        else:
            new_row.append(row[i])  # If value is not 0, 1, 2, or -1, keep it as is

    # Add any remaining entries after the first 300
    if len(row) > 300:
        new_row.extend(row[300:])

    return new_row

# Function to read, process, and write the CSV file
def process_csv(input_file, output_file):
    with open(input_file, mode='r', newline='') as infile:
        csvreader = csv.reader(infile)
        rows = list(csvreader)

    processed_rows = [process_row(row) for row in rows]

    # Write the processed rows to a new CSV file
    with open(output_file, mode='w', newline='') as outfile:
        csvwriter = csv.writer(outfile)
        csvwriter.writerows(processed_rows)

# Example usage
input_file = '../data/training-data/test/MINI-TEST-extensive-small.csv'  # Replace with your input file path
output_file = '../data/training-data/test/MINI-TEST-extensive-large.csv'  # Replace with your desired output file path

process_csv(input_file, output_file)

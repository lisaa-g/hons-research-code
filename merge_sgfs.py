import os

def merge_sgf_files(input_folder, output_file):
    # Open the output file in write mode
    with open(output_file, 'w') as outfile:
        # Loop through all files in the input folder
        for filename in os.listdir(input_folder):
            # Check if the file is an .sgf file
            if filename.endswith(".sgf"):
                file_path = os.path.join(input_folder, filename)
                # Open and read the .sgf file
                with open(file_path, 'r') as infile:
                    content = infile.read()
                    # Write the content to the output file
                    outfile.write(content + "\n\n")  # Add some separation between games
                print(f"Added {filename} to {output_file}")

    print(f"All SGF files have been merged into {output_file}")

# Usage
input_folder = "./Games/13x13/"  # Replace with your folder path
output_file = "13x13_training_data.sgf"     # Desired output file name
merge_sgf_files(input_folder, output_file)

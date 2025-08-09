import os
from pathlib import Path
import sys

def find_lowest_lsd(root_directory: str):
    """
    Searches through sub-subdirectories of a given directory for log files
    to find the minimum 'Best Validation LSD achieved' value.

    This function is optimized to check the second-to-last line of each
    log file. It specifically targets directories two levels deep from the
    root_directory.

    Args:
        root_directory: The full path to the starting folder.
    """

    min_lsd = float('inf')
    # To store the path of the file with the lowest value
    file_with_min_lsd = None
    # The specific text we are looking for at the start of the line
    search_phrase = "Average Validation LSD:"

    print(f"Starting search in: '{os.path.abspath(root_directory)}'")
    print("Targeting log files in sub-subdirectories only...")

    log_files = []
    root_path = Path(root_directory)



    # Iterate through the items in the root directory (level 1 subdirectories)
    for sub_dir in root_path.iterdir():
        # Check if the item is a directory
        log_files.extend(sub_dir.glob('*.log'))
        if sub_dir.is_dir():
            # Iterate through the items in the subdirectory (level 2 sub-subdirectories)
            for sub_sub_dir in sub_dir.iterdir():
                if sub_sub_dir.is_dir():
                    log_files.extend(sub_sub_dir.glob('*.log'))
                    log_files.extend(sub_sub_dir.glob('log'))

    if not log_files:
        print("\nWarning: No log files found in any sub-subdirectories.")


    # Counter for processed files
    files_processed = 0

    for file_path in log_files:
        try:
            # Open the file and read all lines into a list.
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()

            files_processed += 1


            if len(lines) < 50:
                continue

            # Get the second-to-last line, removing any leading/trailing whitespace
            target_line = lines[-3].strip()


            # Check if the line starts with our search phrase
            if target_line.startswith(search_phrase):


                # Extract the numeric value part of the string
                value_str = target_line.replace(search_phrase, "").strip()

                try:
                    # Convert the extracted string to a floating-point number
                    current_lsd = float(value_str)
                    if current_lsd<5.21:
                        print(file_path)
                        print(current_lsd)

                    # If this value is the new lowest, update our variables
                    if current_lsd < min_lsd:

                        min_lsd = current_lsd
                        file_with_min_lsd = file_path

                except ValueError:
                    # This handles cases where the line has the phrase but not a valid number
                    print(f"\nWarning: Found phrase but could not parse number in {file_path} on line: '{target_line}'")
                    continue

        except Exception as e:
            print(f"\nError: Could not process file {file_path}. Reason: {e}")

    # After checking all files, print the final result
    print("\n--- Search Complete ---")
    print(f"Total log files found and processed: {files_processed}")

    if file_with_min_lsd:
        print(f"\nLowest 'Best Validation LSD' found: {min_lsd}")
        print(f"Found in file: {file_with_min_lsd}")
    else:
        print("\nCould not find any lines matching the search criteria in the targeted directories.")


if __name__ == '__main__':

    target_root_folder = '/export/mkrishn9/hrtf_field/experiments_ft/hutubs_cipic'

    # Check if the specified directory exists
    if not os.path.isdir(target_root_folder):
        print(f"Error: The specified directory does not exist: '{target_root_folder}'")
        print("Please update the 'target_root_folder' variable in the script.")
    else:
        find_lowest_lsd(target_root_folder)

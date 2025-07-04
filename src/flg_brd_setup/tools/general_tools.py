import json5
import os

def extract_json_data(file_path):
    """
    Extracts data from a JSON file.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        dict: A dictionary containing the data from the JSON file.
              Returns an empty dictionary if the file is not found or
              if there's a JSON decoding error.
              Prints an error message to the console in case of an issue.
    """
    # Check if the file exists before attempting to open it
    if not os.path.exists(file_path):
        print(f"Error: JSON file not found at path: {file_path}")
        return {} # Return an empty dictionary to avoid NoneType errors

    try:
        with open(file_path, 'r') as f:
            data = json5.load(f)
        return data
    except Exception as e:
        print(f"An unexpected error occurred while reading file {file_path}: {e}")
        return {} # Catch any other unexpected errors and return empty dict
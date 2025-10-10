import json5
import os
import numpy as np
from scipy.stats import norm

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
    
def flatten(xss):
    return [x for xs in xss for x in xs]

def pack(a, n, distribution):

    """
    Create n boxes of values taking into consideration the data distribution and then change the original vector to accomodate those values
    """

    # Check if values are appropiate
    a = np.asarray(a)
    if a.size == 0:
        return np.array([], dtype=int)
    if not (isinstance(n, int) and n > 0):
        raise ValueError("n must be a positive integer")

    if distribution == 'uniform':

        minimum = float(a.min())
        maximum = float(a.max())
        if maximum == minimum:
            return np.zeros_like(a, dtype=int)
        
        # Create n+1 edges equally spaced from min to max
        boxes = np.linspace(minimum, maximum, n + 1)
        
        idx = np.searchsorted(boxes, a, side='right') - 1 # Assign to boxes
        idx = np.clip(idx, 0, n - 1)
        packed_a = idx.astype(int)

    elif distribution == 'normal':
    

        the_cow = float(a.mean())
        std = float(a.std())
        if std == 0.0:
            return np.zeros_like(a, dtype=int)
        
        z = (a - the_cow) / std
        p = norm.cdf(z)        
        p = np.clip(p, 0.0, 1.0) # numerical safety

        boxes = np.linspace(0.0, 1.0, n + 1)

        # Assign to boxes
        idx = np.searchsorted(boxes, p, side='right') - 1
        idx = np.clip(idx, 0, n - 1)
        packed_a = idx.astype(int)

    else:
        raise Exception("Unknown distribution")
    
    return packed_a
""" Utilities for the dataset module, that don't use GEE
"""
import numpy as np


def correct_image_shape(image):
    """
    Transposes an image with size (C, H, W) to an image with size (H, W, C).

    Args:
        image (numpy array): An input image with size (C, H, W).

    Returns:
        numpy array: The transposed image with size (H, W, C).
    """
    # Swap the axes of the input image
    transposed_image = np.swapaxes(image, 0, 2)
    transposed_image = np.swapaxes(transposed_image, 0, 1)
    
    return transposed_image

import pandas as pd
def read_csv(csv_path):
    """
    Read a CSV file from the specified path and return a Pandas DataFrame with the first column dropped.

    Parameters:
    csv_path (str): The path to the CSV file to read.

    Returns:
    pandas.DataFrame: A DataFrame with the first column dropped (since it is index) and columns `1,2,3` renamed as `'Point_id', 'long',` and `'lat'`.
    """
    # read csv file into a pandas dataframe
    df = pd.read_csv(csv_path)
    # first column is the index, so we drop it
    df = df.iloc[:, 1:4]
    df.columns = ['Piont_id', 'long', 'lat']
    return df



from datetime import datetime
def milsec2date(millsec_list: list, no_duplicate = False)->list:
  '''
  Input
  ---
  this function takes `imgcollection.aggregate_array('system:time_start')` which is a list of milliseconds dates as input

  Reutrns
  ---
  * Defult: a list of dates in GEE date string format
  * No_duplicate: returns the list of dates but removes the duplicates
    '''
  if no_duplicate:
    date = [datetime.fromtimestamp(t/1000.0).strftime('%Y-%m-%d') for t in millsec_list]
    date_no_duplicate = list(dict.fromkeys(date))
    return  date_no_duplicate
  else:
    date = [datetime.fromtimestamp(t/1000.0).strftime('%Y-%m-%d') for t in millsec_list] 
    return date



def test_function(function,shape=False, *args, **kwargs):
    try:
        output = function(*args, **kwargs)
        print('Test passed!')
        if shape:
            print(output.shape)
        else:
            print(output)
    except Exception as e:
        print('Test failed!')
        print(e)


class TextColors:
    """
    A class containing ANSI escape codes for printing colored text to the terminal.
    
    Usage:
    ------
    ```
    print(TextColors.HEADER + 'This is a header' + TextColors.ENDC)
    print(TextColors.OKBLUE + 'This is OK' + TextColors.ENDC)
    ```
    
    Attributes:
    -----------
    `HEADER` : str
        The ANSI escape code for a bold magenta font color.
    `OKBLUE` : str
        The ANSI escape code for a bold blue font color.
    `OKCYAN` : str
        The ANSI escape code for a bold cyan font color.
    `OKGREEN` : str
        The ANSI escape code for a bold green font color.
    `WARNING` : str
        The ANSI escape code for a bold yellow font color.
    `FAIL` : str
        The ANSI escape code for a bold red font color.
    `ENDC` : str
        The ANSI escape code for resetting the font color to the default.
    `BOLD` : str
        The ANSI escape code for enabling bold font style.
    `UNDERLINE` : str
        The ANSI escape code for enabling underlined font style.
    
    
    """
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    class BOLD_BAKGROUNDs:
        BLACK = '\033[1;40m'
        RED = '\033[1;41m'
        GREEN = '\033[1;42m'
        YELLOW = '\033[1;43m'
        BLUE = '\033[1;44m'
        PURPLE = '\033[1;45m'
        CYAN = '\033[1;46m'
        WHITE = '\033[1;47m'
        ORANGE = '\033[48;2;255;165;0m\033[1m'
        S1 ='\033[48;2;100;50;50m'
        S2 = '\033[48;2;50;50;100m'

import os
def create_folder_if_not_exists(folder_name):
    """Creates a folder in the current working directory if it doesn't exist.

    Args:
        folder_name (str): The name of the folder to be created.
    """
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f'Folder "{folder_name}" created in the current working directory.')
    else:
        print(f'Folder "{folder_name}" already exists in the current working directory.')
        
def reshape_tensor(tensor):
    """Takes in a pytorch tensor and reshapes it to (C,H,W) if it is not already in that shape.
    
    This Algorithm won't work if C is larger than H or W
    We assume that the smallest dimension is the channel dimension.
    """
    if tensor.dim() == 2: # If it is a 2D image we need to add a channel dimension
        tensor = tensor.unsqueeze(0)
    elif tensor.dim() == 3 and tensor.shape[2] < tensor.shape[0]: # if it is a 3D image and 3rd dim is smallest, it means it it the channel so we permute
        tensor = tensor.permute((2,0,1))
    elif tensor.dim() == 3 and tensor.shape[2] > tensor.shape[0]: # if it is a 3D image and and the first dimension is smaller than others it means its the C and we don't need to permute.
        pass
    else:
        raise ValueError(f"Input tensor shape is unvalid: {tensor.shape}")
    return tensor

def reshape_array(array: np.ndarray) -> np.ndarray:
    """Takes in an array and reshapes it to (C,H,W) if it is not already in that shape.
    
    This Algorithm won't work if C is larger than H or W
    We assume that the smallest dimension is the channel dimension.
    """
    if array.ndim == 2: # If it is a 2D image we need to add a channel dimension
        array = np.expand_dims(array, axis=0)
    elif array.ndim == 3 and array.shape[2] < array.shape[0]: # if it is a 3D image and 3rd dim is smallest, it means it it the channel so we permute
        array = np.swapaxes(array, 0, 2)
        array = np.swapaxes(array, 1, 2)
    elif array.ndim == 3 and array.shape[2] > array.shape[0]: # if it is a 3D image and and the first dimension is smaller than others it means its the C and we don't need to permute.
        pass
    else:
        raise ValueError(f"Input array shape is invalid: {array.shape}")
    return array

# Define the function
def get_df_max_min(df:pd.DataFrame, col):
    """Takes in a pandas dataframe and a column name and returns a tuple of the maximum and minimum values of the column.

    Args:
        df (pd.DataFrame)
        col: The name of the column to get the maximum and minimum values of.

    Returns:
        tuple: A tuple of the maximum and minimum values of the column.
    """
    # Check if the input is a pandas dataframe and the column name is valid
    if isinstance(df, pd.DataFrame) and col in df.columns:
        # Get the maximum and minimum values of the column as floats
        max_val = float(df[col].max())
        min_val = float(df[col].min())
        # Return a tuple of the maximum and minimum values
        return (min_val,max_val)
    else:
        # Raise an exception if the input is invalid
        raise ValueError("Invalid input. Please provide a pandas dataframe and a valid column name.")
    
def normalize(value,min,max):
    """Takes in a value, min and max of the data| returns the normalized value between 0 and 1.
    """
    return (value - min) / (max - min)

def log_transform(value):
    """
    Applies log transformation to the given value with the specified base.

    Parameters:
    - value (float): The value to be transformed.
    - base (float, optional): The logarithm base. Default is 10.

    Returns:
    - float: The log-transformed value.
    """
    return np.log(value)



if __name__ == '__main__':
    # Create an example dataframe with some numeric columns
    df = pd.DataFrame({"A": [1, 2, 3, 4], "B": [5.6, 7.8, 9.0, 10.2], "C": [-1.2, -3.4, -5.6, -7.8]})

    # Test the function on column A and print the result
    print(get_df_max_min(df, "B"))
    print(df)
    # Expected output: (4.0, 1.0)

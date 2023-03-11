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


if __name__ == '__main__':
    #test_function(get_square_roi, 40.02, -105.25, roi_size=1920)
    #test_function(correct_image_shape,True,  np.random.rand(3, 256, 256))
    df = read_csv('D:\\python\\SoilNet\\dataset\\utils\\test.csv')
    print(df)
    
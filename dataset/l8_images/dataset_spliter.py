import os
import random
import shutil

def split_dataset(directory_path, train_ratio, val_ratio, test_ratio, remove_org=False):
    """Splits a dataset into train, validation, and test sets.
    
    Args:
    -----
    `directory_path`(str): A folder that has all the TIF files to be split.
    `train_ratio, val_ratio, test_ratio`(float): The ratio of files to be used for training,validation and test.
    `remove_org`(bool): If True, the original TIF files will be removed from the `directory_path` after copying them into the train test val folders.
    
    
    
    """
    
    if train_ratio + val_ratio + test_ratio != 1.0:
        raise ValueError("Train, validation, and test ratios must add up to 1.0")

    # Create output directories
    train_dir = os.path.join(directory_path, "train")
    val_dir = os.path.join(directory_path, "val")
    test_dir = os.path.join(directory_path, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Get list of TIF files in directory
    tif_files = [f for f in os.listdir(directory_path) if f.endswith('.tif')]

    # Calculate number of files for each set
    num_files = len(tif_files)
    num_train = int(train_ratio * num_files)
    num_val = int(val_ratio * num_files)
    num_test = int(test_ratio * num_files)

    # Shuffle TIF files
    random.shuffle(tif_files)

    # Copy files to output directories
    for i, file_name in enumerate(tif_files):
        if i < num_train:
            shutil.copy(os.path.join(directory_path, file_name), train_dir)
        elif i < num_train + num_val:
            shutil.copy(os.path.join(directory_path, file_name), val_dir)
        else:
            shutil.copy(os.path.join(directory_path, file_name), test_dir)
         
    if remove_org:   
        # Remove TIF files from original directory
        for file_name in tif_files:
            os.remove(os.path.join(directory_path, file_name))
            
if __name__ == "__main__":
    split_dataset("dataset/l8_images", 0.6, 0.2, 0.2)
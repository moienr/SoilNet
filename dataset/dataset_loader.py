import torch
import numpy as np
from torch.utils.data import Dataset
from skimage import io
import os
from torchvision import datasets, transforms
import pandas as pd

class SNDataset(Dataset):
  def __init__(self, l8_dir, csv_dir , l8_bands: list = None ,transform = None):
    # Declaring them becuase we nee them in __getitem__ function
    self.l8_dir = l8_dir
    self.csv_dir = csv_dir
    # List of the names in each path
    self.l8_names = [f for f in os.listdir(l8_dir) if f.endswith('.tif')] # reading only 
    self.l8_names.sort()
    # Declaring the l8 bands we want to use, if None all the bands will be used
    self.l8_bands = l8_bands if l8_bands else None
    # Declaring the transform function
    self.transform = transform


  def __len__(self):
    return len(self.l8_names)
  def __getitem__(self, index):
    l8_img_name = self.l8_names[index] 
    l8_img_path = os.path.join(self.l8_dir,l8_img_name)

    point_id = l8_img_name.split('_')[0]
    df = pd.read_csv(self.csv_dir)
    row = df[df['Point_ID'] == int(point_id)]
    oc = row['OC'].values[0]
    
    
    l8_img = io.imread(l8_img_path)
    if self.l8_bands: l8_img = l8_img[self.l8_bands,:,:]



    if self.transform:
        l8_img  = self.transform(l8_img)
        
    return l8_img,oc

if __name__ == "__main__":
    ds = SNDataset('D:\python\SoilNet\dataset\l8_images\\','D:\python\SoilNet\dataset\LUCAS_2015_Germany_all.csv')
    print(len(ds))
    x = ds.__getitem__(0)
    print('OC: ', x[1])
    print('image shape: ',x[0].shape)
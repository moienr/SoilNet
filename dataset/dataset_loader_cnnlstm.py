import torch
import numpy as np
from torch.utils.data import Dataset
from skimage import io
import os
from torchvision import datasets, transforms
import pandas as pd

from utils.utils import reshape_tensor, reshape_array, get_df_max_min, normalize

class SNDataset(Dataset):
  def __init__(self, l8_dir, oc_csv_dir, climate_csvs_dir , l8_bands: list = None ,transform = None):
    # Declaring them becuase we nee them in __getitem__ function
    self.l8_dir = l8_dir
    # List of the names in each path
    self.l8_names = [f for f in os.listdir(l8_dir) if f.endswith('.tif')] # reading only 
    self.l8_names.sort()
    # Declaring the l8 bands we want to use, if None all the bands will be used
    self.l8_bands = l8_bands if l8_bands else None
    # Declaring the transform function
    self.transform = transform
    # Reading the OC csv file
    self.df_oc = pd.read_csv(oc_csv_dir)
    
    # Reading the climate csv files
    climate_csvs = [f for f in os.listdir(climate_csvs_dir) if os.path.isfile(os.path.join(climate_csvs_dir, f)) and f.endswith('.csv')]
    self.climate_df_list = []
    for climate_csv in climate_csvs:
      self.climate_df_list.append(pd.read_csv(os.path.join(climate_csvs_dir, climate_csv)))


  def __len__(self):
    return len(self.l8_names)
  
  
  def __getitem__(self, index):
    l8_img_name = self.l8_names[index] 
    l8_img_path = os.path.join(self.l8_dir,l8_img_name)

    point_id = l8_img_name.split('_')[0]
    row = self.df_oc[self.df_oc['Point_ID'] == int(point_id)]
    oc = row['OC'].values[0]

    climate_row_list = [climate_row for climate_row in self.climate_df_list if climate_row['Point_ID'] == int(point_id)]
    

    l8_img = io.imread(l8_img_path)
    if self.l8_bands: l8_img = l8_img[self.l8_bands,:,:]



    if self.transform:
        l8_img,oc  = self.transform((l8_img,oc))
        
    return l8_img,oc
  
  

class myNormalize:
  """Normalize the image and the target value"""
  def __init__(self, img_bands_min_max =[[(0,7),(0,1)], [(7,12),(-1,1)]], oc_min = 0, oc_max = 200):
    """
      A class to normalize image and target value arrays.
      
      Args:
      - `img_bands_min_max` (list): A list of tuples defining the bands to normalize and the corresponding minimum and maximum values. Default is [(0,7),(0,1)], [(7,12),(-1,1)], where the first 7 bands are Landsat SR bands and the rest are indices.
      
             `[(from_band, to_band),(min_of_bands , max_of_bands)]`
      - `oc_min` (int or float): The minimum value of the target array. Default is 0.
      - `oc_max` (int or float): The maximum value of the target array. Default is 1000.
      
      Returns:
      - A tuple containing the normalized image and target value arrays.
    """
    self.img_bands_min_max = img_bands_min_max
    self.oc_min = oc_min
    self.oc_max = oc_max

  def __call__(self,sample):
    """
    A class to normalize image and target value arrays.
    
    Args:
    - img_bands_min_max (list): A list of tuples defining the bands to normalize and the corresponding minimum and maximum values. Default is `[(0,7),(0,1)], [(7,12),(-1,1)]`, where the first 7 bands are Landsat SR bands and the rest are indices.
    - oc_min (int or float): The minimum value of the target array. Default is 0.
    - oc_max (int or float): The maximum value of the target array. Default is 1000.
    
    Returns:
    - A tuple containing the normalized image and target value arrays.
    """
    img, oc = sample
    
    # reshaping the image into (bands, height, width)
    img = reshape_array(img)
    
    # IMPORTANT : Replacing NaN values with 0, Just a NAN pixel in the input image will cause the whole image to be NaN in the output
    img[np.isnan(img)] = 0
    
    
    # Normalize the image : first 7 bands are Landsat SR bands, the rest are Indices
    for band_min_max in self.img_bands_min_max:
      if band_min_max[1] != (0,1): # if it is already between 0 and 1 we don't need to normalize it. 
        img[band_min_max[0][0]:band_min_max[0][1]] = normalize(img[band_min_max[0][0]:band_min_max[0][1]], band_min_max[1][0], band_min_max[1][1])
        
        
    # Normalize the target value
    oc = normalize(oc, self.oc_min, self.oc_max)
    

    
    # Cutting out of range Vlaues
    img[img > 1] = 1
    img[img < 0] = 0
    oc = oc if oc < 1 else 1
    oc = oc if oc > 0 else 0


    return img, oc  


class myToTensor:
    def __init__(self,dtype=torch.float32, ouput_size = (64,64)):
        self.dtype = dtype
        self.resize = transforms.Resize(ouput_size)
    def __call__(self,sample):
        image, oc = sample
        return (self.resize(reshape_tensor(torch.from_numpy(image))).to(dtype=self.dtype), torch.tensor(oc).to(dtype=self.dtype))


if __name__ == "__main__":
    ds = SNDataset('D:\python\SoilNet\dataset\l8_images\\train\\','D:\python\SoilNet\dataset\LUCAS_2015_all.csv')
    print(len(ds))
    x = ds[0]
    print('OC: ', x[1], type(x[1]))
    print('image shape: ',x[0].shape , x[0].dtype)
    
    
    # Testing the transforms
    print('Testing MyTransfroms...')
    mynorm = myNormalize()
    rand_img = np.random.rand(100,100,12)
    rand_img[7:12] = rand_img[7:12] * 2 - 1
    rand_oc = np.random.rand(1) * 1000
    
    my_to_tensor = myToTensor()
    
    transform = transforms.Compose([mynorm, my_to_tensor])
    
    y = transform((rand_img, rand_oc))
    print('OC: ', y[1], type(y[1]))
    print('image shape: ',y[0].shape , y[0].dtype , torch.min(y[0]), torch.max(y[0]) , sep=" | ")
    
    
    print("Testing the dataset with transforms...")
    ds = SNDataset('D:\python\SoilNet\dataset\l8_images\\train\\','D:\python\SoilNet\dataset\LUCAS_2015_all.csv',transform=transform)
    x = ds[0]
    print('OC: ', x[1], type(x[1]))
    print('image shape: ',x[0].shape , x[0].dtype)
    
    
import torch
import numpy as np
from torch.utils.data import Dataset
from skimage import io
import os
from torchvision import datasets, transforms
import pandas as pd
import torch.nn.functional as F

from utils.utils import reshape_tensor, reshape_array, get_df_max_min, normalize

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
    # Reading the csv file in __init__ function to avoid reading it in every __getitem__ call
    self.df = pd.read_csv(self.csv_dir)

  def __len__(self):
    return len(self.l8_names)
  def __getitem__(self, index):
    l8_img_name = self.l8_names[index] 
    l8_img_path = os.path.join(self.l8_dir,l8_img_name)

    point_id = l8_img_name.split('_')[0]
    
    row = self.df[self.df['Point_ID'] == int(point_id)]
    oc = row['OC'].values[0]
    

    l8_img = io.imread(l8_img_path)
    if self.l8_bands: l8_img = l8_img[self.l8_bands,:,:]



    if self.transform:
        l8_img,oc  = self.transform((l8_img,oc))
        
    return l8_img,oc
  
class SNDatasetClimate(Dataset):
  def __init__(self, l8_dir, csv_dir , climate_csv_folder,
               l8_bands: list = None ,transform = None,
               dates = ['20100101', '20100201', '20100301', '20100401', '20100501', '20100601',
                        '20100701', '20100801', '20100901', '20101001', '20101101', '20101201',
                        '20110101', '20110201', '20110301', '20110401', '20110501', '20110601',
                        '20110701', '20110801', '20110901', '20111001', '20111101', '20111201',
                        '20120101', '20120201', '20120301', '20120401', '20120501', '20120601',
                        '20120701', '20120801', '20120901', '20121001', '20121101', '20121201',
                        '20130101', '20130201', '20130301', '20130401', '20130501', '20130601',
                        '20130701', '20130801', '20130901', '20131001', '20131101', '20131201',
                        '20140101', '20140201', '20140301', '20140401', '20140501', '20140601',
                        '20140701', '20140801', '20140901', '20141001', '20141101', '20141201', '20150101'],
               climate_dtype = torch.float32, normalize_climate = True
               ):
    """_summary_

    Args:
        l8_dir (_type_): _description_
        csv_dir (_type_): _description_
        climate_csv_folder (_type_): _description_
        l8_bands (list, optional): _description_. Defaults to None.
        transform (_type_, optional): _description_. Defaults to None.
        dates (list, optional): _description_. Defaults to ['20100101', '20100201', '20100301', '20100401', '20100501', '20100601', '20100701', '20100801', '20100901', '20101001', '20101101', '20101201', '20110101', '20110201', '20110301', '20110401', '20110501', '20110601', '20110701', '20110801', '20110901', '20111001', '20111101', '20111201', '20120101', '20120201', '20120301', '20120401', '20120501', '20120601', '20120701', '20120801', '20120901', '20121001', '20121101', '20121201', '20130101', '20130201', '20130301', '20130401', '20130501', '20130601', '20130701', '20130801', '20130901', '20131001', '20131101', '20131201', '20140101', '20140201', '20140301', '20140401', '20140501', '20140601', '20140701', '20140801', '20140901', '20141001', '20141101', '20141201', '20150101'].
        climate_dtype (_type_, optional): Datatype when converting to Tensor | only works if Transfrom is goven. Defaults to torch.float32.
    """
    
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
    # Reading the csv file in __init__ function to avoid reading it in every __getitem__ call
    self.df = pd.read_csv(self.csv_dir)
    
    # Reading Climate csv files
    # List all files in the directory and filter for .csv files
    csv_files = [f for f in os.listdir(climate_csv_folder) if os.path.isfile(os.path.join(climate_csv_folder, f)) and f.endswith('.csv')]
    self.clim_dfs =  [pd.read_csv(os.path.join(climate_csv_folder, f)) for f in csv_files]
    
    if normalize_climate:
        norm_clim = NormalizeClimDF(dates=dates)
        self.clim_dfs = [norm_clim(clim_df) for clim_df in self.clim_dfs]
    
    self.dates = dates
    self.clim_dtype = climate_dtype
    
  def __len__(self):
    return len(self.l8_names)
  
  def __getitem__(self, index):
    l8_img_name = self.l8_names[index] 
    l8_img_path = os.path.join(self.l8_dir,l8_img_name)

    point_id = l8_img_name.split('_')[0]
    
    row = self.df[self.df['Point_ID'] == int(point_id)]
    oc = row['OC'].values[0]
    
    self.clim_dfs_row = [df[df['Point_ID'] == int(point_id)] for df in self.clim_dfs]
    self.clim_dfs_row = [df[self.dates] for df in self.clim_dfs_row]
    clim_arr = np.stack([df.values.squeeze() for df in self.clim_dfs_row], axis=1)

    l8_img = io.imread(l8_img_path)
    if self.l8_bands: l8_img = l8_img[self.l8_bands,:,:]



    if self.transform:
        l8_img,oc  = self.transform((l8_img,oc))
        clim_arr = torch.tensor(clim_arr).to(dtype=self.clim_dtype)
        
    return (l8_img,clim_arr),oc
  
#############################################################################################################    
############################################# Transformations ###############################################    
#############################################################################################################   

class myNormalize:
  """Normalize the image and the target value"""
  def __init__(self, img_bands_min_max =[[(0,7),(0,1)], [(7,12),(-1,1)], [(12), (-4,2963)], [(13), (0, 90)]], oc_min = 0, oc_max = 200):
    """
      A class to normalize image and target value arrays.
      
      Args:
      - `img_bands_min_max` (list): A list of tuples defining the bands to normalize and the corresponding minimum and maximum values.
            * Default is `[(0,7),(0,1)], [(7,12),(-1,1), [(12), (-4,2963)],[(13), (0, 90)]]`, where the first 7 bands are Landsat SR bands and the rest are indices. band 12 is SRTM and band 13 is slope
      
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
        if isinstance(band_min_max[0], tuple): # if it is a range of bands
          img[band_min_max[0][0]:band_min_max[0][1]] = normalize(img[band_min_max[0][0]:band_min_max[0][1]], band_min_max[1][0], band_min_max[1][1])
        elif isinstance(band_min_max[0], int): # if it is a single band
          img[band_min_max[0]] = normalize(img[band_min_max[0]], band_min_max[1][0], band_min_max[1][1])
        else: # if it is not a tuple or an int
          raise ValueError('The first element of the tuple must be a tuple or an int')
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
      
class Augmentations:
    """Data Augmentation Class
    To avoid 0 areas when rotating the image, we pad the image with 1/4 of the output size on each side.
    and after rotating we crop the image to the output size.
    """
    def __init__(self, aug_prob = 0.5, out_shape = (64,64)):
        self.aug_prob = aug_prob
        self.aug = transforms.Compose([
                                      transforms.Pad(out_shape[0]//4, out_shape[1]//4, padding_mode='reflect'),
                                      transforms.RandomHorizontalFlip(p=0.5),
                                      transforms.RandomVerticalFlip(p=0.5),
                                      transforms.RandomRotation((0,90)),
                                      transforms.CenterCrop(size=out_shape)
                                       ])
    def __call__(self,sample):
      image, oc = sample
      return(self.aug(image), oc)
        
        
        
class RFTransform:
  """
  This class is used to transform the image and the target value to be used in the Random Forest model.
  """
  def __init__(self):
    """
    This class is used to transform the image and the target value to be used in the Random Forest model.
    """
    pass

  def __call__(self,sample, oc_max = 87, oc_min = 0):
    """
    Input:
    - sample (tuple): A tuple containing the image and the target value.
    Returns:
    - A tuple containing the reshaped image and the target value, in the form of a numpy array.
    """
    img, oc = sample
    
    # reshaping the image into (bands, height, width)
    img = reshape_array(img)
    
    # Cliping oc values to be between 0 and 87
    oc = oc if oc < oc_max else oc_max
    oc = oc if oc > oc_min else oc_min
    
    # IMPORTANT : Replacing NaN values with 0, Just a NAN pixel in the input image will cause the whole image to be NaN in the output
    img[np.isnan(img)] = 0

    return img, oc  
        
class TensorCenterPixels:
    """Takes in a pytorch tensor and returns the center pixels of the image.

    """
    def __init__(self, pixel_radius = 1, interpolate_center_pixel = False):
        """
        Args:
            pixel_radius (int): The radius of the region around the center of the image to return. Default is 1. (1 returns a 2x2 region, 2 returns a 4x4 region, etc.)
            interpolate_center_pixel (bool): Whether or not to use bilinear interpolation to estimate the center pixel value, If Falce,
            it will return a (c, 2*pixel_radius, 2*pixel_radius) tensor. If True, it will return a (c, 1, 1) tensor. (Default is False
                                              Default is False.
        Returns:
            A tuple containing the center pixels of the image and the target value.
        """
        self.pixel_radius = pixel_radius
        self.interpolate_center_pixel = interpolate_center_pixel
        
    def bilinear_interpolation(self, tensor):
        """A method that resamples a tensor using bilinear interpolation to estimate the center pixel value.
           Input tensors are even, so they don't have a center, we upsample them to an odd nmber (1 pixel larger)
           Then we take the center Pixel.

        Args:
            tensor (torch.Tensor): The tensor to resample.

        Returns:
            A tensor of shape (C,1,1), where C is the number of channels in the input tensor, containing the estimated
            values of the center pixel for each channel.
        """
        # Resample tensor to (c,3,3) using bilinear interpolation
        upsample = torch.nn.Upsample(size=(self.pixel_radius*2+1,self.pixel_radius*2+1), mode='bilinear', align_corners=True)
        resampled_tensor = upsample(tensor.unsqueeze(0))
        resampled_tensor = resampled_tensor.squeeze(0)
        return resampled_tensor[:,self.pixel_radius:self.pixel_radius+1,self.pixel_radius:self.pixel_radius+1]
    
    def __call__(self,sample):
        """A method that returns the center pixels of an image.

        Args:
            sample (tuple): A tuple of (image, oc), where image is a pytorch tensor of shape (C, H, W) and oc is an 
                            object class label.

        Returns:
            If self.interpolate_center_pixel is False, returns a tensor of shape (C, 2*pixel_radius, 2*pixel_radius),
            containing the center pixels of the input image. If self.interpolate_center_pixel is True, returns a tensor
            of shape (C,1,1), containing the estimated values of the center pixel for each channel, obtained by resampling
            the input image using bilinear interpolation.
        """
        image, oc = sample
        image = transforms.functional.center_crop(image, self.pixel_radius*2)
        if self.interpolate_center_pixel:
          image = self.bilinear_interpolation(image)
        return image, oc

class NormalizeClimDF:
    def __init__(self,
                 dates = ['20100101', '20100201', '20100301', '20100401', '20100501', '20100601', '20100701', '20100801', '20100901', '20101001', '20101101', '20101201', '20110101', '20110201', '20110301', '20110401', '20110501', '20110601', '20110701', '20110801', '20110901', '20111001', '20111101', '20111201', '20120101', '20120201', '20120301', '20120401', '20120501', '20120601', '20120701', '20120801', '20120901', '20121001', '20121101', '20121201', '20130101', '20130201', '20130301', '20130401', '20130501', '20130601', '20130701', '20130801', '20130901', '20131001', '20131101', '20131201', '20140101', '20140201', '20140301', '20140401', '20140501', '20140601', '20140701', '20140801', '20140901', '20141001', '20141101', '20141201', '20150101'],
                 ):
        self.dates = dates
    def __call__(self, df):
      if df.isna().values.any() or df.isnull().values.any():
          raise ValueError('NaN or Null values in dataframe')
      # Reading Time Series Data  
      df_vals = df[self.dates]
      X = df_vals.values# Converting to numpy array
      X = (X - np.min(X)) / (np.max(X) - np.min(X)) # Normalizing
      df_vals = pd.DataFrame(X) # Converting back to dataframe
      df[self.dates] = df_vals # Replacing the normalized values in the original dataframe
      return df
      
#############################################################################################################    
#############################################      Tests      ###############################################    
#############################################################################################################   

if __name__ == "__main__":
    ds = SNDataset('D:\python\SoilNet\dataset\l8_images\\train\\','D:\python\SoilNet\dataset\LUCAS_2015_all.csv')
    print(len(ds))
    x = ds[0]
    print('OC: ', x[1], type(x[1]))
    print('image shape: ',x[0].shape , x[0].dtype)
    
    
    # # Testing the transforms
    # print('Testing MyTransfroms...')
    # mynorm = myNormalize()
    # rand_img = np.random.rand(100,100,12)
    # rand_img[7:12] = rand_img[7:12] * 2 - 1
    # rand_oc = np.random.rand(1) * 1000
    
    # my_to_tensor = myToTensor()
    
    # transform = transforms.Compose([mynorm, my_to_tensor])
    
    # y = transform((rand_img, rand_oc))
    # print('OC: ', y[1], type(y[1]))
    # print('image shape: ',y[0].shape , y[0].dtype , torch.min(y[0]), torch.max(y[0]) , sep=" | ")
    
    
    print("Testing the dataset with transforms...")
    mynorm = myNormalize()
    my_to_tensor = myToTensor()
    transform = transforms.Compose([mynorm, my_to_tensor])
    ds = SNDataset('D:\python\SoilNet\dataset\l8_images\\train\\','D:\python\SoilNet\dataset\LUCAS_2015_all.csv',transform=transform)
    rand = np.random.randint(0,len(ds))
    x = ds[rand]
    print('OC: ', x[1], type(x[1]))
    print('image shape: ',x[0].shape , x[0].dtype)
    print('image min: ', torch.min(x[0]), 'image max: ', torch.max(x[0]))
    
    print("Testing the dataset with transforms and Augmentations...")
    augment = Augmentations()
    aug_img = augment(x)
    print("Augmented image shape: ", aug_img[0].shape)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1,2)
    ax[0].imshow(x[0].permute(1,2,0).numpy()[:,:,[3,2,1]]*2)
    ax[1].imshow(aug_img[0].permute(1,2,0).numpy()[:,:,[3,2,1]]*2)
    plt.show()
    
    
    print("Testing Torch Center Pxiel...")
    cp = TensorCenterPixels(1,interpolate_center_pixel=True)
    rand_tensor = torch.tensor([[[1,2],[3,4]],[[1,2],[3,4]],[[1,2],[3,4]]]).to(torch.float32)
    # rand_tensor = torch.rand(8,100,100)
    rand_oc = np.random.rand(1) * 1000
    croped = cp((rand_tensor,rand_oc))
    print("shape after croping: ", croped[0].shape)
    print("Center Pixle of the First band: ",croped[0][0])
    
    
    
    print("Testing SNDatasetClimate...")
    ds = SNDatasetClimate('D:\\python\\SoilNet\\dataset\\l8_images\\train\\',
                          'D:\\python\\SoilNet\\dataset\\LUCAS_2015_all.csv',
                          "D:\\python\\SoilNet\\dataset\\Climate\\All\\filled", transform=transform)
    rand = np.random.randint(0,len(ds))
    x = ds[rand]
    print('OC: ', x[1], type(x[1]))
    print('image shape: ',x[0][0].shape , x[0][0].dtype)
    print('image min: ', torch.min(x[0][0]), 'image max: ', torch.max(x[0][0]))
    print("climate:", x[0][1].shape)
    x = ds[2] # checking if getting another Item creaste an error or not | last time getting the first item was changing global values which leaded to errors
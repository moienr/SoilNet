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
        
    # Since we need All the values in each df to normalize, we will normalize them all together in the __init__ function
    self.climate_df_list_normalized = []
    for df in self.climate_df_list:
      df_vals = df.iloc[:, 1:13].values
      df_vals_norm = (df_vals - df_vals.min()) / (df_vals.max() - df_vals.min()) # normalizing the values
      # Storing the normalized values in the dataframe
      df[df.columns[1:13]] = df_vals_norm
      self.climate_df_list_normalized.append(df)
      #print("The Min and Max value of the df after normalization",df.iloc[:, 1:13].values.min(), df.iloc[:, 1:13].values.max(), df.values.shape)


  def __len__(self):
    return len(self.l8_names)
  
  
  def __getitem__(self, index):
    l8_img_name = self.l8_names[index] 
    l8_img_path = os.path.join(self.l8_dir,l8_img_name)
    #print('l8name: ', l8_img_name)
    point_id = l8_img_name.split('_')[0]
    row = self.df_oc[self.df_oc['Point_ID'] == int(point_id)]
    oc = row['OC'].values[0]

    climate_row_list = [df[df['Point_ID'] == int(point_id)] for df in self.climate_df_list_normalized]
    #print('first row: ', climate_row_list[0].values.shape)
    climate_row_vals = [row.iloc[:, 1:13].values[0] for row in climate_row_list] # row.values is (1, 13) we remove the first dim by using values[0]
    climate_stcked_array = np.stack(climate_row_vals)
    climate_array = climate_stcked_array.T # LSTM expects the input to be (batch_size, seq_len, input_size)
    

    l8_img = io.imread(l8_img_path)
    
    if self.transform: # normalizes the l8_img and the oc, and returns the climate_array unchanged, but converts them to tensors
      l8_img,oc,climate_array  = self.transform((l8_img,oc,climate_array))
    
    
    if self.l8_bands: l8_img = l8_img[self.l8_bands,:,:]




        
    return l8_img,oc,climate_array
  
  

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
    
    Inputs:
    - `sample` (tuple): A tuple containing the (img, oc, climate_vlues)
    
    Returns:
    - A tuple containing the normalized image and oc, and the unchanged climate values.
    """
    img, oc, clims = sample
    
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


    return img, oc, clims

class myToTensor:
    def __init__(self,dtype=torch.float32, ouput_size = (64,64)):
        self.dtype = dtype
        self.resize = transforms.Resize(ouput_size)
    def __call__(self,sample):
        image, oc, clims = sample
        return (self.resize(reshape_tensor(torch.from_numpy(image))).to(dtype=self.dtype), torch.tensor(oc).to(dtype=self.dtype), torch.tensor(clims).to(dtype=self.dtype))




def plot_bands(image, labels, array):
    fig, axs = plt.subplots(3, 7, figsize=(15, 10))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    axs = axs.ravel()

    for i in range(image.shape[2]):
        axs[i].imshow(image[:, :, i], cmap='gray')
        axs[i].set_title(labels[i])
        axs[i].set_axis_off()
        cbar = plt.colorbar(axs[i].imshow(image[:, :, i], cmap='gray'), ax=axs[i])
        cbar.ax.set_ylabel(labels[i], rotation=270, labelpad=15)
        
    # Add the additional plot
    dates = np.arange(1, 13)
    rows = array
    ax_mean = fig.add_subplot(3, 1, 3)
    for i, arr in enumerate(rows):
        ax_mean.plot(dates, arr, label=labels[i])
        # Calculate and add the mean value label
        mean_val = np.mean(arr)
        ax_mean.text(dates[-1], arr[-1], f"Mean: {mean_val:.2f}", va='center', ha='left', color='black')

    # Set the plot title and legend
    ax_mean.set_title('Climate Data over Time')
    ax_mean.legend(loc='upper right')
    #ax_mean.set_xticks(dates)
    ax_mean.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], rotation=45, ha='right')

    plt.show()




if __name__ == "__main__":
    import matplotlib.pyplot as plt
    feature_names =['aet','def','pdsi','pet','pr','ro','soil','srad','swe','tmmn','tmmx','vap','vpd','vs']
    
    ds = SNDataset('D:\python\SoilNet\dataset\l8_images\\train\\','D:\python\SoilNet\dataset\LUCAS_2015_all.csv', climate_csvs_dir="D:\python\SoilNet\dataset\Climate\\filled\\")
    print(len(ds))
    x = ds[0]
    print('OC: ', x[1], type(x[1]))
    print('image shape: ',x[0].shape , x[0].dtype)
    print('climate shape: ',x[2].shape , x[2].dtype)

    # TODO: Plot the image and the climate data
    
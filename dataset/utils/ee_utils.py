""" A collection of utility functions for working with Earth Engine (EE) in Python.

    Functions
    ---------
    ### `get_square_roi` :
        returns a square region of interest (ROI) centered at the given latitude and longitude coordinates with the specified size.
    ### `get_cloud_mask` : 
        Takes an ee.Image and returns the cloud, cloud shadow and  cloud_or_cloudShadow mask
    ### `get_snow_mask` : 
        Takes an ee.Image and returns the snow mask
    ### `get_mean_ndvi` :
        Takes an ee.Image and returns the mean NDVI value of the image
    ### `get_mask_ones_ratio` : 
        Takes a  01 mask as an ee.Image and returns the ratio of ones in the mask
    ### `get_not_nulls_ratio` : 
        Takes an ee.Image and returns the ratio of pixels that are not null in the image.
    ### `add_mineral_indices` : 
        Takes an ee.Image and adds the following mineral indices to it as it bands: clayIndex, ferrousIndex, carbonateIndex, rockOutcropIndex
    ### `get_closest_image` : 
        Takes an ee.ImageCollection and a date and returns the image in the collection that is closest to the given date.
    ### `radiometric_correction`: 
        Takes an ee.Image and returns the radiometrically corrected image. (only the Reflectance bands will change)
"""

import ee
import geemap
import math 

# if __name__ != '__main__':
#     try:
#         ee.Initialize()
#     except Exception as e:
#         print("Failed to initialize Earth Engine: ", e)
#         print("Maybe try ee.Authenticate() and ee.Initialize() again?")


def get_square_roi(lat, lon, roi_size = 1920, return_gee_object = False):
    """
    Returns a square region of interest (ROI) centered at the given latitude and longitude
    coordinates with the specified size. By default, the ROI is returned as a list of
    coordinate pairs (longitude, latitude) that define the corners of the square. If
    `return_gee_object` is True, the ROI is returned as an Earth Engine geometry object.

    Args
    ----
        `lat` (float): Latitude coordinate of the center of the ROI.
        `lon` (float): Longitude coordinate of the center of the ROI.
        `roi_size` (int, optional): Size of the square ROI in meters. Default is 1920 meters. (about `64` pixels of `30m` resolution)
        `return_gee_object` (bool, optional): Whether to return the ROI as an Earth Engine geometry
            object instead of a list of coordinates. Default is False.

    Returns
    -------
        list or ee.Geometry.Polygon: If `return_gee_object` is False (default), a list of coordinate
            pairs (longitude, latitude) that define the corners of the square ROI. If `return_gee_object`
            is True, an Earth Engine geometry object representing the square ROI.

    Usage
    -----
        # Get a square ROI centered at lat=37.75, lon=-122.42 with a size of 1000 meters
        roi = get_square_roi(37.75, -122.42, roi_size=1000)
        print(roi)  # Output: [[-122.431, 37.758], [-122.408, 37.758], [-122.408, 37.741], [-122.431, 37.741], [-122.431, 37.758]]

    """

    # Convert the lat-long point to an EE geometry object
    point = ee.Geometry.Point(lon, lat)

    # Create a square buffer around the point with the given size
    roi = point.buffer(roi_size/2).bounds().getInfo()['coordinates']
    
    if return_gee_object:
        return ee.Geometry.Polygon(roi, None, False)
    else:
        # Return the square ROI as a list of coordinates
        return roi
    
    

# CloudMask but in a function form
def get_cloud_mask(img: ee.Image, pixel_quality_band='QA_PIXEL',
                    cloud_bit = 3,
                    cloud_shadow_bit = 4,
                    cloud_confidence_bit = 8,
                    cloud_shadow_confidence_bit = 10):
    """Takes an ee.Image and returns the cloud, cloud shadow and  cloud_or_cloudShadow mask

    Args:
        `img` (ee.Image): An ee.Image object containing a pixel quality band. (e.g. 'QA_PIXEL' of Landsat8 SR)
        `pixel_quality_band` (str, optional): Name of the pixel quality band. Default is 'QA_PIXEL'. (e.g. 'QA_PIXEL' of Landsat8 SR)
        `cloud_bit` (int, optional): Bit position of the cloud bit. Default is 3.
        `cloud_shadow_bit` (int, optional): Bit position of the cloud shadow bit. Default is 4.
        `cloud_confidence_bit` (int, optional): Bit position of the cloud confidence bit. Default is 8.
        `cloud_shadow_confidence_bit` (int, optional): Bit position of the cloud shadow confidence bit. Default is 10.
        
        * Refrence for Defualt Values: https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC08_C02_T1_L2#bands

    Returns:
        tuple: A tuple containing the cloud mask, cloud shadow mask, and the combined mask. (ee.Image, ee.Image, ee.Image)
    """
    qa = img.select(pixel_quality_band)
    # Get the pixel values for the cloud, cloud shadow, and snow/ice bits
    cloud = qa.bitwiseAnd(1 << cloud_bit).And(qa.bitwiseAnd(3<<cloud_confidence_bit))
    cloud_shadow = qa.bitwiseAnd(1 << cloud_shadow_bit).And(qa.bitwiseAnd(3<<cloud_shadow_confidence_bit))
    return cloud, cloud_shadow, cloud.Or(cloud_shadow)


# Snow/Ice mask
def get_snow_mask(img: ee.Image, pixel_quality_band='QA_PIXEL',
                    snow_bit = 5,
                    snow_confidence_bit = 12):
    """Takes an ee.Image and returns the Snow mask

    Args:
        `img` (ee.Image): An ee.Image object containing a pixel quality band. (e.g. 'QA_PIXEL' of Landsat8 SR)
        `pixel_quality_band` (str, optional): Name of the pixel quality band. Default is 'QA_PIXEL'. (e.g. 'QA_PIXEL' of Landsat8 SR)
        `snow_bit` (int, optional): Bit position of the snow bit. Default is 3.
        `snow_confidence_bit` (int, optional): Bit position of the snow confidence bit. Default is 8.

        
        * Refrence for Defualt Values: https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC08_C02_T1_L2#bands

    Returns:
        tuple: A tuple containing the snow mask, snow shadow mask, and the combined mask. (ee.Image, ee.Image, ee.Image)
    """
    qa = img.select(pixel_quality_band)
    snow = qa.bitwiseAnd(1 << snow_bit).And(qa.bitwiseAnd(3<<snow_confidence_bit))
    return snow


from typing import List
def get_mean_ndvi(image, bands: List[str] = ['SR_B5', 'SR_B4']):
    """
    Returns the mean NDVI of the given image.
    """
    # Compute NDVI
    ndvi = image.normalizedDifference(bands)
    
    # Compute mean of NDVI
    mean_ndvi = ndvi.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=image.geometry(),
        scale=image.projection().nominalScale(),
        maxPixels=1e13
    ).get('nd')
    
    return mean_ndvi



# Function to get the Ratio of ones to total pixels
def get_mask_ones_ratio(mask:ee.Image, scale = 30, in_percentage = True):
    """
    Function to get the percentage or the ratio of ones to total pixels in an Earth Engine image mask.

    Args:
    -----
        `mask` (ee.Image): An Earth Engine image mask.
        `scale` (int, optional): The scale to use for reducing the image. Defaults to 30.
        `in_percentage` (bool, optional): Whether to return the ratio or the percentage. Defaults to True.

    Returns:
    --------
        float: The ratio of ones to total pixels in the mask.
    """
    # Compute the number of ones and total number of pixels in the mask
    #band_name = mask.bandNames().getInfo()[0]
    stats = mask.reduceRegion(
        reducer=ee.Reducer.sum().combine(
            reducer2=ee.Reducer.count(),
            sharedInputs=True
        ),
        geometry=mask.geometry(),
        scale=scale,
        maxPixels=1e9
    )

    # Extract the number of ones and total number of pixels from the result
    ones = stats.get(stats.keys().get(1))
    total = stats.get(stats.keys().get(0))

    # Compute the ratio of ones to total pixels
    ratio = ee.Number(ones).divide(total)
    

    # Return the ratio
    return ratio.multiply(100) if in_percentage else ratio


# Function to get the Ratio of Nulls to total pixels that an roi could have
def get_not_nulls_ratio(image:ee.Image, roi:ee.Geometry ,scale = 30, in_percentage = True) -> ee.Number:
    """
        Calculates the ratio of not null null values to total pixels that an ROI (Region of Interest) could have for a given image.
        
        Args:
        -----
        - image (ee.Image): The image for which the nulls ratio needs to be calculated.
        - roi (ee.Geometry): The region of interest for which the nulls ratio needs to be calculated.
        - scale (int, optional): The scale at which to perform the reduction. Defaults to 30.
        
        Returns:
        --------
        - ratio (ee.Number): The ratio of not null null values to total pixels for the given ROI and image.
    """

    # Creates a 1 & 0 mask of the image, 0 on null areas, and 1 for pixels with values
    # th clip is really important since, mask() method goes over boundries.
    mask = image.mask().select(0).clip(roi)
    # Return the ratio
    return get_mask_ones_ratio(mask, scale = scale, in_percentage = in_percentage)



def add_mineral_indices(inImage): #Please change the name of the function if necessary.
    """
    Adds five new bands (clayIndex, ferrousIndex, carbonateIndex, rockOutcropIndex and ndvi) to an input image.
    
    Parameters:
        inImage (ee.Image): The input image to add the new bands to.
        
    Returns:
        ee.Image: The output image with the added bands.
    """
    # Clay Minerals = swir1 / swir2
    #clayIndex = inImage.select('SR_B6').divide(inImage.select('SR_B7')).rename('clayIndex')
    normClayIndex = inImage.normalizedDifference(['SR_B6','SR_B7']).rename('clayIndex')
    # Ferrous Minerals = swir / nir
    #ferrousIndex = inImage.select('SR_B6').divide(inImage.select('SR_B5')).rename('ferrousIndex')
    normFerrousIndex = inImage.normalizedDifference(['SR_B6','SR_B5']).rename('ferrousIndex')
    # Carbonate Index = (red - green) / (red + green)
    carbonateIndex = inImage.normalizedDifference(['SR_B4','SR_B3']).rename('carbonateIndex')

    # Rock Outcrop Index = (swir1 - green) / (swir1 + green)
    rockOutcropIndex = inImage.normalizedDifference(['SR_B6','SR_B3']).rename('rockOutcropIndex')
    
    # NDVI Index = (Band 5 – Band 4) / (Band 5 + Band 4) 
    ndvi = inImage.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI')

    # Add bands
    outStack = inImage.addBands([normClayIndex, normFerrousIndex, carbonateIndex, rockOutcropIndex, ndvi])

    return outStack

def add_topo(): 
    """
    Adds topographical information, elevation and slope.
    
    Parameters:
        inImage (ee.Image): The input image to add the new bands to.
        
    Returns:
        ee.Image: The output image with the added bands.
    """
    
    elevation = ee.Image("USGS/SRTMGL1_003"); #Global DEM
    topo = ee.Algorithms.Terrain(elevation)
    
    #get % slope
    slopeDeg = topo.select(1)
    slopeRads = slopeDeg.multiply(math.pi).divide(ee.Number(180))
    slopeTan = slopeRads.tan()
    slopePCT = slopeTan.multiply(ee.Number(100)).rename('slopePCT')
    
    #Add topography bands to image composite
    topo = topo.float()
    topo = topo.select('elevation').addBands(slopePCT)
    
    return topo


def get_closest_image(image_collection:ee.ImageCollection, date:str, clip_dates: int = None) -> ee.Image:
    """
    Returns the closest image in the given image collection to the given date.
    Parameters:
    -----------
    `image_collection` : ee.ImageCollection
        The image collection from which to find the closest image.
    `date` : str or datetime
        The target date as a string in "YYYY-MM-DD" format or a datetime object.
    `clip_dates` : int, optional
        The number of days to clip the image collection to. Only images within this range
        of the target date will be considered. If not specified, all images in the collection
        will be considered.

    Returns:
    --------
    closest_image : ee.Image
        The closest image in the image collection to the target date.

    """
    # Convert the date to milliseconds since the Unix epoch
    date_millis = ee.Date(date).millis()
    
    if clip_dates:
        # Filter the collection to images within 7 days of the target date
        filtered_collection = image_collection.filterDate(
            ee.Date(date).advance(-1*clip_dates, 'day'),
            ee.Date(date).advance(clip_dates, 'day')
        )
    else:
        filtered_collection = image_collection
    
    # Compute the time difference between each image and the target date
    filtered_collection = ee.ImageCollection(
        ee.List(filtered_collection.toList(filtered_collection.size()))
        .map(lambda image: image.set('timeDiff', abs(ee.Number(image.date().millis()).subtract(date_millis))))
    )
    
    # Get the image with the minimum time difference
    closest_image = filtered_collection.sort('timeDiff').first()
    
    return closest_image


# applying the Mult and Add function to the image bands but the QABand
def radiometric_correction(image: ee.Image , sr_bands_list = ['SR_B1','SR_B2','SR_B3','SR_B4','SR_B5','SR_B6','SR_B7']):
    """
    Applies radiometric correction to the surface reflectance (SR) bands of an input image, and leaves other bands unchanged.

    Args:
        image: An ee.Image object representing the input image.
        sr_bands_list: A list of strings representing the names of the surface reflectance bands to be corrected.

    Returns:
        An ee.Image object with the radiometrically corrected SR bands added as new bands to the input image.
    """
    sr_bands = image.select(sr_bands_list).multiply(2.75e-05).add(-0.2)
    image = image.addBands(sr_bands, None, True)
    return image

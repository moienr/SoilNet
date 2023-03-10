""" A collection of utility functions for working with Earth Engine (EE) in Python.
"""

import ee
import geemap
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





# Function to get the Ratio of ones to total pixels
def get_mask_ones_ratio(mask:ee.Image, band_name="QA_PIXEL", scale = 30):
    """
    Function to get the ratio of ones to total pixels in an Earth Engine image mask.

    Args:
    -----
        `mask` (ee.Image): An Earth Engine image mask.
        `scale` (int, optional): The scale to use for reducing the image. Defaults to 30.

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
    ones = stats.get(band_name + '_sum')
    total = stats.get(band_name + '_count')

    # Compute the ratio of ones to total pixels
    ratio = ee.Number(ones).divide(total)
    

    # Return the ratio
    return ratio
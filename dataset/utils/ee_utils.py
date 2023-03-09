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
    
    
class CloudMask:
    """
    A class that creates a cloud mask for Earth Engine Images with bit Quality Band (e.g. Landsat, sentinel, etc.)
    Takes in the the bit numbers of the cloud and cloud shadow bands in the Quality Band.
    
    Usage:
    ------
    ```
    cm = CloudMask()
    cloud_mask, cloud_shadow, cloud_or_shadow = cm(img)
 
    ```
    """
    def __init__(self, pixel_quality_band='QA_PIXEL', cloud_bit = 3, cloud_shadow_bit = 4, cloud_confidence_bit = 8, cloud_shadow_confidence_bit = 10):
        """
        Parameters:
        -----------
            pixel_quality_band: str, optional (default='QA_PIXEL')
                The name of the band in the image that contains pixel quality information.

            cloud_bit: int, optional (default=3)
                The bit position for the cloud mask in the pixel quality band.

            cloud_shadow_bit: int, optional (default=4)
                The bit position for the cloud shadow mask in the pixel quality band.

            cloud_confidence_bit: int, optional (default=8)
                The bit position for the cloud confidence mask in the pixel quality band.

            cloud_shadow_confidence_bit: int, optional (default=10)
                The bit position for the cloud shadow confidence mask in the pixel quality band.
            
            * defualt values are for Landsat 8 Level 2 SR
        """
        self.pixel_quality_band = pixel_quality_band
        self.cloud_bit = cloud_bit
        self.cloud_shadow_bit = cloud_shadow_bit
        self.cloud_confidence_bit = cloud_confidence_bit
        self.cloud_shadow_confidence_bit = cloud_shadow_confidence_bit

    def __call__(self, img: ee.Image):
        """Takes an ee.Image and returns the cloud, cloud shadow and  cloud_or_cloudShadow mask

        Args:
            img (ee.Image): An ee.Image object containing a pixel quality band. (e.g. 'QA_PIXEL' of Landsat8 SR)

        Returns:
            tuple: A tuple containing the cloud mask, cloud shadow mask, and the combined mask. (ee.Image, ee.Image, ee.Image)
        """
        self.qa = img.select(self.pixel_quality_band)
        # Get the pixel values for the cloud, cloud shadow, and snow/ice bits
        cloud = self.qa.bitwiseAnd(1 << self.cloud_bit).And(self.qa.bitwiseAnd(3<<self.cloud_confidence_bit))
        cloud_shadow = self.qa.bitwiseAnd(1 << self.cloud_shadow_bit).And(self.qa.bitwiseAnd(3<<self.cloud_shadow_confidence_bit))
        return cloud, cloud_shadow, cloud.Or(cloud_shadow)

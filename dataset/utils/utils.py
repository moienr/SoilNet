import ee
import geemap
try:
    ee.Initialize()
except Exception as e:
    print("Failed to initialize Earth Engine: ", e)
    print("Maybe try ee.Authenticate() and ee.Initialize() again?")

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


def test_function(function, *args, **kwargs):
    try:
        output = function(*args, **kwargs)
        print('Test passed!')
        print(output)
    except Exception as e:
        print('Test failed!')
        print(e)


if __name__ == '__main__':
    test_function(get_square_roi, 40.02, -105.25, roi_size=1920)
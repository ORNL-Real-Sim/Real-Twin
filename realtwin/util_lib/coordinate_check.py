'''
##############################################################
# Created Date: Tuesday, May 27th 2025
# Contact Info: luoxiangyong01@gmail.com
# Author/Copyright: Mr. Xiangyong Luo
##############################################################
'''


def check_lon_lat(coordinates):
    """
    Checks if coordinates are in (lon, lat) or (lat, lon) format.

    Args:
        coordinates: A tuple or list containing two numeric values.

    Returns:
        A string indicating the format ("lon, lat" or "lat, lon") or None if invalid.
    """
    if not isinstance(coordinates, (tuple, list)) or len(coordinates) != 2:
        return None

    lon, lat = coordinates[0], coordinates[1]

    if not isinstance(lon, (int, float)) or not isinstance(lat, (int, float)):
      return None

    if -90 <= lat <= 90 and -180 <= lon <= 180:
        return "lon, lat"
    elif -90 <= lon <= 90 and -180 <= lat <= 180:
        return "lat, lon"
    else:
        return None

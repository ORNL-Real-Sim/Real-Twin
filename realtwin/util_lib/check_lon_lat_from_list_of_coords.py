'''
##############################################################
# Created Date: Friday, July 11th 2025
# Contact Info: luoxiangyong01@gmail.com
# Author/Copyright: Mr. Xiangyong Luo
##############################################################
'''


from typing import List, Tuple


def detect_coord_order(coords: List[Tuple[float, float]]) -> str:
    """
    Checks whether a list of coordinate pairs follows (lon, lat) or (lat, lon).
    Returns "lon, lat", "lat, lon", or "ambiguous".
    """
    def is_lon_lat(x, y):
        return -180 <= x <= 180 and -90 <= y <= 90

    def is_lat_lon(x, y):
        return -90 <= x <= 90 and -180 <= y <= 180

    lonlat_count = 0
    latlon_count = 0
    ambiguous_count = 0

    for x, y in coords:
        # Check each ordering exclusively
        lonlat_valid = is_lon_lat(x, y) and not is_lat_lon(x, y)
        latlon_valid = is_lat_lon(x, y) and not is_lon_lat(x, y)

        if lonlat_valid:
            lonlat_count += 1
        elif latlon_valid:
            latlon_count += 1
        else:
            ambiguous_count += 1

    # Decide based on majority or ambiguity
    if lonlat_count > latlon_count and lonlat_count >= ambiguous_count:
        return "lon, lat"
    elif latlon_count > lonlat_count and latlon_count >= ambiguous_count:
        return "lat, lon"
    else:
        return "ambiguous"

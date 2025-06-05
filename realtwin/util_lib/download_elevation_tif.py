##############################################################################
# Copyright (c) 2024, Oak Ridge National Laboratory                          #
# All rights reserved.                                                       #
#                                                                            #
# This file is part of RealTwin and is distributed under a GPL               #
# license. For the licensing terms see the LICENSE file in the top-level     #
# directory.                                                                 #
#                                                                            #
# Contributors: ORNL Real-Twin Team                                          #
# Contact: realtwin@ornl.gov                                                 #
##############################################################################

import requests
from datetime import datetime


def download_elevation_tif_by_bbox(bbox: tuple | list, output_file: str) -> bool:
    """Download elevation data (TIFF) from USGS National Map based on a bounding box.

    Args:
        bbox (tuple | list): A tuple or list containing the bounding box coordinates in the format (min_lon, min_lat, max_lon, max_lat).
        output_file (str): The path to save the downloaded TIFF file.

    Example:
        >>> from realtwin import download_elevation_tif_by_bbox
        >>> bbox = (-112.185, 36.056, -111.705, 36.368)  # Grand Canyon region
        >>> output_file = "elevation_data.tif"
        >>> download_elevation_tif_by_bbox(bbox, output_file)
        >>> # The function will download the elevation data and save it as "elevation_data.tif".

    Raises:
        ValueError: If the bounding box is not a tuple or list, or if it does not contain 4 coordinates.
        Exception: If the API request fails or if no data is available for the specified bounding box.
        Exception: If the download fails or if the file cannot be saved.

    Returns:
        bool: True if the download was successful, False otherwise.
    """

    # check input parameters
    if not isinstance(bbox, (tuple, list)):
        raise ValueError("Bounding box must be a tuple or list.")

    if len(bbox) != 4:
        raise ValueError("Bounding box must contain 4 coordinates (min_lon, min_lat, max_lon, max_lat).")

    # check whether the output file with .tif extension
    if not output_file.endswith(".tif"):
        raise ValueError("Output file must be a TIFF file.")

    # Download elevation data from USGS National Map
    # try:
    # USGS Elevation Data API endpoint
    usgs_api_url = "https://tnmaccess.nationalmap.gov/api/v1/products"

    # Create a bounding box WKT string
    # bbox_geom = box(*bbox)
    # bbox_wkt = bbox_geom.wkt

    # API parameters
    params = {
        "datasets": "National Elevation Dataset (NED) 1/3 arc-second",
        "bbox": ",".join([str(x) for x in bbox]),
        "outputFormat": "json",
        "extentType": "bbox"
    }

    # Make a request to the API
    response = requests.get(usgs_api_url, params=params, timeout=60 * 10) # 10 minutes timeout

    # Check for a successful response
    if response.status_code != 200:
        raise Exception(f"Failed to query USGS API: {response.status_code} {response.text}")

    # Parse the response
    data = response.json()
    if not data.get("items"):
        raise Exception("No data available for the specified bounding box.")

    # Download the first available GeoTIFF file
    # tiff_url = data["items"][0]["downloadURL"]
    tiff_url_list = {item["downloadURL"] for item in data["items"]}
    print("Downloaded URLs: ", tiff_url_list)

    # Extract date from each URL
    def extract_date(url):
        try:
            date_str = url.split('_')[-1].split('.')[0]
            return datetime.strptime(date_str, '%Y%m%d')
        except Exception:
            return None

    # Find the URL with the latest date
    latest_url = max(tiff_url_list, key=extract_date)
    print(f"Downloading GeoTIFF file from: {latest_url}")

    # Download the TIFF file
    tiff_response = requests.get(latest_url, stream=True, timeout=60 * 10)  # 10 minutes timeout
    if tiff_response.status_code == 200:
        # Get the total file size in MB
        total_size = int(tiff_response.headers.get('content-length', 0))
        total_size_mb = total_size / (1024 * 1024)
        print(f"Total file size: {total_size_mb:.2f} MB")

        downloaded_size = 0
        with open(output_file, "wb") as file:
            # 1 MB chunks
            for chunk in tiff_response.iter_content(chunk_size=1024 * 1024):
                if chunk:  # Filter out keep-alive chunks
                    file.write(chunk)
                    downloaded_size += len(chunk)
                    downloaded_mb = downloaded_size / (1024 * 1024)
                    print(f"Downloaded: {downloaded_mb:.2f} MB", end="\r")

        print(f"\nGeoTIFF file saved as: {output_file}")
    else:
        raise Exception(f"Failed to download GeoTIFF file: {tiff_response.status_code}")
    return None


if __name__ == "__main__":
    # Define bounding box (example: Grand Canyon region)
    bounding_box = (-112.185, 36.056, -111.705, 36.368)

    # Output file path
    output_path = "elevation_data.tif"

    # Download the elevation data
    download_elevation_tif_by_bbox(bounding_box, output_path)

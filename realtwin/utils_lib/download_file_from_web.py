'''
##############################################################
# Created Date: Tuesday, December 31st 2024
# Contact Info: luoxiangyong01@gmail.com
# Author/Copyright: Mr. Xiangyong Luo
##############################################################
'''

import urllib.request


def download_single_file_from_web(url: str, dest_filename: str, chunk_size=1024) -> bool:
    """
    Downloads a large file from a URL in chunks and saves it to the specified destination.

    Args:
        url (str): The URL of the file to download.
        dest_filename (str): filename or path to the filename to save the downloaded file.
        chunk_size (int): Size of each chunk to read in bytes (default: 1024).

    Returns:
        bool: True if the download is successful, False otherwise.
    """
    try:
        with urllib.request.urlopen(url) as response, open(dest_filename, 'wb') as out_file:
            total_size = int(response.getheader('Content-Length', 0))

            if total_size == 0:
                print("  :An error occurred: File size is 0.")
                return False

            downloaded = 0

            print(f"  :Starting download: {url}")
            print(f"  :Total size: {total_size / (1024 * 1024):.2f} MB")

            while chunk := response.read(chunk_size):
                out_file.write(chunk)
                downloaded += len(chunk)
                print(f"\r  :Downloaded: {downloaded / (1024 * 1024):.2f} MB", end="")

    except Exception as e:
        print(f"  :An error occurred: {e}")
        return False
    return True

# Licensed under the Apache License: http://www.apache.org/licenses/LICENSE-2.0
# For details: https://github.com/DLR-SF/sky_imaging/blob/main/NOTICE.txt

"""
This module provides funtionality specific to mobotix cameras.
"""

import re
import logging

import numpy as np
import pandas as pd


def extract_mobotix_meta_data(img_bytes):
    """
    Extract variables from image bytes.

    :param img_bytes: byte array or filename of the image as str or subclass of Path
    :return: Dictionary of image metadata variables
    """
    meta = {}

    try:
        header = img_bytes.decode('utf-8', errors='ignore')  # assume header is never longer than this
        producer = search_header(r"PRD=([\w.-]+)", header, dtype=str)
        if producer != 'MOBOTIX':
            logging.warning(f'No mobotix header found.')
        else:
            meta['name'] = search_header(r"NAM=([\w.-]+)", header, dtype=str)
            meta['timestamp'] = read_timestamp_from_header(header)
            meta['exposure_time'] = search_header(r"EXP=(\d{1,9})", header)
            meta['illuminance'] = search_header(r"LXR=(\d{1,9})", header) / 10
            meta['event.periodic_trigger'] = search_header(r"PEV=tper:PE,(\d{1,9})", header)
            meta['event.last_action_sensor'] = search_header(r"PEV=asen:AS,(\d{1,9})", header)
            meta['event.last_video_motion_1'] = search_header(r"PEV=vmde:VM,(\d{1,9})", header)
            meta['event.last_video_motion_2'] = search_header(r"PEV=vmde:VM2,(\d{1,9})", header)
    except Exception as e:
        logging.warning(f'Error retrieving meta data: {e}')
    return meta

def get_mobotix_meta_data(img_file, max_header_length=5000):
    """
    Load image and read variables from header of a Mobotix image.

    :param img_file: filename of the image as str or subclass of Path
    :param max_header_length: Expected maximum number of bytes in the image header (don't read image further)
    :return: Dictionary of image metadata variables
    """
    with open(img_file, 'rb') as im:
        img_bytes = im.read(max_header_length)
    meta = extract_mobotix_meta_data(img_bytes)
    return meta


def search_header(search_pattern, header, dtype=np.float32):
    """Use re to extract a variable from the mobotix image header"""
    match = re.search(search_pattern, header)
    if match is None:
        logging.warning(f'Reading Mobotix image header, did not find variable with pattern {search_pattern}.')
        return np.nan
    else:
        return dtype(match.groups()[0])


def read_timestamp_from_header(header):
    """Use re to extract image timestamp from the mobotix image header"""
    # Define regular expressions for each component
    date_pattern = r"DAT=(\d{4}-\d{2}-\d{2})"
    time_pattern = r"TIM=(\d{2}:\d{2}:\d{2}\.\d+)"
    timezone_pattern = r"TZN=(GMT[+-]\d+)"

    # Extract components using re.search()
    date_match = re.search(date_pattern, header)
    time_match = re.search(time_pattern, header)
    timezone_match = re.search(timezone_pattern, header)

    datetime_str = f"{date_match.group(1)}T{time_match.group(1)}{timezone_match.group(1).replace('GMT', '')}:00"

    return pd.to_datetime(datetime_str)
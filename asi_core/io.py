# Licensed under the Apache License: http://www.apache.org/licenses/LICENSE-2.0
# For details: https://github.com/DLR-SF/sky_imaging/blob/main/NOTICE.txt

"""
This module provides input/output functionalities, e.g. reading and writing from/to files.
"""

import cv2
import logging
from pathlib import Path


def load_image(img_file, format='rgb'):
    """
    Loads an image from a file and converts it to the specified format.

    :param img_file: Path to the image file.
    :param format: Desired image format ('bgr', 'rgb', or 'gray'). Default is 'rgb'.
    :return: Loaded image as a NumPy array.
    :raises FileNotFoundError: If the file does not exist.
    :raises IOError: If the file cannot be read.
    :raises NotImplementedError: If the specified format is not supported.
    """
    if not Path(img_file).is_file():
        raise FileNotFoundError(f'File {img_file} does not exist.')
    img = cv2.imread(str(img_file))
    if img is None:
        raise IOError(f'File {img_file} could not be read.')
    if format.lower() == 'bgr':
        pass
    elif format.lower() == 'rgb':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif format.lower() == 'gray':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        raise NotImplementedError(f'Format {format} not implemented.')
    return img


def load_images(img_files, format='rgb', ignore_errors=False):
    """
    Loads multiple images from a list of file paths.

    :param img_files: List of image file paths.
    :param format: Desired image format ('bgr', 'rgb', or 'gray'). Default is 'rgb'.
    :param ignore_errors: If True, ignores missing or unreadable files and logs warnings.
                          If False, raises exceptions for such cases. Default is False.
    :return: List of successfully loaded images as NumPy arrays.
    :raises FileNotFoundError: If a file does not exist (unless ignore_errors is True).
    :raises IOError: If a file cannot be read (unless ignore_errors is True).
    :raises Exception: If an unexpected error occurs during loading.
    """
    img_list = []
    for file in img_files:
        try:
            img = load_image(file, format=format)
            img_list.append(img)
        except FileNotFoundError:
            if ignore_errors:
                logging.info(f'File {file} not found. Skipping.')
                continue
            else:
                logging.error(f'File {file} not found.')
                raise
        except IOError:
            if ignore_errors:
                logging.info(f'File {file} could not be read. Skipping.')
                continue
            else:
                logging.error(f'File {file} could not be read.')
                raise
        except Exception as e:
            # Unexpected error
            logging.error(e)
            raise
    return img_list

# Licensed under the Apache License: http://www.apache.org/licenses/LICENSE-2.0
# For details: https://github.com/DLR-SF/sky_imaging/blob/main/NOTICE.txt

"""
This module provides functionality for all-sky imagers.
"""

import os
import cv2
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from fastcore.parallel import parallel
import logging

from asi_core.io import load_image, load_images
from asi_core.basics import get_image_files


def process_directory(directory, save_dir):
    """
    Processes a directory of images by grouping them into short time intervals and creating HDR images.

    :param directory: Path to the directory containing images.
    :param save_dir: Path to the directory where HDR images will be saved.
    """
    # Get all files with the _80.jpg suffix
    # unprocessed_files = glob.glob(os.path.join(directory, "*.jpg"))
    unprocessed_files = get_image_files(directory)
    # unprocessed_files = natsorted(unprocessed_files)
    # ic(unprocessed_files)
    unprocessed_timestamps = [datetime.strptime(os.path.basename(file).split('_')[0], "%Y%m%d%H%M%S") for file in
                              unprocessed_files]

    unprocessed_pairs = [list(x) for x in zip(unprocessed_files, unprocessed_timestamps)]
    while len(unprocessed_pairs) > 0:
        # ic(unprocessed_pairs)
        current_pair = unprocessed_pairs.pop(0)
        # ic(current_pair)
        starting_time = current_pair[1]
        current_files = [current_pair[0]]

        while 0 <= (unprocessed_pairs[0][1] - starting_time).total_seconds() <= 20:  # add all images within 10 seconds
            add_pair = unprocessed_pairs.pop(0)
            current_files.append(add_pair[0])
            if len(unprocessed_pairs) == 0:
                break

        images = []
        exposure_times = []
        for current_file in current_files:
            images.append(load_image(current_file))
            current_time = float(os.path.basename(current_file).split("_")[1].split('.')[0])
            exposure_times.append(current_time)

        # ic(current_files)
        base_name = os.path.basename(current_files[0]).split('_')[0]
        create_and_save_hdr(images, exposure_times, os.path.join(save_dir, base_name + '_hdr.jpg'))


def rescale_0_1(image, min_val=None, max_val=None):
    """
    Rescales an image to the range [0,1].

    :param image: Input image as a NumPy array.
    :param min_val: Minimum value for rescaling (optional).
    :param max_val: Maximum value for rescaling (optional).
    :return: Rescaled image with values between 0 and 1.
    """
    if min_val is not None and max_val is not None:
        image = (image - min_val) / (max_val - min_val)
    else:
        image = (image - np.min(image)) / (np.max(image) - np.min(image))
    return np.clip(image, 0, 1)


def correction_oversatured_regions(images, saturation=255):
    """
    Corrects oversaturated regions in a series of images by setting all channels to maximum intensity.

    :param images: List of images as NumPy arrays.
    :param saturation: Saturation threshold (default: 255).
    :return: Tuple of corrected images and a mask indicating non-oversaturated regions.
    """
    # the HDR algorithms will lead to ugly results if one of the channels is saturated (i.e. equal 255), while the others are not.
    # Thus, I check if one of the channels is saturated (or close to it: >= 254) and set all channels to 255.
    all_mask = np.ones_like(images[0])
    for image in images:
        if saturation < 255:
            image = image / saturation * 255
        mask = np.max(image, axis=2) >= 254
        mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        image[mask] = 255
        all_mask[mask == False] = 0
    return images, all_mask


def merge_exposure_series(img_series, exposure_times, algorithm='mertens', saturation=255, filetype='.jpg'):
    """
    Merges a series of images with different exposures into an HDR image.

    :param img_series: List of images as NumPy arrays.
    :param exposure_times: List of exposure times corresponding to the images.
    :param algorithm: HDR merging algorithm ('mertens' or 'debevec', default: 'mertens').
    :param saturation: Saturation threshold for correction (default: 255).
    :param filetype: Output file format ('.jpg', '.jp2', or '.png').
    :return: Merged HDR image as a NumPy array.
    """
    exposure_times = np.array(exposure_times, dtype=np.float32)
    if algorithm == 'mertens':
        merge_mertens = cv2.createMergeMertens()
        merged_imgs = merge_mertens.process(img_series, exposure_times)  # ,response)
        fmin, fmax = 0, 1.5
    elif algorithm == 'debevec':
        img_series, all_mask = correction_oversatured_regions(img_series, saturation=saturation)
        calibrate = cv2.createCalibrateDebevec()
        response = calibrate.process(img_series, exposure_times)
        # the raw hdr values are... well I don't know what exactly they are, but they are low. the exact values are heuristics
        fmin, fmax = 0.00001, 0.03
        merge_debevec = cv2.createMergeDebevec()
        merged_imgs = merge_debevec.process(img_series, exposure_times, response)
        merged_imgs[all_mask == 1] = np.max(merged_imgs)
    else:
        raise NotImplementedError(f'Algorithm {algorithm} not implemented')

    # Scale between 0 and 1
    merged_imgs = rescale_0_1(merged_imgs, min_val=fmin, max_val=fmax)

    # this is standard gamma correction. improves translation from hdr to jpg
    merged_gamma_corrected = merged_imgs ** (1 / 2)

    if filetype == '.jpg':
        merged_rescale = (merged_gamma_corrected * 255).astype(np.uint8)
    elif filetype == '.jp2' or filetype == '.png':
        merged_rescale = (merged_gamma_corrected * 65535).astype(np.uint16)
    else:
        raise ValueError(f'Unsupported file type {filetype}')
    return merged_rescale


def create_and_save_hdr(img_series, exposure_times, output_path, **kwargs_merging):
    """
    Creates an HDR image from a series of exposures and saves it to a file.

    :param img_series: List of images as NumPy arrays.
    :param exposure_times: List of exposure times corresponding to the images.
    :param output_path: Path where the HDR image will be saved.
    :param kwargs_merging: Additional parameters for the merging function.
    """
    filetype = Path(output_path).suffix
    merged_img = merge_exposure_series(img_series=img_series, exposure_times=exposure_times, filetype=filetype,
                                       **kwargs_merging)
    if filetype == 'png':
        cv2.imwrite(str(output_path), merged_img, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
    else:
        cv2.imwrite(str(output_path), merged_img)
    logging.info(f"Saved HDR image to {output_path}")


def process_timestamp(timestamp_group, root_dir, target_dir):
    """Process all images corresponding to a single timestamp."""
    try:
        timestamp, image_paths = timestamp_group
        exposure_times = image_paths.index.get_level_values(1)
        if len(image_paths) < 3: return False
        # Ensure images are sorted properly
        images = load_images(image_paths.sort_index().apply(lambda x: Path(root_dir) / Path(x)))
        # Construct output path based on relative image paths
        relative_path = Path(image_paths.iloc[0]).parent
        output_path = target_dir / relative_path / timestamp.strftime('%Y%m%d%H%M%S_hdr.jpg')
        output_path.parent.mkdir(parents=True, exist_ok=True)

        create_and_save_hdr(images, exposure_times, output_path)
    except:
        return False
    return True


def process_hdr_series(asi_files: pd.Series, root_dir: str, target_dir: str, n_workers=0):
    """
    Process a multi-index Pandas Series containing image paths in parallel to generate HDR images.

    :param asi_files: A multi-index Pandas Series where the index consists of timestamps and exposure times, and the values contain relative image paths.
    :type asi_files: pandas.Series
    :param root_dir: The root directory containing the source images.
    :type root_dir: str
    :param target_dir: The target directory where the generated HDR images will be stored.
    :type target_dir: str
    :param n_workers: The number of parallel workers to use for processing. Defaults to 0 (no parallelism).
    :type n_workers: int, optional
    :return: A Pandas Series with timestamps as the index and generated HDR file paths as values.
    :rtype: pandas.Series
    """
    target_dir = Path(target_dir)

    # Group images by the primary timestamp index and process in parallel
    timestamps = asi_files.index.get_level_values(0).unique()
    results = parallel(process_timestamp, asi_files.groupby(level=0), root_dir=root_dir, target_dir=target_dir,
                       n_workers=n_workers, total=len(timestamps), progress=True)
    return pd.Series(results, index=timestamps, name='created_hdr')

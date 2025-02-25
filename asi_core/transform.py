# Licensed under the Apache License: http://www.apache.org/licenses/LICENSE-2.0
# For details: https://github.com/DLR-SF/sky_imaging/blob/main/NOTICE.txt

"""
This module implements image transformations that can be applied to all-sky images.
"""

import numpy as np
from PIL import Image


def check_image_array_dimensions(array, height, width):
    """
    Determines the dimensions (N, H, W, C) of an input array representing 
    a single image or a batch of images.

    :param array: np.ndarray, the input array to analyze.
    :param height: int, the known height of the images.
    :param width: int, the known width of the images.
    :return: tuple (N, H, W, C) where N and/or C may be None.
    """
    if not isinstance(array, np.ndarray):
        raise ValueError(f"Expected input to be a numpy array, got {type(array)}.")

    if array.ndim == 2:  # Single grayscale image [HxW]
        if array.shape == (height, width):
            return None, height, width, None
        else:
            raise ValueError(f"Expected shape (H, W) but got {array.shape}.")

    elif array.ndim == 3:  # Could be [HxWxC] (single color image) or [NxHxW] (batch of grayscale images)
        if array.shape[:2] == (height, width):  # Single image with channels
            return None, height, width, array.shape[2]  # C is array.shape[2]
        elif array.shape[1:] == (height, width):  # Batch of grayscale images
            return array.shape[0], height, width, None  # N is array.shape[0]
        else:
            raise ValueError(f"Invalid shape {array.shape} for known H={height} and W={width}.")

    elif array.ndim == 4:  # Batch of images [NxHxWxC]
        if array.shape[1:3] == (height, width):
            return array.shape[0], height, width, array.shape[3]  # N and C are defined
        else:
            raise ValueError(f"Invalid shape {array.shape} for known H={height} and W={width}.")

    else:
        raise ValueError(f"Invalid array dimensions {array.ndim}. Expected 2D, 3D, or 4D array.")


def resize_image_batch(image_batch, resize):
    """
    Resize a batch of images.

    :param image_batch: np.array of shape [N, H, W], [N, H, W, 1], or [N, H, W, C].
    :param resize: tuple (H, W) representing the desired dimensions.
    :return: np.array of resized batch.
    """
    if image_batch.ndim not in {3, 4}:
        raise ValueError(f"Expected input 'batch' to be a 3D or 4D array, got shape {image_batch.shape}.")

    resized_batch = np.array([resize_image(image, resize) for image in image_batch])
    return resized_batch


def resize_image(image, resize):
    """
    Resize a single image.

    :param image: np.array of shape [HxW], [HxWx1], or [HxWxC].
    :param resize: tuple (H, W) representing the desired dimensions.
    :return: np.array of resized image.
    """
    if image.ndim == 2 or (image.ndim == 3 and image.shape[-1] == 1):  # Grayscale or mask
        interpolation = Image.NEAREST
    else:  # Color image
        interpolation = Image.BICUBIC
    resized_image = np.asarray(Image.fromarray(image).resize(resize, interpolation))

    # Preserve the last dimension for grayscale images
    if image.ndim == 3 and image.shape[-1] == 1:
        resized_image = resized_image[..., np.newaxis]

    return resized_image.astype(image.dtype)


def mask_image_batch(image_batch, camera_mask, assign_to_masked_pxls=0):
    """
    Applies a camera mask to a batch of images.

    :param image_batch: np.array of images [NxHxWxC] or [NxHxW].
    :param camera_mask: np.array of camera mask [HxW].
    :param assign_to_masked_pxls: value to assign to masked-out pixels.
    :return: np.array of same shape as batch where non-masked region is set to zero.
    """
    n, h, w = image_batch.shape[:3]
    assert camera_mask.shape == (h, w), \
        f'False camera mask used (size of mask ({camera_mask.shape}) does not match ASI size ({h, w})).'
    if len(image_batch.shape) == 3:
        return np.where(camera_mask[np.newaxis, ...], image_batch, assign_to_masked_pxls)
    else:
        return np.where(camera_mask[np.newaxis, :, :, np.newaxis], image_batch, assign_to_masked_pxls)


def mask_image(image, camera_mask, assign_to_masked_pxls=0):
    """
    Applies a camera mask to a single image.

    :param batch: np.array of images [NxHxWxC] or [NxHxW].
    :param camera_mask: np.array of camera mask [HxW].
    :param assign_to_masked_pxls: value to assign to masked-out pixels.
    :return: np.array of same shape as batch where non-masked region is set to zero.
    """
    if len(image.shape) == 2:
        return np.where(camera_mask, image, assign_to_masked_pxls)
    else:
        return np.where(camera_mask[..., np.newaxis], image, assign_to_masked_pxls)


def asi_index_cropping(asi, crop_x, crop_y, channel_first=False):
    """
    Crops a given array along the x and y dimensions.

    :param asi: The input image to be cropped.
    :param crop_x: The x-coordinates of the crop.
    :param crop_y: The y-coordinates of the crop.
    :param channel_first: Whether the color channel of the input image is the first dimension.
    :return: The cropped array
    """
    if channel_first:
        if len(asi.shape) == 3:
            cropped_asi = asi[:, crop_x, crop_y]
        elif len(asi.shape) == 4:
            cropped_asi = asi[:, :, crop_x, crop_y]
        else:
            cropped_asi = asi[crop_x, crop_y]
    else:
        if len(asi.shape) == 3:
            cropped_asi = asi[crop_x, crop_y, :]
        elif len(asi.shape) == 4:
            cropped_asi = asi[:, crop_x, crop_y, :]
        else:
            cropped_asi = asi[crop_x, crop_y]
    return cropped_asi


def asi_undistortion(asi, lookup_table):
    """
    Undistorts ASI array according to mapping in lookup_table.

    :param asi: np.array of image(s) [NxWxHxC] or [WxHxC]
    :param look_up_table: dict
           lookup_table['mapx']: ndarray [W'xH']
           lookup_table['mapy']: ndarray [W'xH']
    :return: np.array of undistorted asi [NxW'xH'xC] (or [W'xH'xC])
    """
    if len(asi.shape) == 3:
        asi_undistorted = asi[lookup_table['mapy'].astype(int), lookup_table['mapx'].astype(int), :]
    elif len(asi.shape) == 4:
        asi_undistorted = asi[:, lookup_table['mapy'].astype(int), lookup_table['mapx'].astype(int), :]
    else:
        raise ValueError('Argument "asi" has invalid shape.')
    return asi_undistorted


def get_zenith_cropping(elevation_matrix, min_ele=0):
    """
    Generates a cropping mask based on the maximum elevation angle.

    :param elevation_matrix: ndarray
        A matrix of elevation values for each pixel.
    :param min_ele: float, optional
        The minimum elevation angle in degrees. Defaults to 0.
    :return:
        crop_mask_x (slice)
            The x-coordinate slice for cropping.
        crop_mask_y (slice)
            The y-coordinate slice for cropping.
    """
    # Replace NaN values with negative infinity for comparison
    elevation_matrix[np.isnan(elevation_matrix)] = -np.inf

    # Get maximum elevation angle
    max_i = np.argmax(elevation_matrix)
    x_max_ele, y_max_ele = np.unravel_index(max_i, elevation_matrix.shape)

    # Find corresponding min elevation with fixed x-coordinate
    y_min_ele = np.argmax(elevation_matrix[x_max_ele, :] > np.deg2rad(min_ele))

    # Find corresponding min elevation with fixed y-coordinate
    x_min_ele = np.argmax(elevation_matrix[:, y_max_ele] > np.deg2rad(min_ele))

    # Set cropping value (half edge length) to max distance
    if abs(y_max_ele - y_min_ele) > abs(x_max_ele - x_min_ele):
        crop_value = abs(y_max_ele - y_min_ele)
    else:
        crop_value = abs(x_max_ele - x_min_ele)

    # Check that target cropping area does not exceed original image size
    if crop_value > elevation_matrix.shape[0] / 2:
        crop_value = int(np.floor(elevation_matrix.shape[0] / 2))
    if crop_value > elevation_matrix.shape[1] / 2:
        crop_value = int(np.floor(elevation_matrix.shape[1] / 2))

    # If max elevation point is not centered, shift center point of cropped image
    if x_max_ele + crop_value > elevation_matrix.shape[0]:
        x_center = elevation_matrix.shape[0] - crop_value
    elif x_max_ele - crop_value < 0:
        x_center = crop_value
    else:
        x_center = x_max_ele

    if y_max_ele + crop_value > elevation_matrix.shape[1]:
        y_center = elevation_matrix.shape[1] - crop_value
    elif y_max_ele - crop_value < 0:
        y_center = crop_value
    else:
        y_center = y_max_ele

    crop_mask_x = slice(x_center - crop_value, x_center + crop_value)
    crop_mask_y = slice(y_center - crop_value, y_center + crop_value)

    return crop_mask_x, crop_mask_y


def get_mask_cropping(camera_mask):
    """
    Crops ASI array according to camera mask.

    :param camera_mask: ndarray of camera mask [WxH] (binary -> 1 eqauls mask pixel)
    :return:
        crop_mask_x (slice)
            The x-coordinate slice for cropping.
        crop_mask_y (slice)
            The y-coordinate slice for cropping.
    """

    width, height = camera_mask.shape
    # Check for each column if there is any mask pixel and store the corresponding indices (x-dim)
    # First and last indices (x-dim) correspond to the first and last column that contain one or more mask pixels
    mask_indices_x = np.where(np.any(camera_mask > 0, axis=1))[0]
    border_left, border_right = mask_indices_x[0], mask_indices_x[-1]
    # Check for each row if there is any mask pixel and store the corresponding indices (y-dim)
    # First and last indices (y-dim) correspond to the first and last row that contain one or more mask pixels
    mask_indices_y = np.where(np.any(camera_mask > 0, axis=0))[0]
    border_top, border_bottom = mask_indices_y[0], mask_indices_y[-1]
    # Get maximum horizontal and vertical sizes (in pixels) of mask
    delta_x = border_right - border_left + 1
    delta_y = border_bottom - border_top + 1
    # Define center of mask based on half the size of the horizontal/vertical lengths
    center_x = int(np.floor(delta_x / 2)) + border_left
    center_y = int(np.floor(delta_y / 2)) + border_top
    # Get "radius" to crop around center
    radius = min(center_x, center_y, height - center_y, width - center_x, int(np.floor(max(delta_x, delta_y) / 2)))

    x_min = max(0, center_x - radius)
    x_max = min(width, center_x + radius)
    y_min = max(0, center_y - radius)
    y_max = min(height, center_y + radius)
    crop_mask_x = slice(x_min, x_max)
    crop_mask_y = slice(y_min, y_max)
    return crop_mask_x, crop_mask_y

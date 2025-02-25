# Licensed under the Apache License: http://www.apache.org/licenses/LICENSE-2.0
# For details: https://github.com/DLR-SF/sky_imaging/blob/main/NOTICE.txt

"""
This module provides functions to visualize masks in all-sky imagers, like detected cloud layers.
"""

import numpy as np
from PIL import Image, ImageDraw

from asi_core.asi_analysis import get_saturated_mask


def overlay_mask(img, mask, mask_color=(255, 0, 0), alpha=0.5, asarray=True):
    """
    Overlays a binary mask onto an image with a specified color and transparency.

    :param img: The input image, either as a NumPy array or a PIL Image.
    :param mask: A binary mask (NumPy array) where nonzero values indicate the mask region.
    :param mask_color: A tuple representing the RGB color of the mask (default is red: (255, 0, 0)).
    :param alpha: A float (0 to 1) that controls the transparency of the overlay (default is 0.5).
    :param asarray: If True, returns the result as a NumPy array; otherwise, returns a PIL Image.
    :return: The image with the overlaid mask, either as a NumPy array or a PIL Image.
    """
    mask_image = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    mask_image[mask] = mask_color
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    blended_img = Image.blend(img, Image.fromarray(mask_image), alpha)
    if asarray:
        blended_img = np.asarray(blended_img)
    return blended_img


def create_saturation_mask_image(img, camera_mask=None, text_position=(10, 10), font_size=40, asarray=True):
    """
    Create a saturation mask image by identifying and overlaying saturated regions on the input image.

    :param img: The input image as a NumPy array.
    :type img: numpy.ndarray
    :param camera_mask: An optional binary mask specifying the valid region of the image (default is None, meaning all pixels are considered valid).
    :type camera_mask: numpy.ndarray, optional
    :param text_position: The position (x, y) where the saturation percentage text is drawn on the image (default is (10, 10)).
    :type text_position: tuple[int, int], optional
    :param font_size: The font size of the saturation percentage text (default is 40).
    :type font_size: int, optional
    :param asarray: Whether to return the output as a NumPy array (default is True). If False, returns a PIL Image.
    :type asarray: bool, optional
    :return: The image with the saturation mask overlay, either as a NumPy array or a PIL Image.
    :rtype: numpy.ndarray or PIL.Image.Image
    """
    if camera_mask is None:
        camera_mask = np.ones(img.shape[:2])
    img[camera_mask == 0] = 0
    sat_mask = get_saturated_mask(img)
    pct_sat = sat_mask.sum() / camera_mask.sum()
    blended_img = overlay_mask(img, sat_mask, asarray=False)
    draw = ImageDraw.Draw(blended_img)
    draw.text(text_position, f"Share of saturated pixels: {pct_sat:.3f}", fill="white", font_size=font_size)
    if asarray:
        blended_img = np.asarray(blended_img)
    return blended_img
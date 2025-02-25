# Licensed under the Apache License: http://www.apache.org/licenses/LICENSE-2.0
# For details: https://github.com/DLR-SF/sky_imaging/blob/main/NOTICE.txt

"""
This module provides functions to automatically generate camera masks for all-sky imagers.
"""

import numpy as np
import cv2


def adjust_gamma(image, gamma=1.0):
    """
    Apply gamma correction to an image.
    """
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array(
        [((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]
    ).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


def compute_mask(avg_img):
    """
    Create an image mask based on the input image. Input image should be a daily/longterm average image based
    on equalized histogram to enhance contrast
    """

    # 1. Step: Convert image to Gray Scale
    gray_img = cv2.cvtColor(avg_img, cv2.COLOR_BGR2GRAY)

    # 2. Step: Conservative masking of dark pixels
    gray_img[gray_img < 10] = 0

    # 3. Step: Contrast enhancement using CLAHE
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(3, 3))
    lab = cv2.cvtColor(
        avg_img, cv2.COLOR_BGR2LAB
    )  # convert from BGR to LAB color space
    l, a, b = cv2.split(lab)  # split on 3 different channels
    l2 = clahe.apply(l)  # apply CLAHE to the L-channel
    lab = cv2.merge((l2, a, b))  # merge channels
    new_img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)  # convert from LAB to BGR
    clahe = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)  # convert from LAB to GRAY
    clahe[gray_img < 10] = 0

    # 4. Step: Increase contrast (gamma adjustment and histogram equalization)
    gamma_c = 1.3
    gamma = adjust_gamma(clahe, gamma=gamma_c)
    targetval = 120  # Choose a medium color value
    alpha = np.nanmin([1.8, targetval / np.nanmean(gamma)])
    beta = 0
    scaleabs = cv2.convertScaleAbs(gamma, alpha=alpha, beta=beta)

    # 5. Step: Gaussian Blurring to remove noises
    img = cv2.GaussianBlur(scaleabs, (3, 3), cv2.BORDER_DEFAULT)

    # 6. Step: Canny Edge detection
    cminval = 20
    cmaxval = 40
    edges = cv2.Canny(img, cminval, cmaxval, L2gradient=True)
    size = avg_img.shape[0]

    # 7. Step: Detect circular horizon as a next conservative masking
    circles = cv2.HoughCircles(
        scaleabs,
        cv2.HOUGH_GRADIENT,
        1,
        size / 2,
        param1=50,
        param2=20,
        minRadius=int(size / 2) - 100,
        maxRadius=int(size / 2) + 100,
    )

    if circles is not None:
        center = (int(circles[0, 0, 0]), int(circles[0, 0, 1]))
        radius = int(circles[0, 0, 2])
        # Subtract a margin of pixels to decrease the circle
        smaller_radius = radius - 20
        if np.abs(center[0] - size / 2) < 150:
            cv2.circle(edges, center, smaller_radius, 255, 1)
    else:
        print("No Hough circles detected")

    # 8. Step: Apply the maximum filter to strengthen the edges
    k = 5
    iter = 5
    kernel = np.ones((k, k), np.uint8)
    thres = cv2.dilate(edges, kernel, iterations=iter)

    # 9. Step: Smoothen the edges
    thres = cv2.medianBlur(thres, ksize=(2 * k) - 1)

    # 10. Step: Apply the Component analysis function
    analysis = cv2.connectedComponentsWithStats(cv2.bitwise_not(thres), 8, cv2.CV_32S)
    (totalLabels, label_ids, values, centroid) = analysis

    # 11. Step: Find the component corresponding to the inner area
    mask = np.zeros(thres.shape, dtype="uint8")
    xs, ys = int(mask.shape[0] / 2), int(mask.shape[1] / 2)
    lid = label_ids[xs, ys]
    componentMask = (label_ids == lid).astype("uint8") * 255
    # Final Step: Binary masking
    mask = cv2.bitwise_or(mask, componentMask)
    return mask


def aggregate_images(img_list, gray_scale=False, equalization=False, blur=False):
    """
    Process a list of images by applying optional grayscale conversion, histogram equalization,
    and blurring. Computes the average and standard deviation of the processed images.

    :param img_list: List of input images as NumPy arrays.
    :type img_list: list of numpy.ndarray
    :param gray_scale: Whether to convert images to grayscale (default is False).
    :type gray_scale: bool, optional
    :param equalization: Whether to apply histogram equalization to enhance contrast (default is False).
    :type equalization: bool, optional
    :param blur: Whether to apply a blurring filter to reduce noise (default is False).
    :type blur: bool, optional
    :return: A dictionary containing processed images, their average, and standard deviation.
    :rtype: dict
    """

    assert len(img_list) > 0, f'Empty list passed.'

    images = []
    for image in img_list:
        if gray_scale: image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if equalization:
            targetval = 120  # Choose a medium color value
            alpha = targetval / np.nanmean(image)
            image = cv2.convertScaleAbs(image, alpha=alpha, beta=0)
        if blur: image = cv2.blur(image, (3,3))
        images.append(image.astype('uint8')[None,...])
    images = np.vstack(images)

    return_dict = {
        'images': images,
        'avg_image': np.mean(images, axis=0).astype(np.uint8),
        'std_image': np.std(images, axis=0).astype(np.uint8)}

    return return_dict

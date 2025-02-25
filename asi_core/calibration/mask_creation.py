# Licensed under the Apache License: http://www.apache.org/licenses/LICENSE-2.0
# For details: https://github.com/DLR-SF/sky_imaging/blob/main/NOTICE.txt

"""
A small tool to create ASI masks manually.
"""

import cv2
import numpy as np
import os
import re
import pathlib
import scipy
import argparse

from asi_core import config


def adjust_gamma(image, gamma=1.0):
    """
    Only for improved visibility, for radiometric evaluations reconsider

    Taken from https://pyimagesearch.com/2015/10/05/opencv-gamma-correction/

    Build a lookup table mapping the pixel values [0, 255] to their adjusted gamma values and apply it.

    :param image: Input RGB image
    :param gamma: Gamma scalar parameter
    :return: Gamma-corrected image
    """
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


class ObstacleMaskDetection:
    """
    Handles the manual creation of an ASI mask to obscure obstacles in the ASI's field of view
    """
    def __init__(self):
        cfg = config.get('ObstacleMaskDetection')
        self.max_intensity = cfg['max_intensity']
        self.params_cv_detection = cfg['cv_detection']
        self.image_pxl_size = cfg['image_pxl_size']
        self.image_path = cfg['img_path']

        self.save_name = re.split(r'\.jpeg|\.jpg|\.png', os.path.basename(self.image_path))[0]

        self.orig_img = cv2.imread(self.image_path)
        self.orig_img = adjust_gamma(self.orig_img, gamma=2.2)
        self.mask = np.zeros_like(self.orig_img[:, :, 0])

        self.gui_add_to_mask = [[]]
        self.gui_remove_from_mask = [[]]
        self.gui_previous_event = None
        
    def detect_mask_cv(self, params=None):
        """
        Applies computer vision methods to automatically detect a mask of obstacles obscuring the sky in the ASI image.

        :param params: Configuration parameters to the algorithm
        :return: automatically detected mask, dtype boolean, shape of greyscaled RGB input image
        """
        if params is None:
            params = self.params_cv_detection

        # Convert the image to grayscale
        gray = cv2.cvtColor(self.orig_img, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to smooth out any noise
        blur = cv2.GaussianBlur(gray, params['gaussian_kernel'], cv2.BORDER_CONSTANT)

        # Apply adaptive thresholding to binarize the image
        thresh = cv2.adaptiveThreshold(blur, self.max_intensity, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
                                       params['adaptive_thres_block_size'], params['adaptive_thres_mean_offset'])

        # Apply morphological operations to remove small objects and fill in gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, params['erode_dilate_kernel'])
        erode = cv2.erode(thresh, kernel)
        dilate = cv2.dilate(erode, kernel)

        # Find contours and select the contour with the largest area
        contours, hierarchy = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        sky_contour = max(contours, key=cv2.contourArea)

        # Create a mask from the selected contour
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [sky_contour], 0, self.max_intensity, -1)

        # Approximate the circle
        (center_x, center_y), radius = cv2.minEnclosingCircle(sky_contour)
        center = (int(center_x), int(center_y))
        radius = int(radius)

        # Create a new mask with the approximated circle
        circle_mask = np.zeros_like(gray)
        cv2.circle(circle_mask, center, radius, self.max_intensity, -1)

        # Subtract 16 pixels from the radius and create a smaller circle mask
        smaller_radius = radius - params['margin_horizon']
        smaller_circle_mask = np.zeros_like(gray)
        cv2.circle(smaller_circle_mask, center, smaller_radius, self.max_intensity, -1)

        # Combine the original mask and the smaller circle mask
        mask = cv2.bitwise_and(mask, smaller_circle_mask)
        mask[gray < 5] = 0
        # Erode mask boundary by small margin
        kernel = np.ones(params['erode_dilate_kernel'], np.uint8)

        self.mask = cv2.erode(mask, kernel, iterations=2)

    def click_and_crop(self, event, x, y, f, cb):
        """
        From user clicks polygons are created indicating image areas to be masked or not.
        """

        # if the left mouse button was clicked, record the starting (x, y)
        if event in [cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN, cv2.EVENT_MBUTTONDOWN]:
            image_copy = self.masked_img.copy()

        if event in [cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN]:
            if not event == self.gui_previous_event:
                self.gui_add_to_mask = [[]]
                self.gui_remove_from_mask = [[]]

            if event == cv2.EVENT_LBUTTONDOWN:
                self.gui_add_to_mask[-1].append([x, y])
                temp_poly = self.gui_add_to_mask
                color = (0, 255, 0)
            elif event == cv2.EVENT_RBUTTONDOWN:
                self.gui_remove_from_mask[-1].append([x, y])
                temp_poly = self.gui_remove_from_mask
                color = (0, 0, 255)

            if len(temp_poly[-1]) == 1:
                print(temp_poly)
                cv2.circle(image_copy, temp_poly[0][0], radius=0, color=color, thickness=4)
            elif len(temp_poly[-1]) > 1:
                # draw a rectangle around the region of interest
                print(temp_poly)
                cv2.polylines(image_copy, np.array(temp_poly), True, color, 2)
            self.gui_previous_event = event

        elif event == cv2.EVENT_MBUTTONDOWN:
            self.apply_polygons()
            self.apply_mask()
            image_copy = self.masked_img.copy()
            cv2.imshow('Correct mask', self.masked_img)

        if event in [cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN, cv2.EVENT_MBUTTONDOWN]:
            cv2.imshow('Correct mask', image_copy)

    def apply_polygons(self):
        """
        Area inside polygon specified by the user is added to or removed from the mask.
        """
        if len(self.gui_remove_from_mask[-1]) > 1:
            temp_mask = np.stack((np.zeros(np.shape(self.mask)), self.mask, np.zeros(np.shape(self.mask))),
                                 axis=2).astype(np.uint8)
            cv2.fillPoly(temp_mask, np.array(self.gui_remove_from_mask), color=(0, 255, 0))
            self.mask = temp_mask[:, :, 1]
        if len(self.gui_add_to_mask[-1]) > 1:
            temp_mask = np.stack((np.zeros(np.shape(self.mask)), 255-self.mask, np.zeros(np.shape(self.mask))),
                                 axis=2).astype(np.uint8)
            cv2.fillPoly(temp_mask, np.array(self.gui_add_to_mask), color=(0, 255, 0))
            self.mask = 255-temp_mask[:, :, 1]

        self.gui_add_to_mask = [[]]
        self.gui_remove_from_mask = [[]]

    def refine_manually(self):
        """
        Lets user specify image regions to be added or removed from mask.
        """
        cv2.namedWindow('Correct mask', cv2.WINDOW_GUI_NORMAL)
        cv2.resizeWindow('Correct mask', self.image_pxl_size, self.image_pxl_size)
        cv2.moveWindow('Correct mask', 20, 20)
        cv2.setMouseCallback('Correct mask', self.click_and_crop)
        cv2.setWindowTitle('Correct mask', 'Draw polygons with left mouse button to add to mask, right button to remove'
                                           ' from mask, middle button or "a"-key to finish a polygon, "c"-key to finish'
                                           ' mask refinement!')
        self.apply_mask()
        cv2.imshow('Correct mask', self.masked_img)

        while True:
            # display the image and wait for a keypress
            key = cv2.waitKey(1) & 0xFF
            # if the 'a' key is pressed, apply the polygons to mask and reset the polygons
            if key == ord('a'):
                self.apply_polygons()
                self.apply_mask()
                cv2.imshow('Correct mask', self.masked_img)
            # if the 'c' key is pressed, break from the loop
            if key == ord("c"):
                break
        # close all open windows
        cv2.destroyAllWindows()

    def apply_mask(self):
        """
        Applies mask to image used for mask creation
        """
        self.masked_img = self.orig_img + 40 * np.stack((np.zeros(np.shape(self.mask)), 1 - self.mask /
                                                         self.max_intensity, np.zeros(np.shape(self.mask))),
                                                        axis=2).astype(np.uint8)

    def save_mask_and_docu(self):
        """
        Saves the mask in legacy format and docu information

        The following is saved:
        - A mat file which contains the mask and the path to the image based on which it was created
        - A jpg image file visualizing the masked areas in the original image
        """
        self.apply_mask()
        cv2.imwrite('masked_' + self.save_name + '.jpg', self.masked_img)
        scipy.io.savemat('mask_' + self.save_name + '.mat', {'Mask': {'BW': self.mask.astype('bool'),
                                                                      'RawImage': self.image_path}})


if __name__ == '__main__':
    """
    Run the mask creation based on config file
    
    The following is saved:
    - A mat file which contains the mask and the path to the image based on which it was created
    - A jpg image file visualizing the masked areas in the original image
    
    :param -c: Optional path to config file, if not specified, a config file 'mask_creation_cfg.yaml' is expected in 
        the working directory
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', nargs=1, default=[pathlib.Path.cwd() / 'mask_creation_cfg.yaml'])

    args = parser.parse_args()
    config.load_config(args.config[0])

    detector = ObstacleMaskDetection()

    detector.detect_mask_cv()

    detector.refine_manually()

    detector.save_mask_and_docu()

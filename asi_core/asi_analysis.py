# Licensed under the Apache License: http://www.apache.org/licenses/LICENSE-2.0
# For details: https://github.com/DLR-SF/sky_imaging/blob/main/NOTICE.txt

"""
This module provides functions to analyse all-sky images.
"""

import numpy as np
import cv2

from scipy.spatial.transform import Rotation
from scipy.spatial.distance import cdist
from scipy.ndimage import gaussian_filter, median_filter
from pvlib import solarposition
import logging


def _to_radians(ele, az):
    """ Converts elevation and azimuth angles from degrees to radians."""
    ele_rad = np.deg2rad(ele)
    if az > 270:
        az_rad = (np.pi/2 - np.radians(az)) % np.pi
    else:
        az_rad = np.pi/2 - np.radians(az)
    return ele_rad, az_rad


def polar_to_cart(r, ele, az):
    """Converts polar coordinates to Cartesian coordinates."""
    r_cos_ele = r * np.cos(ele)
    return [
         r_cos_ele * np.cos(az),
         r_cos_ele * np.sin(az),
         r * np.sin(ele)
    ]


def get_sun_pos_in_asi(sun_ele, sun_az, ele_mat, az_mat):
    """
    Determines the pixel position of the sun in an all-sky image.

    :param sun_ele: Sun elevation angle in degrees.
    :param sun_az: Sun azimuth angle in degrees.
    :param ele_mat: Elevation matrix of the all-sky image.
    :param az_mat: Azimuth matrix of the all-sky image.
    :return: (row, col) coordinates of the sun in the image.
    """
    sun_ele, sun_az = _to_radians(sun_ele, sun_az)
    diff_mat_ele = np.abs(ele_mat-sun_ele)
    diff_mat_az = np.abs(az_mat-sun_az)
    diff_mat = diff_mat_ele + diff_mat_az
    sun_pos = np.argwhere(diff_mat == np.nanmin(diff_mat))[0]
    return sun_pos


def compute_sun_dist_map(sun_ele, sun_az, ele_mat, az_mat, apply_filter=False, size=5):
    """
    Computes the distance map from each pixel to the sun in an all-sky image.

    :param sun_ele: Sun elevation angle in degrees.
    :param sun_az: Sun azimuth angle in degrees.
    :param ele_mat: Elevation matrix of the all-sky image.
    :param az_mat: Azimuth matrix of the all-sky image.
    :param apply_filter: Whether to apply median filtering (default: False).
    :param size: Kernel size for median filtering (default: 5).
    :return: Distance map (in degrees) to the sun.
    """
    sun_ele, sun_az = _to_radians(sun_ele, sun_az)
    cart_sun = np.array(polar_to_cart(1, sun_ele, sun_az)).reshape(1,3)
    az_mask = ~np.isnan(az_mat)
    cart_coord = np.array(polar_to_cart(1, ele_mat[az_mask], az_mat[az_mask])).T
    sun_dist_mat = np.nan*np.ones_like(az_mat)
    sun_dist_mat[az_mask] = np.rad2deg(np.arccos(-(cdist(cart_sun, cart_coord, 'cosine') -1)).reshape(-1))
    if apply_filter:
        sun_dist_mat = median_filter(sun_dist_mat, size=size)
    return sun_dist_mat


def compute_cloud_coverage_and_distance_to_sun(seg_mask, cam_mask, sun_dist_map, cloud_value=1):
    """
    Computes cloud coverage and the minimum distance between clouds and the sun.

    :param seg_mask: Segmentation mask of the sky (clouds vs. background).
    :param cam_mask: Camera mask indicating valid pixels.
    :param sun_dist_map: Distance map from each pixel to the sun.
    :param cloud_value: Value representing clouds in the segmentation mask (default: 1).
    :return: Tuple (cloud_coverage, min_dist_cloud, coord_cloud):
             - cloud_coverage: Fraction of the sky covered by clouds.
             - min_dist_cloud: Minimum distance between a cloud pixel and the sun.
             - coord_cloud: Coordinates of the closest cloud pixel to the sun.
    """
    is_cloud = seg_mask == cloud_value
    if is_cloud.sum() > 1:
        cloud_coverage = is_cloud.sum()/cam_mask.sum()
        min_dist_cloud = np.nanmin(sun_dist_map[is_cloud])
        coord_cloud = np.argwhere(sun_dist_map == min_dist_cloud)[0]
    else:
        cloud_coverage = 0.
        min_dist_cloud = np.inf
        coord_cloud = [0, 0]
    return cloud_coverage, min_dist_cloud, coord_cloud


def sph2cart(az, el, r):
    """
    Transform spherical to cartesian coordinates

    :param az: [radian] array of the azimuth angle, over positive x-axis, rotating around z-axis
    :param el: [radian] array of the elevation angle
    :param r: array of the radius
    :return: arrays of the cartesian coordinates x, y, z (same unit as radius)
    """
    rcos_theta = r * np.cos(el)
    x = rcos_theta * np.cos(az)
    y = rcos_theta * np.sin(az)
    z = r * np.sin(el)
    return x, y, z


def cart2sph(x, y, z):
    """
    Transform cartesian to spherical coordinates. See reverse function sph2cart, for further convention.

    :param x: cartesian coordinate x
    :param y: cartesian coordinate y, same unit as x
    :param z: cartesian coordinate z, same unit as x
    :return: arrays of the azimuth angle, elevation angle, radius (same unit as x)
    """
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    el = np.arctan2(z, hxy)
    az = np.arctan2(y, x)
    return az, el, r


def create_perspective_undistortion_LUT(o, sf):
    """
    Create a look-up table for perspective undistortion.

    :param o: OcamModel instance
    :param sf: Scaling factor.

    :return: Two look-up tables (mapx and mapy) for perspective undistortion.
    """
    map_x = np.zeros((o.height, o.width))
    map_y = np.zeros((o.height, o.width))

    n_xc = o.height / 2.0
    n_yc = o.width / 2.0
    n_z = -o.width / sf

    for i in range(o.height):
        for j in range(o.width):
            M = []
            M.append(i - n_xc)
            M.append(j - n_yc)
            M.append(n_z)
            m = o.world2cam(M)
            map_x[i, j] = m[1]
            map_y[i, j] = m[0]

    return map_x, map_y


def create_panoramic_undistortion_LUT(r_min, r_max, o):
    """
    Create a look-up table for panoramic undistortion.

    :param r_min: Minimum radial distance.
    :param r_max: Maximum radial distance.
    :param o: Dictionary containing camera parameters (height, width, xc, yc).
    :return: Two look-up tables (map_x and map_y) for panoramic undistortion.
    """
    map_x = np.zeros((o['height'], o['width']))
    map_y = np.zeros((o['height'], o['width']))

    for i in range(o['height']):
        for j in range(o['width']):
            theta = -(float(j)) / o['width'] * 2 * np.pi
            rho = r_max - float(r_max - r_min) / o['height'] * i
            map_x[i, j] = o['yc'] + rho * np.sin(theta)
            map_y[i, j] = o['xc'] + rho * np.cos(theta)

    return map_x, map_y


def get_sun_dist(az, ele, timestamp, location):
    """
    Calculate the sun distance angle based on azimuth and elevation angles.

    :param az: Azimuth angles (in degrees) of the sun.
    :param ele: Elevation angles (in degrees) of the sun.
    :param timestamp: Specific timestamp for which the solar position is calculated.
    :param location: Dictionary containing the latitude, longitude, and altitude of the location.
    :return: Three values: sun distance angle, sun azimuth angle, and sun elevation angle.
    """

    # Check if datetime_obj is timezone aware
    if timestamp.tzinfo is None or timestamp.tzinfo.utcoffset(timestamp) is None:
        raise ValueError("be careful - timestamp must be timezone aware")

    # Calculate the sun's azimuth and altitude angles
    sun_pos = solarposition.get_solarposition(timestamp, location['lat'], location['lon'], location['alt'])
    sun_ele = sun_pos['apparent_elevation'].iloc[0]
    sun_az = sun_pos['azimuth'].iloc[0]

    r = np.ones(np.size(az))
    x, y, z = sph2cart(np.reshape(az, -1), np.reshape(ele, -1), r)

    rot = Rotation.from_euler('z', sun_az, degrees=True)
    ex1 = rot.apply(np.asarray([x, y, z]).T)

    rot = Rotation.from_euler('x', 90-sun_ele, degrees=True)
    ex1 = rot.apply(ex1)

    az_sun_normal_plane, ele_sun_normal_plane, r_sun_normal_plane = cart2sph(ex1[:, 0], ex1[:, 1], ex1[:, 2])
    sun_dist = np.pi/2 - ele_sun_normal_plane
    sun_dist = np.rad2deg(np.reshape(sun_dist, np.shape(az)))

    return sun_dist, sun_az, sun_ele


def is_circle_contour(contour, aspect_ratio_tolerance=0.1, circularity_threshold=0.8):
    """
    Determines if a contour is circular based on its aspect ratio and circularity.

    :param contour: (numpy.ndarray) The contour to analyze.
    :param aspect_ratio_tolerance: (float, optional) The maximum difference between the aspect ratio of the contour's
        bounding rectangle and 1. Default is 0.1.
    :param circularity_threshold: (float, optional) The minimum circularity of the contour. Default is 0.7.

    :returns: bool, True if the contour is circular, False otherwise.
    """
    _, (major_d, minor_d), _ = cv2.fitEllipse(contour)
    aspect_ratio = float(major_d) / minor_d

    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return False

    circularity = 4 * np.pi * area / (perimeter * perimeter)

    logging.debug(f'Circle properties (requirement), aspect_ratio: {aspect_ratio} (<{1 + aspect_ratio_tolerance});'
                  f' circularity: {circularity} (>{circularity_threshold})')
    is_circle_contour = abs(aspect_ratio - 1) < aspect_ratio_tolerance and circularity > circularity_threshold
    return is_circle_contour, aspect_ratio, circularity


def get_saturated_mask(img, saturation_limit=240, gray_scale=True, channel_dim=-1):
    if gray_scale:
        if len(img.shape) > 2:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        saturation_mask = img >= saturation_limit
    else:
        intensity_sum = np.sum(img, axis=channel_dim)
        intensity_threshold = 3 * saturation_limit
        saturation_mask = intensity_sum >= intensity_threshold
    return saturation_mask

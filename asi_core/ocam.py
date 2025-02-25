# Licensed under the Apache License: http://www.apache.org/licenses/LICENSE-2.0
# For details: https://github.com/DLR-SF/sky_imaging/blob/main/NOTICE.txt

"""
This module provides functionality related to geometric transformations of ASI images in particular image undistortion.

The internal camera model used here was described by:
Scaramuzza, D., et al. (2006). A Toolbox for Easily Calibrating Omnidirectional Cameras. RSJ International Conference
on Intelligent Robots and Systems  Beijing, China.

The external camera orientation used here is defined according to:
Luhmann, T. (2000). Nahbereichsphotogrammetrie: Grundlagen, Methoden und Anwendungen. Heidelberg, Germany, Wichmann
Verlag.
"""

import numpy as np
from scipy.io import loadmat
from scipy.spatial.transform import Rotation
from dataclasses import dataclass
from typing import Tuple


@dataclass
class UndistortionParameters:
    """This dataclass contains all configurable parameters for the undistortion lookup table calculation. The parameters
    are stored with every call of create_undistortion_with_zenith_cropping_LUT to avoid unnecessary recalculations of
    the same lookup table for equal undistortion parameters."""
    eor: np.ndarray = None
    camera_mask: np.ndarray = None
    undistorted_resolution: Tuple = None
    limit_angle: float = 78.0
    subgrid_factor: int = 4

    def __eq__(self, other):
        """
        Compare the object's parameter set the one of another object

        :param other: UndistortionParameters instance
        :return: Check result, True if equal
        """
        return np.equal(self.eor, other.eor).all() and np.equal(self.camera_mask, other.camera_mask).all() \
               and self.undistorted_resolution == other.undistorted_resolution \
               and self.limit_angle == other.limit_angle \
               and self.subgrid_factor == other.subgrid_factor


class OcamModel:
    """
    Defines transformations based on the ocam camera model and stores corresponding parameters and lookup tables for an
    individual camera sample
    """
    def __init__(self, ss, invpol, x_center, y_center, c, d, e, width, height, diameter=None):
        """
        Initialize OcamModel instance.

        :param ss: List-like of floats, polynomial coefficients F(p)=ss[0]*p^0+...+ss[3]*p^3, that are needed to map a
            pixel point from the image coordinate system (2D) onto a point in the camera coordinate system (3D).
        :param invpol: List-like of floats, coefficients of inverse polynomial to ss, best to provide empty list
        :param x_center: float, x coordinate of center point of camera in pixels, starting at 0 (C convention)
        :param y_center: float, y  coordinate of center point of camera in pixels, starting at 0 (C convention)
        :param c: Affine transformation parameter
        :param d: Affine transformation parameter
        :param e: Affine transformation parameter
        :param width: Integer, image width in pixels
        :param height: Integer, image height in pixels
        :param diameter: Float, diameter of exposed image area in pixels
        """
        self.ss = np.squeeze(np.asarray(ss))

        if not ss.dtype.kind == 'f':
            raise Exception('Check your configuration! The polynomial ss must only contain float numbers.')

        if not len(invpol):
            invpol = OcamModel.findinvpoly(ss, width, height)
        self.invpol = np.asarray(invpol)
        self.x_center = x_center
        self.y_center = y_center
        self.diameter = diameter
        self.c = c
        self.d = d
        self.e = e
        self.width = width
        self.height = height
        self.undistortion_parameters = None
        self.undistortion_lookup_table = None

    @staticmethod
    def get_ocam_model_from_mat(filename):
        """Create OcamModel instance loading ocam parameters from .mat file, e.g. generated via Matlab OcamCalib Toolbox

        Remarks:

        - The order of "pol" needs to be flipped such that the coefficients fulfill the polynomial order
                   P(x)=invpol[0]*x^0+...+invpol[N-1]*x^N-2
        - The coordinates of the image center need to be subtracted by 1 to fulfill the C convention (start
                   from 0)

        :param filename: path to .mat file containing ocam model struct
        :return: OcamModel object
        """

        matlab_struct = loadmat(filename)['ocam_model']
        return OcamModel(np.asarray(matlab_struct['ss'][0, 0]), np.flip(matlab_struct['pol'][0, 0].T),
                         float(matlab_struct['xc'] - 1), float(matlab_struct['yc'] - 1),
                         float(matlab_struct['c']), float(matlab_struct['d']), float(matlab_struct['e']),
                         int(matlab_struct['width']), int(matlab_struct['height']))

    @staticmethod
    def get_ocam_model_from_dict(ocam_dict):
        """Create OcamModel instance loading ocam parameters from a dictionary, as read from camera_data.yaml

        All parameters in camera_data.yaml are stored in the right format for usage in Python.

        :param ocam_dict: Dictionary with ocam calibration parameters
        :return: OcamModel object
        """
        if 'invpol' not in ocam_dict:
            ocam_dict['invpol'] = []

        if 'diameter' not in ocam_dict:
            ocam_dict['diameter'] = None

        return OcamModel(np.asarray(ocam_dict['ss']), np.asarray(ocam_dict['invpol']),
                         ocam_dict['xc'], ocam_dict['yc'], ocam_dict['c'], ocam_dict['d'], ocam_dict['e'],
                         ocam_dict['width'], ocam_dict['height'], ocam_dict['diameter'])

    def cam2world(self, point_2d):
        """
        Transform pixel coordinates to specific external cartesian coordinates

        Based on the sfmt matlab function cam2world

        :param point_2d: 2d pixel coordinates
        :type point_2d: np.ndarray, shape (N,2)
        :return: 3d cartesian coordinates
        :rtype: np.ndarray, shape (N,3)
        """

        point_3d = []

        inv_det = 1.0 / (self.c - self.d * self.e)

        xp = inv_det * ((point_2d[:, 0] - self.x_center) - self.d * (point_2d[:, 1] - self.y_center))
        yp = inv_det * (-self.e * (point_2d[:, 0] - self.x_center) + self.c * (point_2d[:, 1] - self.y_center))

        r = np.linalg.norm([xp, yp], axis=0)
        zp = self.ss[0]
        r_i = 1.0

        for i in range(1, len(self.ss)):
            r_i *= r
            zp += r_i * self.ss[i]

        inv_norm = 1.0 / np.linalg.norm([xp, yp, zp], axis=0)

        point_3d.append(inv_norm * xp)
        point_3d.append(inv_norm * yp)
        point_3d.append(inv_norm * zp)

        return np.asarray(point_3d).T

    def world2cam(self, point3D):
        """Map a set of 3-D points in world coordinates to pixels in the camera image.

        :param point3D: Vector of 3D-points in world coordinates
        :type point3D: np.ndarray, shape (N,3)
        :return: Vector of 2D points in camera coordinates (pixel)
        :rtype: np.ndarray, shape (N,2)
        """

        point2D = np.zeros([point3D.shape[0], 2])

        norm = np.linalg.norm(point3D[:, :2], axis=1)  # Calculate the norm over x- and y-coordinates

        # if norm != 0:
        theta = np.arctan(np.divide(point3D[:, 2], norm[:], where=norm[:] != 0), where=norm[:] != 0)
        invnorm = np.divide(1.0, norm[:], where=norm[:] != 0)
        t = theta[:]
        rho = np.ones(point3D.shape[0]) * self.invpol[0].item()
        t_i = 1.0

        for i in range(1, self.invpol.size):
            t_i *= t
            rho += t_i * self.invpol[i]

        x = point3D[:, 0] * invnorm[:] * rho
        y = point3D[:, 1] * invnorm[:] * rho
        # else
        x[np.where(norm[:] == 0)] = 0
        y[np.where(norm[:] == 0)] = 0

        point2D[:, 0] = x[:] * self.c + y[:] * self.d + self.x_center
        point2D[:, 1] = x[:] * self.e + y[:] + self.y_center

        return point2D

    def world2cam_ss(self, point3D):
        """
        Convert 3D world coordinates to 2D pixel coordinates in the camera image using the omni3d2pixel function

        Apply an affine transformation to map the coordinates to the final pixel locations.

        :param point3D: tuple or list, x, y, and z coordinates in world space.
        :return: numpy.ndarray, x and y pixel coordinates of the points in the camera image.
        """
        # Convert 3D world coordinates to 2D pixel coordinates using omni3d2pixel function
        point2D = self.omni3d2pixel(self.ss, point3D, self.width, self.height)

        # Apply affine transformation to map the coordinates to the final pixel locations
        point2D = np.vstack([point2D[0] * self.c + point2D[1] * self.d + self.x_center,
                             point2D[0] * self.e + point2D[1] + self.y_center])
        return point2D.T

    def omni3d2pixel(self, ss, xx, width, height):
        """
        Convert 3D world coordinates to 2D pixel coordinates in the camera image.

        :param ss: numpy.ndarray, coefficients for the polynomial function that relates the image pixel coordinates to
            the 3D world coordinates
        :param xx: numpy.ndarray (2D), world coordinates, where each row corresponds to a point in the 3D space
        :param width: int, width of the camera image
        :param height: int, height of the camera image
        :return: 1D numpy arrays containing the x and y pixel coordinates of the points in the camera image
        """
        # Set very small non-zero values to the elements where xx[:, 0] and xx[:, 1] are both 0
        ind0 = np.where((xx[:, 0] == 0) & (xx[:, 1] == 0))
        xx[ind0, 0] = np.finfo(float).eps
        xx[ind0, 1] = np.finfo(float).eps

        # Compute the m values
        m = xx[:, 2] / np.sqrt(xx[:, 0] ** 2 + xx[:, 1] ** 2)

        rho = []
        # Reverse the order of the coefficients for polynomial
        poly_coef = ss[::-1]

        # Iterate through all m values
        for mj in m:
            poly_coef_tmp = poly_coef.copy()
            poly_coef_tmp[-2] = poly_coef[-2] - mj

            # Find the roots of the polynomial for the given mj value
            rho_tmp = np.roots(poly_coef_tmp.flatten())

            # Extract the real roots
            real_roots = np.extract(np.isreal(rho_tmp), rho_tmp).real
            # Filter the real roots to those within the bounds of the image height
            res = real_roots[np.logical_and(real_roots > 0, real_roots < height)]

            # If no roots are found within the bounds or there are multiple roots, append NaN
            if res.size == 0 or res.size > 1:
                rho.append(np.nan)
            else:
                rho.append(res[0])

        # Convert the list of rho values into a numpy array
        rho = np.array(rho)
        # Compute x and y pixel coordinates using rho values
        x = xx[:, 0] / np.sqrt(xx[:, 0] ** 2 + xx[:, 1] ** 2) * rho
        y = xx[:, 1] / np.sqrt(xx[:, 0] ** 2 + xx[:, 1] ** 2) * rho

        return x, y

    def world2cam_eor(self, eor, object_coo, cam_pos=np.zeros((3,)), use_ss=False):
        """
        Transform coordinates from an external cartesian coordinate system to pixel coordinates.

        :param eor: Vector containing the camera position and orientation.
        :param object_coo: 3D object coordinates with or without point numbers. Either shape (N,3) with (Y, X, Z) in
                           each row or shape (N,4) with (Point_Id, Y, X, Z) in each row.
                           If cam_pos is zero, the origin is in the camera and the following applies:
                                North (1, 0, 0)
                                East (0, 1, 0)
                                Zenith (0, 0, 1)
        :param cam_pos: Offset in world coordinates between camera and origin
        :param use_ss: If True, the slower method using the "forward" polynomial ss is used.
        :return: 2D image coordinates in pixels with or without point numbers depending on input. Origin is located in
                 the upper left corner. If shape is (N,2) data  is interpreted as (X, Y); if (N, 3), data  is
                 interpreted as (Point_Id, X, Y). Increasing X means moving right in image. Increasing Y means moving
                 down in image
        """

        n_coo = np.shape(object_coo)[1]
        if n_coo == 4:
            object_coo_xyz = object_coo[:, 1:]
            point_ids = object_coo[:, 0]
            use_point_ids = True
        elif n_coo == 3:
            object_coo_xyz = object_coo
            use_point_ids = False
        else:
            object_coo_xyz = None
            point_ids = None
            use_point_ids = None
            Exception('Wrong format for object_coo')

        object_coo_xyz = object_coo_xyz[:, [1, 0, 2]]

        rot = Rotation.from_euler('zyx', eor[::-1], degrees=False)
        object_coo_xyz_local = rot.inv().apply(object_coo_xyz) + cam_pos[np.newaxis, :]
        if use_ss:
            image_coo_xy = np.asarray(self.world2cam_ss(object_coo_xyz_local))
        else:
            image_coo_xy = np.asarray(self.world2cam(object_coo_xyz_local))

        image_coo_xy = image_coo_xy[:, ::-1]

        if use_point_ids:
            image_coo = np.concatenate((np.reshape(point_ids, (-1, 1)), image_coo_xy), axis=1)
        else:
            image_coo = image_coo_xy
        return image_coo

    def cam2world_eor(self, eor, image_coo, cam_pos=np.zeros((3,))):
        """
        Transform pixel coordinates to coordinates in an external cartesian coordinate system.

        Note: based on sfmt_matlab function call_cam2world

       :param eor: External orientation as a list or 1-D array of 3 entries with the angles roll, pitch, yaw
            (order and relation to axes should be documented).
       :param image_coo: 2D image coordinates in pixels with or without point numbers depending on input. Origin is
            located in the upper left corner. If shape is (N,2) data  is interpreted as (X, Y); if (N, 3), data is
            interpreted as (Point_Id, X, Y). Increasing X means moving right in image. Increasing Y means moving down
            in image
       :param cam_pos: Offset in world coordinates between camera and origin
       :return: 3D object coordinates with or without point numbers depending on input data. Either shape (N,3) with
            (Y, X, Z) in each row or shape (N,4) with (Point_Id, Y, X, Z) in each row. If cam_pos is zero, the origin is
            in the camera and the following applies:
                North (1, 0, 0)
                East (0, 1, 0)
                Zenith (0, 0, 1)
        """
        # check format of object_coo
        n_coo = np.shape(image_coo)[1]
        if n_coo == 3:
            image_coo_xy = image_coo[:, 1:]
            point_ids = image_coo[:, 0]
            use_point_ids = True
        elif n_coo == 2:
            image_coo_xy = image_coo
            use_point_ids = False
        elif n_coo in [4, 5]:
            image_coo_xy = image_coo[:, 2:4]
            point_ids = image_coo[:, 1]
            use_point_ids = True
        else:
            image_coo_xy = None
            point_ids = None
            use_point_ids = None
            Exception('Wrong format for image_coo')

        image_coo_xy = image_coo_xy[:, ::-1]

        object_coo_xyz_local = np.asarray(self.cam2world(image_coo_xy))

        rot = Rotation.from_euler('zyx', eor[::-1], degrees=False)
        object_coo_xyz = rot.apply(object_coo_xyz_local) - cam_pos[np.newaxis, :]

        object_coo_xyz = object_coo_xyz[:, [1, 0, 2]]

        if use_point_ids:
            object_coo = np.concatenate((np.reshape(point_ids, (-1, 1)), object_coo_xyz), axis=1)
        else:
            object_coo = object_coo_xyz

        return object_coo

    def create_undistortion_with_zenith_cropping_LUT(self, eor, camera_mask, undistorted_resolution=None,
                                                     limit_angle=78.0, subgrid_factor=4):
        """Create a look-up table between pixels of the ASI image and pixels of a undistorted georeferenced map.

        This functions doesn't only undistort images, it also does the EOR correction and cropping around the zenith as
        specified in the function arguments.

        The returned map's size and the map grid cell's size is not defined by this function. It depends on the height
        of the observed object relative to the camera (i.e. cloud base height).

        :param eor: ndarray, shape(3,)
            3-D array with external orientation (euler angles) [Rx, Ry, Rz].
        :param camera_mask: ndarray, shape(H,W)
            Defines masked pixels in origin image. True means that the pixel is mapped onto a valid pixel.
        :param undistorted_resolution: array_like, shape(2,)
            Resolution of undistorted image [H,W].
        :param limit_angle: Angle in degrees from zenith that shall be visible in the undistorted image.
        :param subgrid_factor: Factor by which to increase the resolution of the grid used for undistortion.
            Higher values result in finer grids, improving the precision of the undistortion process.
        :return: lookup_table: dict, look-up table from georeferenced map, centered around camera, to image pixels
                 lookup_table['mapx']: ndarray, shape(H,W), ascending south-north (1st dim) and west-east (2nd dim),
                     each element contains the x-coord (2nd dim) of the image pixel monitoring the grid cell's location
                 lookup_table['mapy']: ndarray, shape(H,W), ascending south-north (1st dim) and west-east (2nd dim),
                     each element contains the y-coord (1st dim) of the image pixel monitoring the grid cell's location
                 lookup_table['is_inside_mask']: ndarray, shape(H,W),
        """

        # Check if undistortion lookup table already has been calculated with given undistortion parameters.
        if self.undistortion_lookup_table is not None \
                and self.undistortion_parameters == UndistortionParameters(eor, camera_mask, undistorted_resolution,
                                                                           limit_angle, subgrid_factor):
            pass

        # Create new undistortion lookup table based on given undistortion parameters.
        else:
            # Safe new undistortion parameters in class
            self.undistortion_parameters = UndistortionParameters(eor, camera_mask,
                                                                  undistorted_resolution,
                                                                  limit_angle, subgrid_factor)

            if undistorted_resolution is not None:
                destination_height = undistorted_resolution[0]
                destination_width = undistorted_resolution[1]
            else:
                destination_height = self.height
                destination_width = self.width

            # Create model of destination image in world coordinates
            limit = np.tan(np.deg2rad(limit_angle))
            shift = (2.0 * limit / destination_width) * (subgrid_factor - 1.0) / (2.0 * subgrid_factor)
            x_vector = np.linspace(-limit - shift, limit + shift, destination_width * subgrid_factor)
            y_vector = np.linspace(-limit - shift, limit + shift, destination_height * subgrid_factor)

            # Create variables to store intermediate results for each subgrid
            image_points_2D = np.zeros([destination_height * destination_width, 2, subgrid_factor ** 2], 'float32')
            is_inside_vector, is_not_masked_vector = np.zeros(
                [2, destination_height * destination_width, subgrid_factor ** 2],
                'bool')
            for idy in range(subgrid_factor):
                for idx in range(subgrid_factor):
                    x_vector_subgrid = x_vector[idx::subgrid_factor]
                    y_vector_subgrid = y_vector[idy::subgrid_factor]
                    subgrid_index = idx + subgrid_factor * idy
                    world_points_3D = np.ones([destination_height * destination_width, 3], 'float32')
                    # Loop row-wise through image to put all destination image points into single vector
                    for row in range(destination_height):
                        world_points_3D[row * destination_width:(row + 1) * destination_width, 0] = x_vector_subgrid[:]
                        world_points_3D[row * destination_width:(row + 1) * destination_width, 1] = y_vector_subgrid[row]
                    # Apply external orientation
                    world_points_3D = np.dot(world_points_3D, Rotation.from_euler('XYZ', eor, degrees=False).as_matrix())
                    # Map each point of the destination image onto a pixel of the origin image
                    pixel_coordinates = self.world2cam(world_points_3D)
                    # Change order of x- and y coordinates to keep opencv convention
                    image_points_2D[:, 0, subgrid_index] = pixel_coordinates[:, 1]
                    image_points_2D[:, 1, subgrid_index] = pixel_coordinates[:, 0]
                    # check if assigned pixels are inside origin image
                    is_inside_vector[:, subgrid_index] = (image_points_2D[:, 0, subgrid_index] <= self.width) * (
                            image_points_2D[:, 1, subgrid_index] <= self.height)
                    # check if assigned pixels are not masked in origin image
                    is_not_masked_vector[:, subgrid_index] = camera_mask[image_points_2D[:, 1, subgrid_index].astype(int),
                                                                         image_points_2D[:, 0, subgrid_index].astype(int)]

            # Calculate final results over the results of all subgrids
            camera_mask_vector_final = np.all(is_inside_vector, axis=1) * np.all(is_not_masked_vector, axis=1)
            # Calculate average of pixel coordinates over all subgrids where mapped pixels lay inside origin image
            mapx_vector_final = np.divide(np.sum(image_points_2D[:, 0, :] * is_inside_vector, axis=1),
                                          np.sum(is_inside_vector, axis=1), where=np.sum(is_inside_vector, axis=1) != 0)
            mapy_vector_final = np.divide(np.sum(image_points_2D[:, 1, :] * is_inside_vector, axis=1),
                                          np.sum(is_inside_vector, axis=1), where=np.sum(is_inside_vector, axis=1) != 0)

            self.undistortion_lookup_table = \
                    {'mapx': np.reshape(mapx_vector_final, [destination_height, destination_width]).astype('float32'),
                    'mapy': np.reshape(mapy_vector_final, [destination_height, destination_width]).astype('float32'),
                    'is_inside_mask': np.reshape(camera_mask_vector_final, [destination_height, destination_width])}

        return self.undistortion_lookup_table

    @staticmethod
    def findinvpoly(ss, width, height):
        """
        Approximate pol from ss. Adapted from Scaramuzza Matlab toolbox.

        :param ss: list-like, ss polynomial coefficients of ocam model
        :param width: int, image width from ocam model
        :param height: int, image height from ocam model
        :return: numpy 1-D array of polynomial coefficients of inverse polynomial to be used in world2cam
        """
        ss = np.squeeze(ss)

        radius = np.sqrt(np.square(width / 2) + np.square(height / 2))

        max_err = np.inf
        N = 0
        while max_err > 0.01:
            N = N + 1

            theta = np.arange(-np.pi / 2, 1.2, 0.01)

            m = np.tan(theta)

            r = []
            poly_coef = ss[::-1]
            poly_coef_tmp = poly_coef.copy()

            for i in range(len(m)):
                poly_coef_tmp[-2] = poly_coef[-2] - m[i]
                rho_tmp = np.roots(poly_coef_tmp)
                res = rho_tmp[np.where((np.imag(rho_tmp) == 0) & (rho_tmp > 0) & (rho_tmp < radius))]

                if len(res) == 1:
                    r = np.append(r, np.real(res[0]))
                else:
                    r = np.append(r, np.inf)

            theta = theta[r != np.inf]
            r = r[r != np.inf]

            pol = np.polyfit(theta, r, N)
            err = abs(r - np.polyval(pol, theta))
            max_err = np.max(err)

        # TODO: reshape done to be compliant with legacy, 1D arrays would be more reasonable for ss, invpol.
        return np.reshape(pol[::-1], (-1, 1))

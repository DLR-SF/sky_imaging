# Licensed under the Apache License: http://www.apache.org/licenses/LICENSE-2.0
# For details: https://github.com/DLR-SF/sky_imaging/blob/main/NOTICE.txt

"""
This module provides functionality for all-sky imagers.
"""

import os
from os import times

import jsonschema
import numpy as np
from PIL import Image
import scipy.io
import yaml
import logging
import cv2
import pandas as pd
import re
import glob
import pytz
from datetime import datetime, date, timedelta
from pathlib import Path
from asi_core.basics import ifnone, assemble_path, fstring_to_re

from asi_core.transform import (
    mask_image_batch,
    resize_image_batch,
    asi_undistortion,
    get_zenith_cropping,
    get_mask_cropping,
    asi_index_cropping,
    check_image_array_dimensions
)

from asi_core.ocam import OcamModel
from asi_core import asi_analysis, config
from asi_core.util.mobotix import extract_mobotix_meta_data


CAMERA_DATA_FILE = os.path.join((os.path.dirname(__file__)), 'camera_data.yaml')
CAMERA_DATA_SCHEMA = Path(__file__).parent / 'camera_data.schema.yaml'


class AllSkyImager:
    img_paths_buffer = {'prev_date': date(1975, 1, 1),
                        'imgs': pd.DataFrame({'timestamp': [], 'exp_time': [], 'path': []})}
    tolerance_timestamp = [timedelta(seconds=-2), timedelta(seconds=12)]
    rel_exp_tol = 0.1  # relative tolerance of an image's exposure time compared to the setting

    """Class for All-Sky Imagers."""
    def __init__(self, camera_data, camera_id=None, image_path=None, tfms=None):
        """
        Initializes the All-Sky Imager class.

        :param camera_data: A dictionary containing the camera data.
        :type camera_data: dict
        :param camera_id: The ID of the camera.
        :type camera_id: str, optional
        :param image_path: The path to the image.
        :type image_path: str, optional
        :param tfms: A dictionary of image transformations.
        :type tfms: dict, optional
        :return: None
        """
        self.id = camera_id
        self.name = camera_data['camera_name']
        self.latitude = camera_data['latitude']  # northing in dec. degrees;
        self.longitude = camera_data['longitude']  # easting in dec. degrees;
        self.altitude = camera_data['altitude']  # meters (according to Google Earth)
        if 'exposure_settings' in camera_data and 'rel_exp_tol' in camera_data['exposure_settings']:
            self.rel_exp_tol = camera_data['exposure_settings']['rel_exp_tol']
        if 'exposure_settings' in camera_data and 'tolerance_timestamp' in camera_data['exposure_settings']:
            self.tolerance_timestamp = [timedelta(seconds=v) for v in
                                        camera_data['exposure_settings']['tolerance_timestamp']]
        config_tz = camera_data['timezone']
        # adapt from ISO 8601 to posix convention
        # ISO 8601: Positive east of Greenwich
        if 'GMT' in config_tz:
            if '+' in config_tz:
                config_tz = config_tz.replace('+', '-')
            elif '-' in config_tz:
                config_tz = config_tz.replace('-', '+')
            config_tz = 'Etc/' + config_tz
        self.img_timezone = pytz.timezone(config_tz)
        self.start_recording = camera_data['mounted']
        self.end_recording = camera_data['demounted']
        self.external_orientation = camera_data['external_orientation']
        self.min_ele_evaluated = camera_data['min_ele_evaluated']
        if image_path is not None:
            self.img_path_structure = image_path
        self.ocam_model = OcamModel.get_ocam_model_from_dict(camera_data['internal_calibration'])
        self.azimuth_mask, self.elevation_mask = \
            self.get_azimuth_elevation(self.ocam_model, self.min_ele_evaluated, self.external_orientation)
        self.height = self.ocam_model.height
        self.width = self.ocam_model.width
        if 'camera_mask_file' in camera_data and camera_data['camera_mask_file'] is not None:
            camera_mask_file = Path(camera_data.get('_basedir', '.')) / camera_data['camera_mask_file']
            self.camera_mask = load_camera_mask(mask_file=camera_mask_file, allow_failure=True)
        if 'exposure_settings' in camera_data and 'exposure_times' in camera_data['exposure_settings']:
            self.exp_times = camera_data['exposure_settings']['exposure_times']

        # Define image transforms for all-sky imager instance
        tfms = ifnone(tfms, {})
        self.resize = tfms.get('resize')
        self.crop = tfms.get('crop')
        self.crop_min_ele = tfms.get('min_ele', 0)
        self.undistort = tfms.get('undistort', False)
        self.undistort_limit_angle = tfms.get('undistort_limit_angle', 78)
        self.apply_camera_mask = tfms.get('apply_camera_mask', False)
        self.apply_elevation_mask = tfms.get('apply_elevation_mask', False)
        self.cropping_indexes = self.get_cropping_indexes()

    @classmethod
    def from_file(cls, camera_data_file, image_path=None, tfms=None):
        """
        Create a list of AllSkyImager instances

        :param camera_data_file: File with a single camera parameter set (specifying camera and time period)
        :param image_path: path from which AllSkyImager instances will load images
        :param tfms: A dictionary of image transformations.
        :return: list of AllSkyImager instances
        """
        camera_data = load_camera_data(camera_data_file=camera_data_file)

        return cls(camera_data, image_path=image_path, tfms=tfms)

    @classmethod
    def from_files(cls, camera_data_dir, camera_name, image_path=None, tfms=None):
        """
        Create a list of AllSkyImager instances

        :param camera_data_dir: Directory containing camera data yaml files (one file per camera and period)
        :param camera_name: Name of the camera for which AllSkyImager instances will be created
        :param image_path: path from which AllSkyImager instances will load images
        :param tfms: A dictionary of image transformations.
        :return: list of AllSkyImager instances
        """
        camera_data = load_camera_data(camera_data_dir=camera_data_dir, camera_name=camera_name)

        ASIs = []
        for cam_id, cam_dict in camera_data.items():
            ASIs.append(cls(cam_dict, image_path=image_path, tfms=tfms))
        return ASIs

    @classmethod
    def from_file_as_dict(cls, camera_data_dir, camera_name, image_path=None):
        """
        Create a dictionary with camera_data parameter sets corresponding to multiple periods of a single camera id

        :param camera_data_dir: Path to folder with camera_data yamls (one parameter set per camera and period)
        :param camera_name: Name of the camera in the camera_data_dir for which the parameters sets will be extracted
        :param image_path: path from which AllSkyImager instances will load images
        :return: dictionary of AllSkyImager/ camera_data parameter sets, one set valid per period indicated by the key,
            each key consists of a tuple of two dates (mounted, unmounted)
        """
        camera_data = load_camera_data(camera_data_file=None, camera_data_dir=camera_data_dir, camera_name=camera_name)

        ASIs = {}
        for cam_id, cam_dict in camera_data.items():
            time_range = (cam_dict['mounted'], cam_dict['demounted'])
            ASIs[time_range] = (cam_dict, cam_id, image_path)

        return {k: v for k, v in sorted(ASIs.items(), key=lambda item: item[0][0])}

    @classmethod
    def from_config(cls):
        """
        Creates a list of AllSkyImager instances, based on the camera name from config file and the camera_data
        collection.

        :return: list of AllSkyImager instances
        """
        cfg_vals = config.get('Camera')
        if 'camera_data_dir' not in cfg_vals.keys():
            cfg_vals['camera_data_dir'] = None
        if 'transforms' not in cfg_vals.keys():
            cfg_vals['transforms'] = None
        return cls.from_files(cfg_vals['camera_data_dir'], cfg_vals['camera_name'],
                              image_path=cfg_vals['img_path_structure'], tfms=cfg_vals['transforms'])

    @classmethod
    def from_config_and_period(cls, evaluated_timestamps=None):
        """
        Creates a single AllSkyImager instance, based on a config file and an evaluated period

        An error is returned if the evaluated period contains a change of the camera setup.

        :param evaluated_timestamps: timezone-aware datetime or pandas series of timezone-aware datetime
        :return: AllSkyImager instance
        """
        potential_cameras = cls.from_config()

        if type(evaluated_timestamps) is datetime:
            evaluated_timestamps = pd.Series([evaluated_timestamps])
        elif not issubclass(type(evaluated_timestamps), pd.core.base.IndexOpsMixin):
            raise Exception('Expected a single datetime or a subclass of pandas.core.base.IndexOpsMixin.')

        potential_cameras = [pc for pc in potential_cameras if pc.end_recording >= evaluated_timestamps.min() and
                             pc.start_recording <= evaluated_timestamps.max()]

        if len(potential_cameras) == 0:
            raise Exception('No camera configuration found for the evaluated period. This was not expected!')
        elif len(potential_cameras) > 1:
            raise Exception('More than one camera configuration found for the evaluated period. This was not expected!')
        else:
            return potential_cameras[0]

    @staticmethod
    def load_image(image_file):
        """
        Load image from file to numpy array.

        :param image_file: Path to the image to be loaded
        :return: Loaded image as numpy array
        """
        return np.asarray(Image.open(image_file))

    @staticmethod
    def save_image(image, image_file):
        """
        Save image from numpy array to file.

        :param image: Image to be saved as numpy array
        :param image_file: Path to save the image to
        """
        Path(image_file).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(image).save(image_file)

    def transform(self, images: [list, np.ndarray]):
        """
        Apply image transformations to all-sky images as specified during initialization.

        Can be used to preprocess images for deep learning models, e.g., to crop, resize or undistort an image.

        :param images: list or array of raw all-sky images or single image.
        :return: array of transformed image(s).
        """
        if type(images) is list:
            images = np.asarray(images)
        n, h, w, c = check_image_array_dimensions(images, height=self.height, width=self.width)
        if n is None:
            images = images[np.newaxis, ...]
        tfmd_images = images
        if self.undistort:
            undistorted_resolution = (self.resize, self.resize) if self.resize else None
            undistortion_lookup_table = self.ocam_model.create_undistortion_with_zenith_cropping_LUT(
                self.external_orientation, self.camera_mask, undistorted_resolution=undistorted_resolution,
                limit_angle=self.undistort_limit_angle)
            tfmd_images = asi_undistortion(tfmd_images, undistortion_lookup_table)
        else:
            if self.apply_camera_mask:
                tfmd_images = mask_image_batch(tfmd_images, self.camera_mask)
            if self.apply_elevation_mask:
                mask_elevation = self.elevation_mask > self.min_ele_evaluated
                tfmd_images = mask_image_batch(tfmd_images, mask_elevation)
            if self.crop is not None:
                tfmd_images = asi_index_cropping(tfmd_images, *self.cropping_indexes)
            if self.resize:
                tfmd_images = resize_image_batch(tfmd_images, resize=(self.resize, self.resize))
        if n is None:
            tfmd_images = tfmd_images[0]
        return tfmd_images

    def check_timestamp(self, timestamp):
        """
        Check if timestamp is within recording time range of camera.

        :param timestamp: Timezone-aware datetime
        :return: Boolean test result
        """
        if self.start_recording <= timestamp <= self.end_recording:
            return True
        else:
            return False

    def get_cropping_indexes(self):
        """
        Generate cropping indexes based on the selected cropping method.

        :return: Tuple of cropping indexes along x and y axes.
        """
        if self.crop is None:
            crop_x, crop_y = None, None
        elif self.crop == 'zenith':
            crop_x, crop_y = get_zenith_cropping(elevation_matrix=self.elevation_mask, min_ele=self.crop_min_ele)
        elif self.crop == 'mask':
            crop_x, crop_y = get_mask_cropping(camera_mask=self.camera_mask)
        else:
            raise ValueError(f'Invalid cropping method {self.crop}')
        logging.info(f'Calculate cropping indices. crop_x: {crop_x}, crop_y: {crop_y}.')
        return crop_x, crop_y

    def get_azimuth_elevation(self, ocam_model, min_ele_mask, external_orientation):
        """
        Get azimuth and elevation angle of every pixel in current image.

        The map depends on actual camera model, camera mask and projection function. Calls Scaramuzzas cam2world
        function to  get the 3-D coordinates of 2-D pixel points. The cartesian coordinates will be transformed to
        spherical ones.

        :param ocam_model: Instance of OcamModel
        :param min_ele_mask: [degree] Elevation angle over horizontal below which image is not evaluated (masked)
        :param external_orientation: 3-entry array of angles indicating external_orientation, see cam2world_eor
        :return: Matrices indicating azimuth and elevation angle for each image pixel
        """
        logging.info('Create azimuth and elevation matrices')

        img_size = (ocam_model.height, ocam_model.width)

        # Get index of pixel which are inside the mask
        flat_idx = np.arange(img_size[0] * img_size[1])
        i_multi, j_multi = np.unravel_index(flat_idx, img_size)

        # Create function input and call function
        image_coord_vec_xy = np.asarray([flat_idx, j_multi, i_multi]).T
        image_coord_yxz = self.ocam_model.cam2world_eor(external_orientation, image_coord_vec_xy)

        # Calc from cart to sph(X, Y, Z) -> (AZ, ELE, R), output is in rad
        az, ele, _ = asi_analysis.cart2sph(image_coord_yxz[:, 2], image_coord_yxz[:, 1], image_coord_yxz[:, 3])

        # Create Matrix
        az_matrix = np.reshape(az, img_size).astype('float32')
        ele_matrix = np.reshape(ele, img_size).astype('float32')

        is_inside_mask = ele_matrix > np.deg2rad(min_ele_mask)
        ele_matrix[~is_inside_mask] = np.nan
        az_matrix[~is_inside_mask] = np.nan

        # Define output
        return az_matrix, ele_matrix

    def get_img_and_meta(self, timestamp=None, exp_time=160, img_path=None):
        """
        Load an image and determine the image path if needed

        :param timestamp: timezone-aware datetime, timestamp of the image
        :param exp_time: [microseconds] Integer, exposure time of the image
        :param img_path: String, path to image: If None it will be determined.
        :return: Dict containing the image array ('img') and meta data ('meta')
        """

        invalid_img_meta = {'img': [], 'meta': np.nan}
        if img_path is None:
            img_path = self.get_img_path(timestamp, exp_time=exp_time)

        if not len(img_path):
            logging.warning(f'No image found for timestamp {timestamp} with exposure time {exp_time} mus.')
            return invalid_img_meta

        with open(img_path, 'rb') as im:
            img_bytes = im.read()

        if not len(img_bytes):
            # Image file empty
            logging.warning(f'Image file was empty: {img_path}')
            return invalid_img_meta

        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, flags=cv2.IMREAD_COLOR)
        if np.shape(img) != (self.ocam_model.height, self.ocam_model.width, 3):
            logging.warning(f'Image did not match expected size: {img_path}')
            return invalid_img_meta

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return {'img': img, 'meta': self.get_meta(img_bytes)}

    @staticmethod
    def get_meta(img_bytes):
        """
        Read metadata from an image byte array or a file.

        Note: This method currently only handles Mobotix cameras. It should be generalized as needed.

        :param img_bytes: Binary encoded image
        :return: Dict, metadata read from image
        """
        try:
            meta = extract_mobotix_meta_data(img_bytes)
            return meta
        except Exception as e:
            logging.warning(f'Error retrieving meta data: {e}')
            return None

    def get_img(self, timestamp, exp_time=0):
        """
        Return an image which complies with the requested timestamp and exposure time.

        :param timestamp: Timezone-aware datetime, requested timestamp
        :param exp_time: [microseconds] requested exposure time
        :return: numpy array, loaded image
        """
        img_path = self.get_img_path(timestamp, exp_time=exp_time)
        if len(img_path):
            img = cv2.imread(img_path)
        else:
            img = []
        return img

    def get_img_path(self, timestamp, exp_time=160, tolerance_timestamp=None):
        """
        Return the path to an image which complies with the requested timestamp and exposure time.

        :param timestamp: Timezone-aware datetime, requested timestamp
        :param exp_time: [microseconds] requested exposure time
        :param tolerance_timestamp: Timedelta, acceptable temporal deviation between actual and requested timestamp of
            the image
        :return: String, path to a suited image if available, empty string if unavailable
        """
        if tolerance_timestamp is None:
            tolerance_timestamp = self.tolerance_timestamp

        self.update_buffer(timestamp)
        buffer = self.img_paths_buffer['imgs']
        logging.debug(f"Buffer content before filtering: {buffer}")
        if not buffer.empty:
            buffer['timestamp_diff'] = buffer['timestamp'] - timestamp
            qualified = buffer[(buffer['exp_time'] >= exp_time * (1 - self.rel_exp_tol)) &
                               (buffer['exp_time'] <= exp_time * (1 + self.rel_exp_tol)) &
                               (buffer['timestamp_diff'] >= tolerance_timestamp[0]) &
                               (buffer['timestamp_diff'] <= tolerance_timestamp[1])]
            logging.debug(f"Qualified paths: {qualified}")
            if not qualified.empty:
                qualified.reset_index(inplace=True)
                chosen_path = qualified['path'][np.argmin(abs(qualified['timestamp_diff']))]
                logging.debug(f"Chosen path: {chosen_path}")
                return chosen_path

        return ''

    def update_buffer(self, timestamp):
        """
        Update a collection of available image files from the same day as a requested timestamp.

        TODO: waive key 'prev_date'

        :param timestamp: datetime, requested timestamp
        :return: Dict, collection of all available images from the same day as the requested timestamps with keys
            'prev_date' -- date to which collection corresponds, 'imgs' -- dataframe describing all images of the day
            with the columns 'timestamp', exposure time ('exp_time'), image path ('path')
        """
        present_day = timestamp.date()
        if not self.img_paths_buffer.get('prev_date') == present_day:
            paths_pattern = assemble_path(self.img_path_structure, self.name, present_day, set_subday_to_wildcard=True)
            paths = glob.glob(paths_pattern)
            if not len(paths):
                logging.warning(f'No images found for {timestamp:%Y-%m-%d}!')
                self.img_paths_buffer = {'prev_date': present_day, 'imgs': pd.DataFrame(
                    columns=['timestamp', 'exp_time', 'path'])}  # Ensure buffer is empty
                return

            timestamps = []
            exp_times = []
            paths_valid = []
            filename_pattern = Path(self.img_path_structure).name
            filename_re = fstring_to_re(filename_pattern)
            for path in paths:
                info = re.search(filename_re, path)
                if info is None:
                    logging.warning(f'Ignoring image file which does not meet naming convention: {path}.')
                    continue
                datetime_pattern = re.search(r'{timestamp:(.*?)}', filename_pattern).groups()[0]
                timestamps.append(
                    self.img_timezone.localize(datetime.strptime(info.group('timestamp'), datetime_pattern))
                        .astimezone(pytz.timezone('UTC')))
                if 'exposure_time' in info.groupdict().keys():
                    exp_time = info.group('exposure_time')
                else:
                    # TODO consider reading from image meta data at this point
                    exp_time = 0
                exp_times.append(int(exp_time))
                paths_valid.append(path)

            img_infos_day = pd.DataFrame({'timestamp': timestamps, 'exp_time': exp_times, 'path': paths_valid})

            before_setup = img_infos_day.timestamp < self.start_recording
            if np.any(before_setup):
                first_reject, last_reject = img_infos_day.loc[before_setup, 'timestamp'].iloc[[0, -1]]
                logging.warning(f'Timestamps ({first_reject}...{last_reject}) are before the start date of recording '
                                f'({self.start_recording}) for the current camera configuration. Ignoring images.')

            after_dismantle = img_infos_day.timestamp > self.end_recording
            if np.any(after_dismantle):
                first_reject, last_reject = img_infos_day.loc[after_dismantle, 'timestamp'].iloc[[0, -1]]
                logging.warning(f'Timestamps ({first_reject}...{last_reject}) are after the end date of recording '
                                f'({self.start_recording}) for the current camera configuration. Ignoring images.')

            img_infos_day = img_infos_day.loc[~(before_setup | after_dismantle)]

            if not len(img_infos_day.timestamp):
                logging.warning(f'Not any images with matching exposure time for {timestamp:%Y-%m-%d}.')

            self.img_paths_buffer = {'prev_date': present_day, 'imgs': img_infos_day}


class RadiometricImager(AllSkyImager):
    def __init__(self, camera_data, camera_id=None, image_path=None, tfms=None):
        """
        Initializes a RadiometricImager instance.

        :param camera_data: A dictionary containing the camera data.
        :type camera_data: dict
        :param camera_id: The ID of the camera.
        :type camera_id: str, optional
        :param image_path: The path to the image.
        :type image_path: str, optional
        :param tfms: A dictionary of image transformations.
        :type tfms: dict, optional
        """
        super().__init__(camera_data, camera_id, image_path, tfms)

        self.color_temperature = camera_data['exposure_settings']['color_temperature']

        self.weighting_luminosity = camera_data['radiometric_model']['weighting_luminosity']
        self.satVal = camera_data['radiometric_model']['saturation_val']
        self.base_sensitivity = camera_data['radiometric_model']['base_sensitivty']
        self.rel_overest_with_DNI = camera_data['radiometric_model']['rel_overest_with_DNI']
        self.satur_cor = camera_data['radiometric_model']['saturation_corr']
        self.beta_planck = camera_data['radiometric_model']['beta_planck']

    def invert_gamma_corr(self, img):
        """
        This function so far only implements the trivial case when the gamma correction is inactive!

        :param img: Gamma-corrected image
        :return: Image
        """

        return img


class GenericImager(RadiometricImager):
    """
    A generic version of RadiometricImager which can be used for cameras with unknown properties.
    """
    def __init__(self, camera_data, camera_id=None, image_path=None):
        """
        Initializes a GenericImager instance.

        :param camera_data: A dictionary containing the camera data.
        :type camera_data: dict
        :param camera_id: The ID of the camera.
        :type camera_id: str, optional
        :param image_path: The path to the image.
        :type image_path: str, optional
        """
        super().__init__(camera_data, camera_id=camera_id, image_path=image_path)

    def get_meta(self, _):
        """
        Return generic image meta data.

        :return: Dict of image meta data
        """
        return {'illuminance': np.nan}


def load_camera_data(camera_data_file=None, camera_data_dir=None, camera_name=None, timestamp=None):
    """Load data corresponding to available all-sky imagers.

    :param camera_data_file: yaml file path containing camera data.
    :param camera_data_dir: directory where camera data yaml files are located.
    :param camera_name: specifies a camera by name to filter data from (multiple ids possible for single name).
    :param timestamp: If not None, timestamp based on which the camera_id will be determined
    :return: camera data as dictionary.
    """
    camera_data_schema = yaml.safe_load(CAMERA_DATA_SCHEMA.open('r'))

    camera_data = {}
    camera_id = None
    if camera_data_file is None:
        camera_data_dir = ifnone(camera_data_dir, Path(__file__).parent / 'camera_data')
        for camera_data_file in Path(camera_data_dir).glob('*.yaml'):
            camera_data[camera_data_file.name] = yaml.safe_load(camera_data_file.open('r'))
            camera_data[camera_data_file.name]['_basedir'] = str(camera_data_dir)
    else:
        camera_data_file = Path(camera_data_file)
        camera_data_dir = camera_data_file.parent
        camera_id = camera_data_file.name
        with open(Path(camera_data_file), 'r') as file:
            camera_data[camera_id] = yaml.safe_load(file)
            camera_data[camera_id]['_basedir'] = str(camera_data_dir)

    cam_ids = list()
    for cam_id, cam_dict in camera_data.items():
        try:
            jsonschema.validate(cam_dict, camera_data_schema)
        except jsonschema.ValidationError as e:
            logging.warning("Camera data %s is not valid: %s", cam_id, e)
            continue

        if camera_name is not None and cam_dict['camera_name'] != camera_name:
            continue
        cam_ids.append(cam_id)
        mounted = cam_dict['mounted']
        demounted = cam_dict['demounted']
        assert type(mounted) is datetime, f"Mounted timestamp could not be recognized as datetime for cam id " \
                                    f"{cam_id}. Use the following format: !!timestamp 'YYYY-mm-ddtHH:MM:SS+HH:MM'"
        assert type(demounted) is datetime, f"Mounted timestamp could not be recognized as datetime for cam id " \
                                      f"{cam_id}. Use the following format: !!timestamp 'YYYY-mm-ddtHH:MM:SS+HH:MM'"
        assert mounted.tzinfo is not None and mounted.tzinfo.utcoffset(mounted) is not None, \
            f"Mounted timestamp could not be recognized as timezone aware for cam id {cam_id}. Use the following " \
            f"format: !!timestamp 'YYYY-mm-ddtHH:MM:SS+HH:MM.'"
        assert demounted.tzinfo is not None and demounted.tzinfo.utcoffset(demounted) is not None, \
            f"Demounted timestamp could not be recognized as timezone aware for cam id {cam_id}. Use the following " \
            f"format: !!timestamp 'YYYY-mm-ddtHH:MM:SS+HH:MM.'"
        camera_data[cam_id]['mounted'] = mounted
        camera_data[cam_id]['demounted'] = demounted
        if timestamp is not None:
            if pd.to_datetime(mounted) < pd.to_datetime(timestamp) < pd.to_datetime(demounted):
                camera_id = cam_id
    assert len(camera_data) > 0, \
        f"No valid camera data schemes found in {camera_data_dir}. Check if yaml files with correct JSON Scheme exist."
    assert timestamp is None or (timestamp is not None and camera_name is not None and camera_id is not None), \
        f'No valid camera data scheme was found for camera {camera_name} and timestamp {timestamp}.'
    if camera_id is not None:
        camera_data = camera_data[camera_id]
    else:
        camera_data = {cam_id: camera_data[cam_id] for cam_id in cam_ids}
    return camera_data


def load_camera_mask(mask_file, struct_name='Mask', allow_failure=True):
    """Load a camera mask as numpy array from a mat file.

    :param mask_file: file path of camera mask.
    :param struct_name: name of matlab struct containing camera mask array.
    :return: camera mask as 2D numpy array.
    """
    try:
        camera_mask = scipy.io.loadmat(mask_file)[struct_name][0][0][0]
        return camera_mask
    except Exception as e:
        if allow_failure:
            logging.warning(f'Could not load camera mask from {mask_file}. '
                            f'Trying to apply the mask will result in an error.')
        else:
            raise e


def load_celestial_coordinate_masks(mask_file, struct_name='AngleMatrix', field_el='ELE', field_az='AZ'):
    """Load elevation and azimuth masks from a mat file.

    :param mask_file: file path of mat file.
    :param struct_name: name of matlab struct.
    :param field_el: variable name of elevation angles array.
    :param field_az: variable name of azimuth angles array.
    :return: tuple of elevation angle mask and azimuth angle mask.
    """
    elevation_mask = scipy.io.loadmat(mask_file)[struct_name][0][field_el][0]
    azimuth_mask = scipy.io.loadmat(mask_file)[struct_name][0][field_az][0]
    return elevation_mask, azimuth_mask


def get_camera_location(camera_name):
    """Get geolocation based on camera name from camera config."""
    camera_data = load_camera_data(camera_name=camera_name)
    _, first_entry = next(iter(camera_data.items()))
    location = {key: first_entry[key] for key in ('latitude', 'longitude', 'altitude')}
    return location

# Licensed under the Apache License: http://www.apache.org/licenses/LICENSE-2.0
# For details: https://github.com/DLR-SF/sky_imaging/blob/main/NOTICE.txt

"""
Tools for the geometric self calibration of all-sky imagers

The internal camera model assumed by the calibration is described by:
Scaramuzza, D., et al. (2006). A Toolbox for Easily Calibrating Omnidirectional Cameras. RSJ International Conference
on Intelligent Robots and Systems  Beijing, China.

The external camera orientation is defined according to:
Luhmann, T. (2000). Nahbereichsphotogrammetrie: Grundlagen, Methoden und Anwendungen. Heidelberg, Germany, Wichmann
Verlag.

The self-calibration functionality itself was introduced and described by
Niklas Blum, Paul Matteschk, Yann Fabel, Bijan Nouri, Roberto Roman, Luis F. Zarzalejo, Juan Carlos Antuna-Sanchez,
Stefan Wilbert "Geometric Calibration of All-Sky Cameras Using Sun and Moon Positions: A Comprehensive Analysis",
Solar Energy (in review).

When using this self-calibration in any publication or other work, make sure to reference the latter publication.
"""

import os
import logging
import numpy as np
import scipy.optimize as scio
import pandas as pd
from datetime import timedelta, datetime
import cv2
import matplotlib.pyplot as plt

from asi_core import asi_analysis
from asi_core.camera import AllSkyImager


q25_ss_means = (-653.3, 2.677e-4, 4.498e-07)
q25_ss_std = (5.4, 2.89e-05, 3.066e-08)
"""
statistics taken from Rafal Broda's evaluation of PyranoCam Q25 cameras start values and search range should be tested 
if a new camera type needs to be calibrated. For the same camera type, calibration parameters do not vary a lot.
The second polynomial coefficient (a_1 of a_0 + a_1*x + a_2*x^2 + a_3*x^3) is assumed to be zero which holds for 
hyperbolic and parabolic mirrors or fisheye cameras according to Scaramuzza (2006).
"""


class Calibration:
    """
    Provides the ASI self-calibration functionality.

    The functionality is:

    - Detecting celestial bodies i.e. orbs
    - Transforming spherical coordinates of orbs into pixel coordinates
    - Calculating residuals between expected and found orb positions in ASI image
    - Optimizing external and internal calibration
    - Looping over timestamps
    - Visualizing results

    :param processed_timestamps: One (datetime, timezone-aware) or preferably all processed timestamps as pd.Series
        or similar (subclass of pandas.core.base.IndexOpsMixin)
    :param ss_expected: Specifies the mean values and the ranges within which coefficients of ss are optimized.
    :param eor_expected: Specifies the mean values and the ranges within which Euler angles are optimized.
    :param ignore_outliers_above_percentile: Percentile of the deviation used in the calibration. Observations with
        deviations beyond this will be sorted out.
    :param min_rel_dist_mask_orb: Threshold for minimum required distance between orb center and mask relative to
        orb diameter
    :param save_orb_quality_indicators: If True, save geometric parameters with each orb observation which
        indicate the observation's quality
    """

    def __init__(self, processed_timestamps, ss_expected={'mean': q25_ss_means, 'std': q25_ss_std},
                 eor_expected={'mean': (0, np.pi, np.pi/2), 'min': (-np.pi/4, np.pi*3/4, -2*np.pi),
                               'max': (np.pi/4, np.pi*5/4, 4*np.pi-1e-5)}, ignore_outliers_above_percentile=99,
                 min_rel_dist_mask_orb=1.5, save_orb_quality_indicators=True):
        """
        Initializes the calibration class.

        :param processed_timestamps: One (datetime, timezone-aware) or preferably all processed timestamps as pd.Series
            or similar (subclass of pandas.core.base.IndexOpsMixin)
        :param ss_expected: Specifies the mean values and the ranges within which coefficients of ss are optimized.
        :param eor_expected: Specifies the mean values and the ranges within which Euler angles are optimized.
        :param ignore_outliers_above_percentile: Percentile of the deviation used in the calibration. Observations with
            deviations beyond this will be sorted out.
        :param min_rel_dist_mask_orb: Threshold for minimum required distance between orb center and mask relative to
            orb diameter
        :param save_orb_quality_indicators: If True, save geometric parameters with each orb observation which
            indicate the observation's quality
        """

        # Instantiate the Camera class
        self.camera = AllSkyImager.from_config_and_period(processed_timestamps)
        self.ocam = self.camera.ocam_model
        self.orb_observations = pd.DataFrame()

        self.ss_expected = ss_expected
        self.eor_expected = eor_expected
        self.ignore_outliers_above_percentile = ignore_outliers_above_percentile
        self.min_rel_dist_mask_orb = min_rel_dist_mask_orb
        self.save_orb_quality_indicators = save_orb_quality_indicators

        self.bounds_eor = ((self.eor_expected['min'][0], self.eor_expected['max'][0]),
                           (self.eor_expected['min'][1], self.eor_expected['max'][1]),
                           (self.eor_expected['min'][2], self.eor_expected['max'][2]))

        if 'min' not in self.ss_expected.keys():
            # coefficients depend on camera resolution
            scaling_cam_resolution = min(self.camera.ocam_model.height, self.camera.ocam_model.width)/1970
            self.ss_expected['min'] = tuple([(v[0]-60*v[1])*scaling_cam_resolution for v in zip(ss_expected['mean'],
                                                                                                ss_expected['std'])])
        if 'max' not in self.ss_expected.keys():
            # coefficients depend on camera resolution
            scaling_cam_resolution = min(self.camera.ocam_model.height, self.camera.ocam_model.width)/1970
            self.ss_expected['max'] = tuple([(v[0]+60*v[1])*scaling_cam_resolution for v in zip(ss_expected['mean'],
                                                                                                ss_expected['std'])])

        self.bounds_eor_ior = self.bounds_eor + ((self.ss_expected['max'][0]/self.ss_expected['mean'][0],
                                                  self.ss_expected['min'][0] / self.ss_expected['mean'][0]),
                                                 (self.ss_expected['min'][1]/self.ss_expected['mean'][1],
                                                  self.ss_expected['max'][1]/self.ss_expected['mean'][1]),
                                                 (self.ss_expected['min'][2]/self.ss_expected['mean'][2],
                                                  self.ss_expected['max'][2]/self.ss_expected['mean'][2]))

        self.bounds_eor_ior_center = self.bounds_eor_ior + ((0.95, 1.05), (0.95, 1.05))

        self.x_center_expected = self.camera.ocam_model.height/2
        self.y_center_expected = self.camera.ocam_model.width/2

    def optimize_eor(self, orb_observations, orientation_init):
        """
        Calls optimizer to minimize calc_residual_eor setting only the EOR's angles as free parameters

        :param orb_observations: dataframe of detected orbs with identified pixel coordinates and spherical coordinates
            from astronomy
        :param orientation_init: Initial guess of the Euler angles determining the ASI's external orientation
        :return: Results of the optimization see scio.minimize
        """
        self.orb_observations = orb_observations

        options = {'ftol': 1e-20, 'gtol': 1e-20}
        res = scio.minimize(self.calc_residual_eor, orientation_init, bounds=self.bounds_eor, options=options)
        return res

    def optimize_eor_ior(self, orb_observations, orientation_init=None, ocam_init=None, test_various_azimuths=True):
        """
        Calls optimizer to minimize calc_residual_eor_ior setting the EOR's angles and the coefficients of ss of the
        internal calibration as free parameters.

        :param orb_observations: dataframe of detected orbs with identified pixel coordinates and spherical coordinates
            from astronomy
        :param orientation_init: Initial guess of the Euler angles determining the ASI's external orientation
        :param ocam_init: Initial guess of camera's internal calibration in particular of the coefficients of ss
        :param test_various_azimuths: If True, loop over azimuth angles which represent 360° rotation of ASI around
            vertical axis
        :return: Results of the optimization see scio.minimize
        """
        self.orb_observations = orb_observations

        if orientation_init is None:
            orientation_init = list(self.eor_expected['mean'])
        elif isinstance(orientation_init, np.ndarray):
            orientation_init = orientation_init.tolist()

        if ocam_init is None:
            ocam_init = self.camera.ocam_model

        self.ocam = ocam_init
        self.ocam.ss[0] /= self.ss_expected['mean'][0]
        self.ocam.ss[2] /= self.ss_expected['mean'][1]
        self.ocam.ss[3] /= self.ss_expected['mean'][2]

        if test_various_azimuths:
            azimuth_inits = np.arange(0, 2*np.pi, np.pi/8)
        else:
            azimuth_inits = [orientation_init[2]]

        res_min = None
        for azimuth_init in azimuth_inits:
            orientation_init[2] = azimuth_init
            options = {'ftol': 1e-20, 'gtol': 1e-20}
            res = scio.minimize(self.calc_residual_eor_ior, orientation_init + self.ocam.ss[[0, 2, 3]].tolist(),
                                bounds=self.bounds_eor_ior, options=options)

            res.x[3] = res.x[3]*self.ss_expected['mean'][0]
            res.x[4] = res.x[4]*self.ss_expected['mean'][1]
            res.x[5] = res.x[5]*self.ss_expected['mean'][2]

            if res_min is None or res.fun < res_min.fun:
                res_min = res

        self.ocam.ss[0] *= self.ss_expected['mean'][0]
        self.ocam.ss[2] *= self.ss_expected['mean'][1]
        self.ocam.ss[3] *= self.ss_expected['mean'][2]

        return res_min

    def optimize_eor_ior_center(self, orb_observations, orientation_init=None, ocam_init=None):
        """
        WARNING THIS FUNCTION WAS NOT TESTED PROPERLY YET. DON'T USE IT.
        
        Calls optimizer to minimize calc_residual_eor_ior setting the EOR's angles and the coefficients of ss of the
        internal calibration and image center as free parameters.

        :param orb_observations: dataframe of detected orbs with identified pixel coordinates and spherical coordinates
            from astronomy
        :param orientation_init: Initial guess of the Euler angles determining the ASI's external orientation
        :param ocam_init: Initial guess of camera's internal calibration in particular of the coefficients of ss
        :return: Results of the optimization see scio.minimize
        """
        self.orb_observations = orb_observations

        if orientation_init is None:
            orientation_init = list(self.eor_expected['mean'])

        if ocam_init is None:
            ocam_init = self.camera.ocam_model

        self.ocam = ocam_init
        self.ocam.ss[0] /= self.ss_expected['mean'][0]
        self.ocam.ss[2] /= self.ss_expected['mean'][1]
        self.ocam.ss[3] /= self.ss_expected['mean'][2]
        self.ocam.x_center /= self.x_center_expected
        self.ocam.y_center /= self.y_center_expected

        res_min = None
        for azimuth_init in np.arange(0, 2*np.pi, np.pi/8):
            orientation_init[2] = azimuth_init
            options = {'ftol': 1e-20, 'gtol': 1e-20}
            res = scio.minimize(self.calc_residual_eor_ior_center, orientation_init + self.ocam.ss[[0, 2, 3]].tolist() +
                                [self.ocam.x_center, self.ocam.y_center], bounds=self.bounds_eor_ior_center,
                                options=options)

            res.x[3] = res.x[3]*self.ss_expected['mean'][0]
            res.x[4] = res.x[4]*self.ss_expected['mean'][1]
            res.x[5] = res.x[5]*self.ss_expected['mean'][2]
            res.x[6] = res.x[6]*self.x_center_expected
            res.x[7] = res.x[7]*self.y_center_expected

            if res_min is None or res.fun < res_min.fun:
                res_min = res

        return res_min

    def calc_residual_eor_ior_center(self, orientation_ior_center):
        """
        Calculates deviation metric of the difference between expected and found coordinates of celestial bodies.

        :param orientation_ior_center: Array of the three Euler angles and coefficients no. 0, 2, 3 of ss and x_center,
            y_center
        :return: Deviation used as target for the optimization
        """
        orientation = orientation_ior_center[:3]
        self.ocam.ss[0] = orientation_ior_center[3]*self.ss_expected['mean'][0]
        self.ocam.ss[2] = orientation_ior_center[4]*self.ss_expected['mean'][1]
        self.ocam.ss[3] = orientation_ior_center[5]*self.ss_expected['mean'][2]
        self.ocam.x_center = orientation_ior_center[6]*self.x_center_expected
        self.ocam.y_center = orientation_ior_center[7]*self.y_center_expected

        return self.angles_pixels_to_vector_deviation(self.orb_observations, self.ocam, orientation,
                                                      self.ignore_outliers_above_percentile)

    def calc_residual_eor_ior(self, orientation_ior):
        """
        Calculates deviation metric of the difference between expected and found coordinates of celestial bodies.

        :param orientation_ior: Array of the three Euler angles and coefficients no. 0, 2, 3 of ss
        :return: Deviation used as target for the optimization
        """
        orientation = orientation_ior[:3]
        self.ocam.ss[0] = orientation_ior[3]*self.ss_expected['mean'][0]
        self.ocam.ss[2] = orientation_ior[4]*self.ss_expected['mean'][1]
        self.ocam.ss[3] = orientation_ior[5]*self.ss_expected['mean'][2]

        dev = self.angles_pixels_to_vector_deviation(self.orb_observations, self.ocam, orientation,
                                                     self.ignore_outliers_above_percentile)

        self.ocam.ss[0] = orientation_ior[3]/self.ss_expected['mean'][0]
        self.ocam.ss[2] = orientation_ior[4]/self.ss_expected['mean'][1]
        self.ocam.ss[3] = orientation_ior[5]/self.ss_expected['mean'][2]
        return dev

    def calc_residual_eor(self, orientation):
        """
        Calculates deviation metric of the difference between expected and found coordinates of celestial bodies.

        :param orientation: Array of the three Euler angles
        :return: Deviation used as target for the optimization
        """

        return self.angles_pixels_to_vector_deviation(self.orb_observations, self.ocam, orientation,
                                                      self.ignore_outliers_above_percentile)

    @staticmethod
    def angles_to_pixels(orb_observations, ocam, eor):
        """
        Transforms angle coordinates of a celestial body seen by a camera into pixel coordinates on the chip

        :param orb_observations: dataframe with columns azimuth and elevation indicating the positions of one or
            more celestial bodies observed at one or more timestamps
        :param ocam: Ocam object describing the ASI's camera model and internal calibration
        :param eor: (array) External orientation of the ASI indicated by the 3 Euler angles
        :return: Pixel coordinates of the celestial body in the camera image
        """
        # 1. set arbitrary radius
        r = np.ones(np.shape(orb_observations['azimuth']))
        # 2. transform azimuth and elevation angle (provided as dataframe over all timestamps) into image coordinates
        x, y, z = asi_analysis.sph2cart(np.deg2rad(orb_observations['azimuth']),
                                        np.deg2rad(orb_observations['elevation']), r)
        # 3. Apply rotation accounting for EOR and distortion of the camera
        orb_coos_pixels = ocam.world2cam_eor(eor, np.asarray([x, y, z]).T, use_ss=True)

        orb_observations['expected_x'] = orb_coos_pixels[:, 0]
        orb_observations['expected_y'] = orb_coos_pixels[:, 1]
        return orb_observations

    @staticmethod
    def angles_pixels_to_vector_deviation(orb_observations, ocam, eor, ignore_outliers_above_percentile=None,
                                          compute_found_angles=False):
        """
        Calculates root mean squared deviation between detected and expected orb coordinates in cartesian coordinates.

        Transforms angle coordinates of a celestial body seen by a camera and pixel coordinates of detected body to
        cartesian coordinates, calculates area spun up by both vectors and calculates root mean square of all areas
        (i.e. deviations).

        :param orb_observations: dataframe with columns azimuth and elevation indicating the positions of one or
            more celestial bodies observed at one or more timestamps
        :param ocam: Ocam object describing the ASI's camera model and internal calibration
        :param eor: (array) External orientation of the ASI indicated by the 3 Euler angles
        :param ignore_outliers_above_percentile: If number between 0 and 100 is provided observations with deviation
            above corresponding percentile are excluded from calculation of deviation/ optimization
        :param compute_found_angles: If True, return azimuth and elevation angles of found orb positions
        :return: Root mean square deviation from all orb observations
        """
        # 1. set arbitrary radius
        r = np.ones(np.shape(orb_observations['azimuth']))
        # 2. transform azimuth and elevation angle (provided as dataframe over all timestamps) into image coordinates
        expected_3d_x, expected_3d_y, expected_3d_z = asi_analysis.sph2cart(
            np.deg2rad(orb_observations['azimuth']), np.deg2rad(orb_observations['elevation']), r)

        expected_3d_coos = np.asarray([expected_3d_x, expected_3d_y, expected_3d_z]).T

        # 3. Apply rotation accounting for EOR and distortion of the camera
        found_3d_coos = ocam.cam2world_eor(eor, np.asarray([orb_observations['found_x'],
                                                            orb_observations['found_y']]).T)

        deviation = np.linalg.norm(np.cross(found_3d_coos, expected_3d_coos), axis=1)

        if ignore_outliers_above_percentile is not None and 0 < ignore_outliers_above_percentile < 100:
            valid_rows = deviation <= np.percentile(deviation, ignore_outliers_above_percentile)
        else:
            valid_rows = np.ones(np.shape(deviation), dtype=bool)
        logging.debug(f'Valid rows: {np.sum(valid_rows)}, rejected rows: {np.sum(np.logical_not(valid_rows))}, '
                      f'deviation: {np.sqrt(np.mean(deviation[valid_rows] ** 2))}')

        rmse_crossproduct = np.sqrt(np.mean(deviation[valid_rows] ** 2))

        if not compute_found_angles:
            return rmse_crossproduct

        orb_observations['found_azimuth'], orb_observations['found_elevation'], _ = asi_analysis.cart2sph(
            found_3d_coos[:, 0], found_3d_coos[:, 1], found_3d_coos[:, 2])

        orb_observations['found_azimuth'] = np.rad2deg(orb_observations['found_azimuth'])
        orb_observations['found_elevation'] = np.rad2deg(orb_observations['found_elevation'])

        return rmse_crossproduct, orb_observations

    @staticmethod
    def get_deviation(orb_observations, ignore_outliers_above_percentile=99):
        """
        Calculates deviation metric between found and astronomically expected orb positions.

        :param orb_observations: Dataframe with expected 'expected_x, *_y' and found orb positions 'found_x, *_y' as
            columns
        :param ignore_outliers_above_percentile: (in percent) If not None and between 0 and 100, datapoints with a
            deviation above this percentile (e.g. above 99%) are ignored.
        :return: Deviation metric
        """
        # solver can better work with large deviations than with "nan" deviations, replace accordingly
        orb_observations.loc[np.isnan(orb_observations['expected_x']), 'expected_x'] = 1e4
        orb_observations.loc[np.isnan(orb_observations['expected_y']), 'expected_y'] = 1e4

        deviation = np.sqrt((orb_observations['found_x'] - orb_observations['expected_x']) ** 2 +
                            (orb_observations['found_y'] - orb_observations['expected_y']) ** 2)

        if ignore_outliers_above_percentile is not None and 0 < ignore_outliers_above_percentile < 100:
            valid_rows = deviation <= np.percentile(deviation, ignore_outliers_above_percentile)
        else:
            valid_rows = np.ones(np.shape(deviation), dtype=bool)

        return np.sqrt(np.mean(deviation[valid_rows]**2))

    def find_orb_positions(self, detector):
        """
        Detects orb positions for multiple timestamps.

        :param detector: CelestialBodyDetector instance
        :return: dataframe of orb observations with columns timestamp, found pixel position of the orb and expected
        elevation and azimuth angle of the orb in degree.
        """
        processed_count = 0
        detected_count = 0
        celestial_data = None
        outage_dates = set()

        # used later to detect if orb is too close to mask
        mask_edge = np.where(cv2.Canny(self.camera.camera_mask, 0, 1))

        for timestamp in detector.timestamps:
            day = timestamp.date()
            if day in outage_dates:
                continue

            logging.info(f"Detecting orb for timestamp: {timestamp}")

            img_meta = self.camera.get_img_and_meta(timestamp, exp_time=detector.exp_time)

            if not len(self.camera.img_paths_buffer['imgs']):
                logging.info(f"Skipping date {day} as no images were found.")
                outage_dates.add(day)
                continue
            logging.debug(f"Image metadata: {img_meta}")

            if not len(img_meta['img']):
                logging.info(f"No valid image for timestamp: {timestamp}")
                continue
            processed_count += 1
            img = img_meta['img']

            if len(self.camera.camera_mask):
                img = self.camera.transform(img)

            orb_observation_props = detector.detect_celestial_body(img)

            if orb_observation_props is None:
                logging.debug(f"No celestial body detected.")
                continue
            logging.info(f"Celestial body detected at: "
                         f"{orb_observation_props['found_x'], orb_observation_props['found_y']}")

            dist_mask = np.min(np.sqrt((mask_edge[0] - orb_observation_props['found_y'][0])**2 +
                                       (mask_edge[1] - orb_observation_props['found_x'][0])**2))

            if dist_mask < self.min_rel_dist_mask_orb*np.sqrt(orb_observation_props['area']/np.pi):
                logging.info(f"Orb rejected. Center too close to mask. (Distance {dist_mask:.1f} pixels, "
                              f"orb area {orb_observation_props['area']:.0f} pixels)")
                continue

            detected_count += 1

            if not self.save_orb_quality_indicators:
                for k in ['area', 'aspect_ratio', 'circularity']:
                    orb_observation_props.pop(k, None)

            new_data = pd.DataFrame({'timestamp': [timestamp]} | orb_observation_props)

            new_data['azimuth'], new_data['elevation'] = detector.calculate_azimuth_elevation(timestamp)

            celestial_data = pd.concat([celestial_data, new_data], ignore_index=True)

            logging.info(f'Processed: {timestamp}')

        if not processed_count and all([v in outage_dates for v in
                                        pd.unique(pd.DatetimeIndex(detector.timestamps).date)]):
            err_msg = (f"Not any images found for requested timestamps and exposure times. Please check your"
                       f" configuration: \n"
                       f"- Are exposure times set correctly?\n"
                       f"- Is img_path_structure set correctly?\n")
            logging.error(err_msg)
            raise Exception(err_msg)
        celestial_data = celestial_data[celestial_data['elevation'] > detector.min_orb_ele]

        logging.info(f'Total images processed: {processed_count}')
        logging.info(f'Total celestial bodies detected: {detected_count}')
        return celestial_data

    def get_all_orb_positions(self, orb_detectors, output_file=f'orb_observations.csv', reset_temp=True):
        """
        Loads or detects orb positions of multiple types, especially sun and moon.

        :param orb_detectors: list of CelestialBodyDetector instances
        :param output_file: path to save dataframe of orb observations to
        :param reset_temp: reprocess orb positions and do not load previously saved dataframes
        :return: dataframe of orb observations with columns timestamp, found pixel position of the orb and expected
        elevation and azimuth angle of the orb in degree.
        """
        processed_count = 0
        celestial_data = None

        if reset_temp and os.path.isfile(output_file):
            os.remove(output_file)

        if not os.path.isfile(output_file):
            for detector in orb_detectors:
                new_data = self.find_orb_positions(detector)
                processed_count += new_data.shape[0]
                if celestial_data is None:
                    celestial_data = new_data
                    continue
                celestial_data = pd.concat([celestial_data, new_data], ignore_index=True)
            celestial_data.to_csv(output_file)
        else:
            celestial_data = pd.read_csv(output_file, converters={'timestamp': pd.to_datetime})

        logging.info(f'Total images processed: {processed_count}')
        logging.info(f'Total celestial bodies detected: {np.shape(celestial_data)[0]}')

        return celestial_data

    def compute_and_save_azimuth_elevation(self, ocam_model, min_ele_evaluated, external_orientation, save_npy=False):
        """
        Computes the azimuth and elevation matrices, and optionally saves them as .npy files.

        :param ocam_model: Updated ocam model.
        :param min_ele_evaluated: Updated minimum elevation evaluated.
        :param external_orientation: Updated external orientation.
        :param save_npy: Boolean flag to save the matrices as .npy files.
        """
        azimuth_matrix, elevation_matrix = self.camera.get_azimuth_elevation(
            ocam_model, min_ele_evaluated, external_orientation)

        if save_npy:
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            azimuth_file = f'azimuth_matrix_{self.camera.name}_{timestamp}.npy'
            elevation_file = f'elevation_matrix_{self.camera.name}_{timestamp}.npy'

            np.save(azimuth_file, azimuth_matrix)
            np.save(elevation_file, elevation_matrix)

            logging.info(f'Saved azimuth matrix to {azimuth_file}')
            logging.info(f'Saved elevation matrix to {elevation_file}')

        return azimuth_matrix, elevation_matrix


def plot_orb_positions(orb_observations_path, background_image_path,
                       output_path_figure='found_and_expected_orb_positions_after_calibration.png',
                       exp_area_diameter=None, center_x=None, center_y=None,
                       outliers_above_percentile_ignored_in_calib=99):
    """
    Creates and saves a plot which compares expected and found orb pixel positions.

    :param orb_observations_path: Path to load dataframe of expected and found orb positions from
    :param background_image_path: Path to background image laid under plot
    :param output_path_figure: Path to save the figure to
    :param exp_area_diameter: Diameter of the exposed area in the fisheye image
    :param center_x: x-coordinate of the center of the ASI image
    :param center_y: y-coordinate of the center of the ASI image
    :param outliers_above_percentile_ignored_in_calib: Percentile of outliers to be filtered to recalculate deviation
        metric received in calibration
    """

    # Load the sun data
    sun_data = pd.read_csv(orb_observations_path, converters={'timestamp': pd.to_datetime})

    # Load the background image
    background_image = cv2.imread(background_image_path)
    background_image = cv2.cvtColor(background_image, cv2.COLOR_BGR2RGB)

    # Create a figure with the desired DPI
    plt.figure()

    # Reverse the y-axis to place the coordinate origin on the top left
    plt.gca().invert_yaxis()

    # Plot the background image
    plt.imshow(background_image)

    # Plot the center point as a thin circle
    if center_x is None or center_y is None:
        center_x = center_x
        center_y = center_y
    if exp_area_diameter is not None:
        circle = plt.Circle((center_x, center_y), exp_area_diameter / 2, color='white', fill=False, linewidth=0.5)
        plt.gca().add_artist(circle)

    # Plot the detected and expected sun positions
    plt.scatter(sun_data['found_x'], sun_data['found_y'], s=2, c='red', marker='o', label='Detected orb positions')
    plt.scatter(sun_data['expected_x'], sun_data['expected_y'], s=2, c='blue', marker='x',
                label='Expected orb positions')
    plt.scatter(center_x, center_y, s=1, c='white', marker='o')

    # Calculate the root mean squared error (RMSE) between detected and expected sun positions
    rmse = Calibration.get_deviation(sun_data, ignore_outliers_above_percentile=100)
    rmse_cleaned = Calibration.get_deviation(sun_data, ignore_outliers_above_percentile=
                                             outliers_above_percentile_ignored_in_calib)

    # Add a textbox with the average RMSE
    plt.text(0.98, 0.02, f'RMSE: {rmse:.2f}\nRMSE excluding outliers ({100-outliers_above_percentile_ignored_in_calib}%): {rmse_cleaned:.2f}',
             transform=plt.gca().transAxes, ha='right', va='bottom', fontsize=8, color='white')

    # Add labels and a legend
    plt.xlabel('x-coordinate (pixels)')
    plt.ylabel('y-coordinate (pixels)')
    plt.title('Detected and expected orb positions')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0))
    plt.savefig(output_path_figure, bbox_inches='tight')


def get_background_img_and_plot(camera, orb_obs_file, exp_time, image_file_name,
                                outliers_above_percentile_ignored_in_calib):
    """
    Find background image and plot all found and expected orb positions over this image.

    :param camera: AllSkyImager instance
    :param orb_obs_file: csv file containing dataframe of found and expected orb positions
    :param exp_time: [microseconds], exposure time of the background image
    :param image_file_name: Identifier to be used as first part of the saved figure's name
    :param outliers_above_percentile_ignored_in_calib: Percentile of outliers to be filtered to recalculate deviation
        metric received in calibration
    """
    orb_data = pd.read_csv(orb_obs_file, converters={'timestamp': pd.to_datetime})

    background_image_path = []
    for img_timestamp in pd.date_range(orb_data.timestamp[0].astimezone(camera.img_timezone),
                                       orb_data.timestamp.iloc[-1].astimezone(camera.img_timezone),
                                       freq=timedelta(days=1)):
        background_image_path = camera.get_img_path(img_timestamp, exp_time=exp_time,
                                                    tolerance_timestamp=[timedelta(days=-1), timedelta(days=1)])
        if len(background_image_path):
            break

    if not len(background_image_path):
        raise Exception('No valid background image found to visualize the orbs used in calibration.')

    plot_orb_positions(orb_obs_file, background_image_path, output_path_figure=image_file_name,
                       outliers_above_percentile_ignored_in_calib=outliers_above_percentile_ignored_in_calib)


class CenterDetector:
    """
    Provides functionality to detect and visualize the center of the exposed area of an ASI image.
    """
    max_intensity = 255
    max_threshold_value = 250
    gaussian_blur_kernel = (5, 5)

    def find_best_threshold(self, img, min_contour_area_ratio=0.1):
        """
        Finds the boundary of a fisheye image's exposed area by maximizing the circularity of the detected contours.

        :param img: Image based on which center is detected.
        :param min_contour_area_ratio: (float) Minimum contour area ratio to consider for circle detection.

        :return: tuple of the best threshold value (int), and a list of pixel coordinates indicating the best contour
            found.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, self.gaussian_blur_kernel, 0)

        min_contour_area = min_contour_area_ratio * img.shape[0] * img.shape[1]

        best_circularity = -1
        best_threshold = None
        best_contour = None

        # Test thresholds from 1 to 250 in increments of 1
        for threshold_value in range(1, self.max_threshold_value, 1):
            _, threshold = cv2.threshold(blur, threshold_value, self.max_intensity, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                if cv2.contourArea(cnt) >= min_contour_area and asi_analysis.is_circle_contour(cnt)[0]:
                    circularity = 4 * np.pi * cv2.contourArea(cnt) / (cv2.arcLength(cnt, True) ** 2)

                    if circularity > best_circularity:
                        best_circularity = circularity
                        best_threshold = threshold_value
                        best_contour = cnt

        return best_threshold, best_contour

    def find_fisheye_circle_center(self, img, min_contour_area_ratio=0.1, save_visualization=True):
        """
        Finds the center and radius of a fisheye image's exposed area.

        :param img: image used to detect center
        :param min_contour_area_ratio: (float) Minimum contour area ratio to consider for circle detection.
        :param save_visualization: If true, a visualization of the found image center is created and saved.
        :return: Center coordinates of the fisheye circle (tuple), exposed radius and diameter.
        """
        best_threshold, best_contour = self.find_best_threshold(img, min_contour_area_ratio)

        if best_contour is not None:
            moments = cv2.moments(best_contour)
            center_y = moments["m10"] / moments["m00"]
            center_x = moments["m01"] / moments["m00"]

            _, radius = cv2.minEnclosingCircle(best_contour)
            diameter = 2 * radius

            # Comparison of the image center with the fisheye center
            img_height, img_width, _ = img.shape
            image_center_y = img_width / 2
            image_center_x = img_height / 2
            center_difference_x = abs(center_y - image_center_y)
            center_difference_y = abs(center_x - image_center_x)

            logging.info(f"Fisheye objective's center: ({center_x}, {center_y})")
            logging.info(f"Offset from image center (x, y): ({center_difference_x}, {center_difference_y})")
            logging.info(f"Exposed area's radius: {radius}")

            if save_visualization:
                self.visualize_center(img, best_contour, center_x, center_y)

            return (center_x, center_y), radius, diameter
        else:
            logging.info("No suitable contour found")
            return None

    def visualize_center(self, img, best_contour, center_x, center_y, output_image_path='fisheye_rim.jpg'):
        """
        Visualizes the position of the fisheye objective's center in the image.

        :param img: Background image
        :param best_contour: Contour indicating the most reasonable boundary of the fisheye's exposed area
        :param center_x: x-coordinate of the ASI image's center
        :param center_y: y-coordinate of the ASI image's center
        :param output_image_path: Path to save the figure to
        """

        cv2.drawContours(img, [best_contour], -1, (0, self.max_intensity, 0), 3)
        cv2.circle(img, (int(center_x), int(center_y)), 2, (0, 0, self.max_intensity), -1)

        # Customize the preview window and save the image as a JPEG
        preview_window_name = "Fisheye Circle"
        cv2.namedWindow(preview_window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(preview_window_name, 600, 600)
        cv2.imshow(preview_window_name, img)
        cv2.imwrite(output_image_path, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def find_center_timerange(self, camera, start, end, sampling_time=timedelta(hours=2), exp_time=160):
        """
        Loops over range of timestamps, detecting the average image center in all respective images

        :param camera: Camera object to load images with
        :param start: First timestamp processed
        :param end: Last timestamp processed
        :param sampling_time: Time difference between evaluated images
        :param exp_time: Required exposure time of considered images
        :return: Mean values of the image center's x and y coordinates and radius of the exposed area
        """
        processed_count = 0
        detected_count = 0

        center_x = []
        center_y = []
        radius = []

        for timestamp in pd.date_range(start, end, freq=sampling_time):
            img_meta = camera.get_img_and_meta(timestamp, exp_time=exp_time)
            if not len(img_meta['img']):
                # skip timestamp if no valid image available
                continue
            processed_count += 1

            results_ts = self.find_fisheye_circle_center(img_meta['img'], save_visualization=False)

            if results_ts is None:
                continue
            detected_count += 1

            center_x.append(results_ts[0][0])
            center_y.append(results_ts[0][1])
            radius.append(results_ts[1])

        logging.info(f"Fisheye center x (mean, std): ({np.mean(center_x)}, {np.std(center_x)})")
        logging.info(f"Fisheye center y (mean, std): ({np.mean(center_y)}, {np.std(center_y)})")
        logging.info(f"Fisheye radius (mean, std): ({np.mean(radius)}, {np.std(radius)})")

        return np.mean(center_x), np.mean(center_y), np.mean(radius)


def find_center_via_cam_model(calibrator, orb_data, x_samples=6, y_samples=None, max_rel_center_dev=0.25,
                              number_iterations=11, slow_but_roubst=False):
    """
    Determines all parameters of the camera model iteratively refining the lens center image coordinates

    Reuses the optimization algorithm of the lens center of Scaramuzza 2006 however calling our calibration procedure

    :param calibrator: calibrator instance
    :param orb_data: Dataframe of image- and astronomy-based orb positions
    :param x_samples: Number of sample points in each iterations mesh grid of test points in x-direction
    :param y_samples: Number of sample points in each iterations mesh grid of test points in x-direction
    :param max_rel_center_dev: Maximum expected deviation of the lens center's image coordinates from the image center
        as fraction of the image side height and width
    :param number_iterations: Number of refinement iterations for center detection
    :param slow_but_roubst: If True, don't use options to speed up process. Make sure that process converges.
    :return: calibrator instance with refined lens center position (x_center, y_center)
    """

    if len(orb_data) == 0:
        logging.info(r'No orb data available for center detection.')
        return

    logging.info(r'Computing center coordinates.')

    if y_samples is None:
        y_samples = x_samples

    dx_reg = calibrator.ocam.height * max_rel_center_dev
    dy_reg = calibrator.ocam.width * max_rel_center_dev

    # do first optimization to get rough start values of EOR and IOR assuming lens center to coincide with image center
    calibrator.ocam.x_center = calibrator.ocam.height/2
    calibrator.ocam.y_center = calibrator.ocam.width/2
    orientation_optimized = calibrator.optimize_eor_ior(orb_data, calibrator.camera.external_orientation,
                                                        calibrator.ocam)
    first_guess_ss = orientation_optimized.x[3:]
    first_guess_eor = orientation_optimized.x[:3]
    calibrator.ocam.ss[[0, 2, 3]] = first_guess_ss
    calibrator.camera.external_orientation = first_guess_eor

    logging.info('Starting iteration, stepwise refining grid of potential lens centers.')
    for glc in range(number_iterations):
        x_reg_start = calibrator.ocam.x_center - dx_reg
        x_reg_stop = calibrator.ocam.x_center + dx_reg
        y_reg_start = calibrator.ocam.y_center - dy_reg
        y_reg_stop = calibrator.ocam.y_center + dy_reg

        [x_reg, y_reg] = np.meshgrid(
            np.linspace(x_reg_start, x_reg_stop, x_samples),
            np.linspace(y_reg_start, y_reg_stop, y_samples))
        loss_grid = np.inf * np.ones(np.shape(x_reg))

        logging.info(rf'Testing a grid of center coordinates with '
                     rf'x-coordinates: ({x_reg[0, :]}) and '
                     rf'y-coordinates: ({y_reg[:, 0]})')

        for ic in range(np.shape(x_reg)[0]):
            for jc in range(np.shape(x_reg)[1]):
                calibrator.ocam.x_center = x_reg[ic, jc]
                calibrator.ocam.y_center = y_reg[ic, jc]
                calibrator.ocam.ss[[0, 2, 3]] = first_guess_ss
                calibrator.camera.external_orientation = first_guess_eor

                orientation_optimized = calibrator.optimize_eor_ior(orb_data, calibrator.camera.external_orientation,
                                                                    calibrator.ocam,
                                                                    test_various_azimuths=slow_but_roubst)
                logging.debug(f'ic: {ic}, jc: {jc}, loss: {orientation_optimized.fun}')
                loss_grid[ic, jc] = orientation_optimized.fun

        min_loss = np.min(loss_grid)
        idx_min_loss = np.where(loss_grid == min_loss)
        calibrator.ocam.x_center = x_reg[idx_min_loss]
        calibrator.ocam.y_center = y_reg[idx_min_loss]

        # refine mesh
        dx_reg = abs((x_reg_stop - x_reg_start) / (x_samples-1))
        dy_reg = abs((y_reg_stop - y_reg_start) / (y_samples-1))

        logging.info(rf'Refinement iteration no. {glc+1} completed. Minimum loss is {loss_grid[idx_min_loss]}. '
                     f'Center is detected in ({calibrator.ocam.x_center}, {calibrator.ocam.y_center})')

    logging.info(r'\n')
    orientation_optimized = calibrator.optimize_eor_ior(orb_data, calibrator.camera.external_orientation,
                                                        calibrator.ocam)
    calibrator.ocam.ss[[0, 2, 3]] = orientation_optimized.x[3:]
    calibrator.camera.external_orientation = orientation_optimized.x[:3]

    logging.info(f'Iterative optimization of lens center coordinates and remaining camera model completed.')
    return calibrator, min_loss

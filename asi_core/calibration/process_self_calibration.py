# Licensed under the Apache License: http://www.apache.org/licenses/LICENSE-2.0
# For details: https://github.com/DLR-SF/sky_imaging/blob/main/NOTICE.txt

"""
Tools to process and validate the all-sky imager geometric self calibration
"""

import glob
import os
import logging
import shutil
from datetime import datetime, timedelta
import pytz
import numpy as np
from dateutil.tz import tzlocal
import scipy.io as scio
import pandas as pd
import re
from pathlib import Path
import yaml

from asi_core import config
from asi_core.calibration import self_calibration
from asi_core.calibration.celestial_bodies import Moon, Sun
from asi_core.basics import fstring_to_re


def calibrate_from_images(last_timestamp_calibration):
    """
    Main function for ASI self calibration (when used as a batch processing tool).

    Instantiate the self calibration process and determine the ASI's calibration (IOR + EOR).

    :param last_timestamp_calibration: Timestamp indicating the end of the calibration period
    """
    cfg_calib = config.get('Calibration')

    calib = self_calibration.Calibration(last_timestamp_calibration, ss_expected=cfg_calib['ss_statistics'],
                                         ignore_outliers_above_percentile=cfg_calib['ignore_outliers_above_percentile'],
                                         min_rel_dist_mask_orb=cfg_calib['min_rel_dist_mask_orb'],
                                         save_orb_quality_indicators=cfg_calib['save_orb_quality_indicators'])

    # Check for x_center and y_center in the configuration
    xy_center_in_config = hasattr(calib.camera.ocam_model, 'x_center') and hasattr(calib.camera.ocam_model, 'y_center')
    diameter_in_config = hasattr(calib.camera.ocam_model, 'diameter') and calib.camera.ocam_model.diameter is not None

    if xy_center_in_config:
        logging.info(f'Using x_center and y_center from configuration: ({calib.ocam.x_center}, {calib.ocam.y_center})')
    else:
        if cfg_calib['target_calibration'] in ['optimize_eor_ior', 'optimize_eor_ior_center']:
            first_timestamp = last_timestamp_calibration - timedelta(days=cfg_calib['center_detection']['number_days'])
            first_timestamp = np.max([first_timestamp, calib.camera.start_recording])
            detector = self_calibration.CenterDetector()
            calib.ocam.x_center, calib.ocam.y_center, radius = detector.find_center_timerange(
                calib.camera, first_timestamp, last_timestamp_calibration,
                sampling_time=timedelta(hours=cfg_calib['center_detection']['sampling_time']),
                exp_time=np.max(calib.camera.exp_times['day'])
            )
            diameter_exposed = 2 * radius
        else:
            diameter_exposed = min(calib.camera.ocam_model.height, calib.camera.ocam_model.width)

    if xy_center_in_config and not diameter_in_config:
        diameter_exposed = min(calib.camera.ocam_model.height, calib.camera.ocam_model.width)
    elif diameter_in_config:
        diameter_exposed = calib.camera.ocam_model.diameter

    orbs = []
    cfg_calib['orb_types'] = [v.lower() for v in cfg_calib['orb_types']]
    if 'moon' in cfg_calib['orb_types']:
        orb = Moon(calib.camera.latitude, calib.camera.longitude, calib.camera.altitude,
                   (calib.ocam.y_center, calib.ocam.x_center), diameter_exposed,
                   np.min(calib.camera.exp_times['night']), thresholds=cfg_calib['moon_detection']['thresholds'])
        if 'number_days' in cfg_calib['moon_detection']:
            first_timestamp = last_timestamp_calibration - timedelta(days=cfg_calib['moon_detection']['number_days'])
            first_timestamp = np.max([first_timestamp, calib.camera.start_recording])
            orb.timestamps_from_moon_period([first_timestamp, last_timestamp_calibration],
                                            sampling_time=timedelta(minutes=
                                                                    cfg_calib['moon_detection']['sampling_time']))
        else:
            orb.timestamps_from_moon_period(last_timestamp_calibration, sampling_time=timedelta(
                minutes=cfg_calib['moon_detection']['sampling_time']))

        orbs.append(orb)

    if 'sun' in cfg_calib['orb_types']:
        orb = Sun(calib.camera.latitude, calib.camera.longitude, calib.camera.altitude,
                  (calib.ocam.y_center, calib.ocam.x_center), diameter_exposed, np.min(calib.camera.exp_times['day']),
                  thresholds=cfg_calib['sun_detection']['thresholds'])
        first_timestamp = last_timestamp_calibration - timedelta(days=cfg_calib['sun_detection']['number_days'])
        first_timestamp = np.max([first_timestamp, calib.camera.start_recording])
        orb.timestamps_from_daytime(first_timestamp, last_timestamp_calibration,
                                    sampling_time=timedelta(minutes=cfg_calib['sun_detection']['sampling_time']))
        orbs.append(orb)

    min_date = datetime.now(tzlocal())
    max_date = pytz.UTC.localize(datetime(1, 1, 1, 0, 0, 0))

    for orb in orbs:
        min_date = np.minimum(min_date, np.min(orb.timestamps))
        max_date = np.maximum(max_date, np.max(orb.timestamps))
    orb_data = calib.get_all_orb_positions(orbs, f'{"".join(cfg_calib["orb_types"])}_observations_'
                                                 f'{min_date:%Y%m%d%H%M%S}_{max_date:%Y%m%d%H%M%S}.csv')

    if not len(orb_data):
        logging.info('Not any valid observations found in image processing. Quitting the calibration.')
        return

    if cfg_calib['target_calibration'] == 'optimize_eor_ior_center':
        calib, _ = self_calibration.find_center_via_cam_model(
            calib, orb_data, x_samples=cfg_calib['center_detection']['x_samples'],
            max_rel_center_dev=cfg_calib['center_detection']['max_rel_center_dev'],
            number_iterations=cfg_calib['center_detection']['number_iterations'])
    elif cfg_calib['target_calibration'] == 'optimize_eor_ior':
        orientation_optimized = calib.optimize_eor_ior(orb_data, [0, np.pi, 0], calib.ocam)
    elif cfg_calib['target_calibration'] == 'optimize_eor':
        orientation_optimized = calib.optimize_eor(orb_data, [0, np.pi, 0])
    else:
        raise Exception('Check your configuration! Unexpected value in "target_calibration"')

    if cfg_calib['target_calibration'] in ['optimize_eor', 'optimize_eor_ior']:
        calib.camera.external_orientation = orientation_optimized.x[:3]

    if cfg_calib['target_calibration'] in ['optimize_eor_ior']:
        calib.ocam.ss[[0, 2, 3]] = orientation_optimized.x[3:6]

    logging.info(f'Ocam parameters ss in optimized ocam model {np.squeeze(calib.ocam.ss)}')
    logging.info(f'Orientation in optimized camera model {np.rad2deg(calib.camera.external_orientation)}')
    logging.info(f'x,y_center {(calib.ocam.x_center, calib.ocam.y_center)}')

    results_dict = {'x_center': float(calib.ocam.x_center), 'y_center': float(calib.ocam.y_center),
                    'ss': [float(si) for si in calib.ocam.ss],
                    'external_orientation': [float(eori) for eori in calib.camera.external_orientation]}

    results_file = f'calib_{calib.camera.name}_{min_date:%Y%m%d%H%M%S}_{max_date:%Y%m%d%H%M%S}'
    scio.savemat(results_file + '.mat', results_dict)
    results_file = results_file + '.yaml'
    with open(results_file, 'w') as results_handle:
        yaml.dump(results_dict, results_handle, default_flow_style=False)

    if cfg_calib['compute_and_save_azimuth_elevation']:
        # Recompute and save azimuth and elevation matrices after calibration
        calib.compute_and_save_azimuth_elevation(
            ocam_model=calib.camera.ocam_model,
            min_ele_evaluated=calib.camera.min_ele_evaluated,
            external_orientation=calib.camera.external_orientation,
            save_npy=True
        )

    orb_data = calib.angles_to_pixels(orb_data, calib.ocam, calib.camera.external_orientation)
    _, orb_data = calib.angles_pixels_to_vector_deviation(orb_data, calib.ocam, calib.camera.external_orientation,
                                                          compute_found_angles=True)
    final_orb_obs_file = f'calibrated_observations_{min_date:%Y%m%d%H%M%S}_{max_date:%Y%m%d%H%M%S}.csv'
    orb_data.to_csv(final_orb_obs_file)

    self_calibration.get_background_img_and_plot(calib.camera, final_orb_obs_file, orbs[0].exp_time,
                                                 f'calibrated_observations_{min_date:%Y%m%d%H%M%S}_'
                                                 f'{max_date:%Y%m%d%H%M%S}.png',
                                                 cfg_calib['ignore_outliers_above_percentile'])

    validate_from_images(last_timestamp_calibration, results_file)


def validate_from_images(last_timestamp_calibration, path_calib_results=None):
    """
    Test a camera's IOR and EOR based on a csv of orb positions which were detected in advance

    :param last_timestamp_calibration: tz-aware datetime, last timestamp included in the calibration to be validated
    :param path_calib_results: str, path to mat file of calibration results to be validated
    """
    cfg_calib = config.get('Calibration')

    calib = self_calibration.Calibration(last_timestamp_calibration,
                                         ignore_outliers_above_percentile=cfg_calib['ignore_outliers_above_percentile'],
                                         min_rel_dist_mask_orb=cfg_calib['min_rel_dist_mask_orb'],
                                         save_orb_quality_indicators=cfg_calib['save_orb_quality_indicators'])

    if path_calib_results is None:
        path_calib_results = cfg_calib['path_calib_results']
    with open(path_calib_results, 'r') as stream:
        calib_dict = yaml.load(stream, Loader=yaml.Loader)

    calib.ocam.x_center = calib_dict['x_center']
    calib.ocam.y_center = calib_dict['y_center']
    calib.ocam.ss = np.asarray(calib_dict['ss'])
    calib.camera.external_orientation[:] = calib_dict['external_orientation']
    if calib.camera.ocam_model.diameter is None:
        calib.camera.ocam_model.diameter = min(calib.camera.ocam_model.height, calib.camera.ocam_model.width)

    # instantiate orbs over validation period
    if 'last_timestamp_validation' in cfg_calib:
        last_timestamp_validation = cfg_calib['last_timestamp_validation']
    else:
        last_timestamp_validation = last_timestamp_calibration
    cfg_calib['orb_types_validation'] = [v.lower() for v in cfg_calib['orb_types_validation']]
    orbs_validation = []
    if 'moon' in cfg_calib['orb_types_validation']:
        orb = Moon(calib.camera.latitude, calib.camera.longitude, calib.camera.altitude,
                   (calib.ocam.y_center, calib.ocam.x_center), calib.camera.ocam_model.diameter,
                   np.min(calib.camera.exp_times['night']), thresholds=cfg_calib['moon_detection']['thresholds'])
        if 'number_days' in cfg_calib['moon_detection']:
            first_timestamp = last_timestamp_validation - timedelta(days=cfg_calib['moon_validation']['number_days'])
            first_timestamp = np.max([first_timestamp, calib.camera.start_recording])
            orb.timestamps_from_moon_period([first_timestamp, last_timestamp_validation],
                                            sampling_time=timedelta(minutes=
                                                                    cfg_calib['moon_validation']['sampling_time']))
        else:
            orb.timestamps_from_moon_period(last_timestamp_validation, sampling_time=timedelta(
                minutes=cfg_calib['moon_validation']['sampling_time']))
        orbs_validation.append(orb)

    if 'sun' in cfg_calib['orb_types_validation']:
        orb = Sun(calib.camera.latitude, calib.camera.longitude, calib.camera.altitude,
                  (calib.ocam.y_center, calib.ocam.x_center), calib.camera.ocam_model.diameter,
                  np.min(calib.camera.exp_times['day']), thresholds=cfg_calib['sun_detection']['thresholds'])
        first_timestamp = last_timestamp_validation - timedelta(days=cfg_calib['sun_validation']['number_days'])
        first_timestamp = np.max([first_timestamp, calib.camera.start_recording])
        orb.timestamps_from_daytime(first_timestamp, last_timestamp_validation,
                                    sampling_time=timedelta(minutes=cfg_calib['sun_validation']['sampling_time']))
        orbs_validation.append(orb)

    min_date = datetime.now(tzlocal())
    max_date = pytz.UTC.localize(datetime(1, 1, 1, 0, 0, 0))

    for orb in orbs_validation:
        min_date = np.minimum(min_date, np.min(orb.timestamps))
        max_date = np.maximum(max_date, np.max(orb.timestamps))

    if len(orbs_validation):
        # get all orb positions
        orb_data_validation = calib.get_all_orb_positions(orbs_validation,
                                                          f'{"".join(cfg_calib["orb_types_validation"])}'
                                                          f'_{min_date:%Y%m%d%H%M%S}_{max_date:%Y%m%d%H%M%S}.csv')
        orb_data_validation = calib.angles_to_pixels(orb_data_validation, calib.ocam, calib.camera.external_orientation)
        _, orb_data_validation = calib.angles_pixels_to_vector_deviation(orb_data_validation, calib.ocam,
                                                                         calib.camera.external_orientation,
                                                                         compute_found_angles=True)
        final_orb_obs_file = f'validation_observations_{min_date:%Y%m%d%H%M%S}_{max_date:%Y%m%d%H%M%S}.csv'
        orb_data_validation.to_csv(final_orb_obs_file)

        if not len(orb_data_validation):
            logging.info('Not any valid observations suited for the validation found in image processing. Quitting the '
                         'calibration.')
            return

        self_calibration.get_background_img_and_plot(calib.camera, final_orb_obs_file, orbs_validation[0].exp_time,
                                                     f'validation_observations_{min_date:%Y%m%d%H%M%S}_'
                                                     f'{max_date:%Y%m%d%H%M%S}.png',
                                                     cfg_calib['ignore_outliers_above_percentile'])

    if cfg_calib['compute_and_save_azimuth_elevation']:
        # Recompute and save azimuth and elevation matrices after calibration
        calib.compute_and_save_azimuth_elevation(
            ocam_model=calib.camera.ocam_model,
            min_ele_evaluated=calib.camera.min_ele_evaluated,
            external_orientation=calib.camera.external_orientation,
            save_npy=True
        )


def calibrate_from_csv():
    """
    Determine a camera's IOR and EOR based on a csv of orb positions which were detected in advance
    """

    cfg_calib = config.get('Calibration')

    orb_data = pd.read_csv(cfg_calib['path_orb_observations'], converters={'timestamp': pd.to_datetime})

    if not len(orb_data):
        logging.info('Not any valid observations found in provided csv file. Quitting the calibration.')
        return

    calib = self_calibration.Calibration(orb_data.timestamp, ss_expected=cfg_calib['ss_statistics'],
                                         ignore_outliers_above_percentile=cfg_calib['ignore_outliers_above_percentile'],
                                         min_rel_dist_mask_orb=cfg_calib['min_rel_dist_mask_orb'])

    if cfg_calib['sort_out_imgs_manually']:
        orb_data = sort_out_imgs_manually(calib, orb_data)

    if not len(orb_data):
        logging.info('Not any valid observations found after manually sorting out invalid images. Quitting the '
                     'calibration.')
        return

    calib, _ = self_calibration.find_center_via_cam_model(
        calib, orb_data, x_samples=cfg_calib['center_detection']['x_samples'],
        max_rel_center_dev=cfg_calib['center_detection']['max_rel_center_dev'],
        number_iterations=cfg_calib['center_detection']['number_iterations'])

    results_dict = {'x_center': float(calib.ocam.x_center), 'y_center': float(calib.ocam.y_center),
                    'ss': [float(si) for si in calib.ocam.ss],
                    'external_orientation': [float(eori) for eori in calib.camera.external_orientation]}

    min_date = pd.to_datetime(orb_data.timestamp.min())
    max_date = pd.to_datetime(orb_data.timestamp.max())

    id_str = (f'{calib.camera.name}_{min_date:%Y%m%d%H%M%S}_to_{max_date:%Y%m%d%H%M%S}_processed_'
              f'{datetime.now():%Y%m%d%H%M%S}')
    results_file = f'recalibrated_EOR_IOR_{id_str}.mat'
    scio.savemat(results_file, results_dict)    
    results_file = f'recalibrated_EOR_IOR_{id_str}.yaml'
    with open(results_file, 'w') as results_handle:
        yaml.dump(results_dict, results_handle, default_flow_style=False)

    orb_data = calib.angles_to_pixels(orb_data, calib.ocam, calib.camera.external_orientation)
    _, orb_data = calib.angles_pixels_to_vector_deviation(orb_data, calib.ocam, calib.camera.external_orientation,
                                                          compute_found_angles=True)
    final_orb_obs_file = f'recalibrated_{id_str}.csv'
    orb_data.to_csv(final_orb_obs_file)

    if cfg_calib['orb_types'] == ['Moon']:
        exp_time = calib.camera.exp_times['night'][0]
    else:
        exp_time = calib.camera.exp_times['day'][0]

    self_calibration.get_background_img_and_plot(calib.camera, final_orb_obs_file, exp_time,
                                                 f'recalibrated_observations_{id_str}',
                                                 cfg_calib['ignore_outliers_above_percentile'])

    if cfg_calib['compute_and_save_azimuth_elevation']:
        # Recompute and save azimuth and elevation matrices after calibration
        calib.compute_and_save_azimuth_elevation(
            ocam_model=calib.camera.ocam_model,
            min_ele_evaluated=calib.camera.min_ele_evaluated,
            external_orientation=calib.camera.external_orientation,
            save_npy=True
        )


def validate_from_csv():
    """
    Test a camera's IOR and EOR based on a csv of orb positions which were detected in advance
    """

    cfg_calib = config.get('Calibration')

    orb_data = pd.read_csv(cfg_calib['path_orb_observations'], converters={'timestamp': pd.to_datetime})

    if not len(orb_data):
        logging.info('Not any valid observations found in provided csv file. Quitting the calibration.')
        return

    calib = self_calibration.Calibration(orb_data.timestamp, ss_expected=cfg_calib['ss_statistics'],
                                         ignore_outliers_above_percentile=cfg_calib['ignore_outliers_above_percentile'],
                                         min_rel_dist_mask_orb=cfg_calib['min_rel_dist_mask_orb'])

    if cfg_calib['filter_detected_orbs']:
        orb_data = filter_detected_orbs(cfg_calib, orb_data)

    if cfg_calib['sort_out_imgs_manually']:
        orb_data = sort_out_imgs_manually(calib, orb_data)

    if not len(orb_data):
        logging.info('Not any valid observations found after manually sorting out invalid images. Quitting the '
                     'calibration.')
        return

    with open(cfg_calib['path_calib_results'], 'r') as stream:
        calib_dict = yaml.load(stream, Loader=yaml.Loader)

    calib.ocam.x_center = calib_dict['x_center']
    calib.ocam.y_center = calib_dict['y_center']
    calib.ocam.ss = np.asarray(calib_dict['ss'])
    calib.camera.external_orientation[:] = calib_dict['external_orientation']

    min_date = pd.to_datetime(orb_data.timestamp.min())
    max_date = pd.to_datetime(orb_data.timestamp.max())
    id_str = f'{calib.camera.name}_{min_date:%Y%m%d%H%M%S}_to_{max_date:%Y%m%d%H%M%S}_processed_' \
             f'{datetime.now():%Y%m%d%H%M%S}'

    orb_data = calib.angles_to_pixels(orb_data, calib.ocam, calib.camera.external_orientation)
    _, orb_data = calib.angles_pixels_to_vector_deviation(orb_data, calib.ocam, calib.camera.external_orientation,
                                                          compute_found_angles=True)
    final_orb_obs_file = f'recalibrated_{id_str}_validation_period.csv'

    orb_data.to_csv(final_orb_obs_file)

    if cfg_calib['orb_types_validation'] == ['Moon']:
        exp_time = calib.camera.exp_times['night'][0]
    else:
        exp_time = calib.camera.exp_times['day'][0]

    self_calibration.get_background_img_and_plot(calib.camera, final_orb_obs_file, exp_time,
                                                 f'recalibrated_observations_{id_str}',
                                                 cfg_calib['ignore_outliers_above_percentile'])

    if cfg_calib['compute_and_save_azimuth_elevation']:
        # Recompute and save azimuth and elevation matrices after calibration
        calib.compute_and_save_azimuth_elevation(
            ocam_model=calib.camera.ocam_model,
            min_ele_evaluated=calib.camera.min_ele_evaluated,
            external_orientation=calib.camera.external_orientation,
            save_npy=True
        )


def sort_out_imgs_manually(calib, orb_data):
    """
    Use this function to copy images in which orbs were detected to the working dir. Then manually erase invalid images.

    If you may run calibrate_from_csv twice on the same set of images and you have already sorted out the images
    manually in the first run answer 'no' to the question 'Do you really want to copy images to used_imgs folder?'.

    If the RMSD is larger then around 6 consider manually sorting out images with clouds in the area around the sun and
    images where the sun disk would be cut/ obscured by the mask boundary. Otherwise you can save time and skip the
    manual sorting.

    :param calib: Calibration instance
    :param orb_data: Dataframe of astronomically expected and image-processing-based orb positions
    :return: Dataframe orb_data without rows for which the corresponding images were sorted out manually
    """
    os.makedirs('used_imgs', exist_ok=True)
    if input('Do you really want to copy images to used_imgs folder?') in ['Y', 'y', 'yes']:
        logging.info('User has chosen to generate a new selection of valid images.')
        for _, ti in orb_data.iterrows():
            img_path = calib.camera.get_img_path(ti.timestamp, exp_time=ti.exp_time)
            shutil.copyfile(img_path, 'used_imgs/' + os.path.split(img_path)[-1])
        logging.info('User is requested to sort out invalid images manually.')
        input('Please sort out images manually! Then press enter.')
        logging.info('User has confirmed that invalid images have been sorted out manually.')
    else:
        logging.info('User has chosen to reuse a prior selection of valid images.')

    orb_data['is_valid'] = True
    checked_imgs = glob.glob('used_imgs/*')
    checked_img_dates = []
    filename_pattern = Path(calib.camera.img_path_structure).name
    filename_re = fstring_to_re(filename_pattern)

    for path in checked_imgs:
        info = re.search(filename_re, path)
        datetime_pattern = re.search(r'{timestamp:(.*?)}', filename_pattern).groups()[0]
        checked_img_dates.append(calib.camera.img_timezone.localize(
            datetime.strptime(info.group('timestamp'), datetime_pattern)).astimezone(pytz.timezone('UTC')))

    for it, ti in orb_data.iterrows():
        if ti.timestamp not in checked_img_dates:
            orb_data.loc[it, 'is_valid'] = False
    return orb_data.loc[orb_data.is_valid, [k for k in orb_data.keys() if not k == 'is_valid']]


def filter_detected_orbs(cfg_calib, orb_data):
    """
    Filter out low-quality orb observations.

    :param cfg_calib: Dictionary with configuration parameters
    :param orb_data: Dataframe of orb observations
    :return: Filtered of with orb observations
    """

    if cfg_calib['orb_types_validation'] == ['Moon']:
        thresholds = cfg_calib['moon_detection']['thresholds']
    elif cfg_calib['orb_types_validation'] == ['Sun']:
        thresholds = cfg_calib['sun_detection']['thresholds']
    else:
        logging.error('Refiltering only implemented for orb_types_validation exactly one of the following Sun or '
                      'Moon')
        raise Exception('Not implemented. See log file.')

    return orb_data[(orb_data.aspect_ratio >= 1 - thresholds['aspect_ratio_tolerance'])
                    & (orb_data.circularity >= thresholds['circularity_threshold'])
                    & (orb_data.area >= thresholds['min_area'])
                    & (orb_data.area <= thresholds['max_area'])]

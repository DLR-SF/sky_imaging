# Licensed under the Apache License: http://www.apache.org/licenses/LICENSE-2.0
# For details: https://github.com/DLR-SF/sky_imaging/blob/main/NOTICE.txt

"""
Main tool for the all-sky imager self calibration
"""
import logging
import argparse
from datetime import datetime, timedelta
import pathlib
import pytz
import ephem

from asi_core import config
from asi_core.calibration import process_self_calibration as process_calib


if __name__ == '__main__':
    """
    Run the geometric calibration tool based on ASI images or based on a preprocessed csv table of orb positions.
            
    :param config: path to config file. If not provided a file 'self_calibration_cfg.yaml' is expected in the working 
        directory
    :param last_timestamp: Last timestamp included in calibration in ISO format including timezone. If not provided the 
        timestamp will be read from the config file or if not found there set to six days after the most recent full 
        moon date.
    :param mode: One of four modes:
        calibrate_from_csv: Perform only the calibration using a csv file of orb observations 
        validate_from_csv: Perform only the validation using a csv file of orb observations 
        calibrate_validate_from_images: Perform calibration and validation receiving orb positions from image files
        validate_from_images: Perform only the validation receiving orb positions from image files
    """
    IMPLEMENTED_EVALUATION_MODES = ['calibrate_validate_from_images', 'validate_from_images',
                                    'calibrate_from_csv', 'validate_from_csv']

    logging.basicConfig(filename=f'geometric_calib_processed_{datetime.now():%Y%m%d%H%M%S}.log', level=logging.INFO)

    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--config', nargs=1, default=[pathlib.Path.cwd() /
                                                            'self_calibration_cfg.yaml'])
    parser.add_argument('--last_timestamp', nargs=1, default=[None])
    parser.add_argument('--mode', nargs=1, default=[None])

    args = parser.parse_args()
    config.load_config(args.config[0])

    calib_params = config.get('Calibration')

    if args.mode[0] in IMPLEMENTED_EVALUATION_MODES:
        mode = args.mode[0]
        logging.info(f'Running calibration tool in mode: {mode} (received from user input)')
    elif calib_params.get('mode', 'not_expected') in IMPLEMENTED_EVALUATION_MODES:
        mode = calib_params.get('mode', 'not_expected')
        logging.info(f'Running calibration tool in mode: {mode} (received from config file)')
    else:
        raise Exception('No valid evaluation mode was selected.')

    if mode == 'calibrate_from_csv':
        process_calib.calibrate_from_csv()
    elif mode == 'validate_from_csv':
        process_calib.validate_from_csv()
    else:
        if args.last_timestamp[0] is not None:
            last_timestamp = args.last_timestamp[0]
        elif 'last_timestamp' in calib_params.keys():
            last_timestamp = calib_params['last_timestamp']
        else:
            last_timestamp = pytz.utc.localize(ephem.previous_full_moon(datetime.now()).datetime()) + timedelta(days=6)

        if type(last_timestamp) is str:
            last_timestamp = datetime.fromisoformat(last_timestamp)
        if mode == 'calibrate_validate_from_images':
            process_calib.calibrate_from_images(last_timestamp)
        elif mode == 'validate_from_images':
            process_calib.validate_from_images(last_timestamp)


# Licensed under the Apache License: http://www.apache.org/licenses/LICENSE-2.0
# For details: https://github.com/DLR-SF/sky_imaging/blob/main/NOTICE.txt

"""Tool for the ASI image data acquisition"""

import requests
import threading
import os
import pathlib
import pytz
from argparse import ArgumentParser
from datetime import datetime, timedelta, date
import time
import logging
import re
import pvlib
import numpy as np
import ephem
from PIL import Image
from io import BytesIO

from asi_core import config


EXIF_KEYS = {'cam_brand': 271, 'cam_model': 272, 'software': 305, 'img_modify_date': 306, 'exposure_time': 33434,
             'f_number': 33437, 'ISO': 34855, 'timezone_offset': 34858, 'sensitivity_type': 34864,
             'datetime_taken': 36867, 'datetime_digitized': 36868, 'color_space': 40961, 'exposure_mode': 41986,
             'cam_serial_number': 42033}
"""defined by the exif metadata convention, see e.g. https://exiftool.org/TagNames/EXIF.html"""

KNOWN_MOBOTIX_MODELS = ['Q71', 'Q26', 'Q25', 'Q24']
"""Mobotix camera models which are known to be compatible with the data acquisition"""


detected_local_tz = datetime.now().astimezone().tzinfo


class ImageReceiver:
    """
    Handles the image acquisition of an IP camera with http-API.

    :param url_snapshot: str, URL which is sent to camera to get a snapshot
    :param storage_path: path to which image series will be stored; folders for month, day, hour will be created
    :param location: dict, with fields indicating coordinates:
        "lat" (decimal degrees, northing positive),
        "lon" (decimal degrees, easting positive),
        "alt" (meters)
    :param sampling_time: datetime timedelta -- temporal offset between image series
    :param night_sampling_time: datetime timedelta -- temporal offset between image series during nighttime
    :param timezone_images: Specify pytz.timezone of the images and log file. If not provided computer timezone is
        used.
    """

    min_sun_ele_exp = 0  # degree, sun elevation above which images will be taken
    max_sun_ele_exp = 90  # degree, sun elevation below which images will be taken
    min_moon_ele = 10  # degree, moon elevation above which images will be taken
    min_moon_phase = 0.8

    def __init__(self, url_snapshot, storage_path, location, sampling_time=timedelta(seconds=15),
                 night_sampling_time=timedelta(seconds=60), timezone_images=detected_local_tz):
        """
        Initializes ImageReceiver with user parameters

        :param url_snapshot: str, URL which is sent to camera to get a snapshot
        :param storage_path: path to which image series will be stored; folders for month, day, hour will be created
        :param location: dict, with fields indicating coordinates:
            "lat" (decimal degrees, northing positive),
            "lon" (decimal degrees, easting positive),
            "alt" (meters)
        :param sampling_time: datetime timedelta -- temporal offset between image series
        :param night_sampling_time: datetime timedelta -- temporal offset between image series during nighttime
        :param timezone_images: Specify pytz.timezone of the images and log file. If not provided computer timezone is
            used.
        """
        self.url_snapshot = url_snapshot
        self.storage_path = storage_path
        self.sampling_time = sampling_time
        self.location = location
        self.timezone_images = timezone_images
        self.day_sampling_time = sampling_time
        self.night_sampling_time = night_sampling_time
        self.record_night_image = False
        self.midnight = None
        self.ephem_observer = ephem.Observer()
        self.ephem_observer.lat = str(location['lat'])
        self.ephem_observer.lon = str(location['lon'])
        self.ephem_observer.elevation = location['alt']

    def url_get_snapshot(self):
        """
        Sends a URL request and extract binary JPEG from response

        :returns: Binary encoded JPEG image
        """

        snap_resp = requests.get(self.url_snapshot)

        return snap_resp.content

    def store_img_to_path(self, ts, img, exp_time=0):
        """
        Stores an image record in the folder structure according to the convention below

        :param ts: datetime, timestamp at which the image was taken
        :param img: binary encoded image (jpg)
        :param exp_time: int, exposure time of the image (microseconds)
        """

        target_folder = f'{self.storage_path}/{ts:%m}/{ts:%d}/{ts:%H}'
        if not os.path.isdir(target_folder):
            os.makedirs(target_folder)
        target_file = f'{target_folder}/{ts:%Y%m%d%H%M%S}_{exp_time:01}.jpg'
        with open(target_file, 'wb') as tf:
            tf.write(img)

    def record_continuously(self):
        """
        Once called, this function runs infinitely taking images at the defined sampling rate
        """

        # check if conditions are met to record an image
        record_image = self.prepare_image_acquisition()
        # schedule the next execution of record_continuously
        timer = threading.Timer(self.sampling_time.seconds, self.record_continuously)
        timer.start()

        if not record_image:
            logging.info('Conditions to record image not met.')
            return

        while True:
            """ stay in this loop until end of next sampling interval
            to make sure that samples are received from full minutes etc."""
            now = datetime.now().astimezone(self.timezone_images)
            time_since_midnight = now - self.midnight
            frac_seconds = time_since_midnight.seconds / self.sampling_time.seconds
            floor_frac_seconds = time_since_midnight.seconds // self.sampling_time.seconds
            if frac_seconds - floor_frac_seconds < 1 / self.sampling_time.seconds:
                break
            time.sleep(0.1)

        self.trigger_store_exposure()

    def trigger_store_exposure(self):
        """
        Triggers, receives and saves a snapshot from the camera.

        Exposure time etc. remain unchanged. Only a snapshot is requested and saved
        """
        now = datetime.now().astimezone(self.timezone_images)
        logging.info(f'taking snapshot for {now}')
        snapshot = self.url_get_snapshot()
        logging.info(f'snapshot taken before {datetime.now().astimezone(self.timezone_images)}')
        self.store_img_to_path(now, snapshot)
        logging.info(f'snapshot stored before {datetime.now().astimezone(self.timezone_images)}')

    def prepare_image_acquisition(self):
        """
        Prepare image acquisition.

        Check if conditions for recording images are met. Either for standard daylight images (depending on sun
        elevation) or nighttime images when there is (almost) full moon and the moon is above the horizon.

        :returns: Boolean whether image should be recorded.
        """
        self.midnight = datetime.combine(date.today(), datetime.min.time()).astimezone(self.timezone_images)
        now = datetime.utcnow()
        sun_pos = pvlib.solarposition.get_solarposition(now, self.location['lat'], self.location['lon'],
                                                        altitude=self.location['alt'])
        # Use ephem observer to get information about moon phase and elevation
        # Datetime needs to be UTC for ephem
        self.ephem_observer.date = now
        moon = ephem.Moon(self.ephem_observer)
        moon_phase = moon.moon_phase
        moon_ele = np.rad2deg(moon.alt.real)
        sun_ele = sun_pos['apparent_elevation'].iloc[0]
        logging.info(f'Current sun elevation is {sun_ele:.1f} degree')
        # Check if image acquisition is at night (e.g., for determining exposure times)
        self.record_night_image = (moon_phase > self.min_moon_phase) & (moon_ele > self.min_moon_ele) & (sun_ele < 0)
        self.record_night_image &= self.night_sampling_time.seconds > 0

        # Adjust sampling time at night
        if self.record_night_image:
            self.sampling_time = self.night_sampling_time
        else:
            self.sampling_time = self.day_sampling_time

        record_image = (self.min_sun_ele_exp <= sun_ele <= self.max_sun_ele_exp) | self.record_night_image
        return record_image


class SeriesReceiver(ImageReceiver):
    """
    Defines the acquisition of image series with different exposure times.

    :param url_cam: URL/ IP of the camera to be controlled
    :param storage_path: see ImageReceiver
    :param location: see ImageReceiver
    :param sampling_time: see ImageReceiver
    :param night_sampling_time: timedelta, sampling time used during nighttime
    :param exposure_times: list, exposure times in microseconds. One snapshot per exp. time will be stored
    :param settling_time: Time to wait until requesting an image from camera after setting new exposure time or
        after previous request
    :param timezone_images: see ImageReceiver
    :param rel_threshold_exposure: Relative deviation of exp. time. Image will be rejected if real exp. time
        deviates stronger from expected value
    :param safety_deadtime_between_exposure_series: Cancel acquisition of exposure series if less than this time
        remains until end of sampling period
    """
    def __init__(self, url_cam, storage_path, location, sampling_time=timedelta(seconds=15),
                 night_sampling_time=timedelta(seconds=60), exposure_times=[80, 160, 320, 640, 1280],
                 night_exposure_times=[80000], settling_time=timedelta(seconds=0.25),
                 timezone_images=detected_local_tz, rel_threshold_exposure=0.1,
                 safety_deadtime_between_exposure_series=timedelta(seconds=3)):
        """
        Initializes SeriesReceiver

        :param url_cam: URL/ IP of the camera to be controlled
        :param storage_path: see ImageReceiver
        :param location: see ImageReceiver
        :param sampling_time: see ImageReceiver
        :param exposure_times: list, exposure times in microseconds. One snapshot per exp. time will be stored
        :param settling_time: Time to wait until requesting an image from camera after setting new exposure time or
            after previous request
        :param timezone_images: see ImageReceiver
        :param rel_threshold_exposure: Relative deviation of exp. time. Image will be rejected if real exp. time
            deviates stronger from expected value
        :param safety_deadtime_between_exposure_series: Cancel acquisition of exposure series if less than this time
            remains until end of sampling period
        """
        ImageReceiver.__init__(self, '/'.join([url_cam, self.command_record]), storage_path, location, sampling_time, 
                               night_sampling_time=night_sampling_time, timezone_images=timezone_images)
        self.exposure_times = exposure_times
        self.night_exposure_times = night_exposure_times
        self.url_cam = url_cam
        self.settling_time = settling_time
        self.rel_threshold_exposure = rel_threshold_exposure
        self.safety_deadtime_between_exposure_series = safety_deadtime_between_exposure_series

    def set_exposure(self, exp_set):
        """"
        Request camera to change exposure time

        :param exp_set: Exposure time in microseconds
        """
        raise NotImplementedError('This method needs to be implemented in an inheriting class')

    def trigger_store_exposure(self):
        """
        Records an image series with different exposure times
        """
        
        start_time_series = datetime.now().astimezone(self.timezone_images)
        latest_end_time_series = start_time_series + self.sampling_time - self.safety_deadtime_between_exposure_series

        snapshots = []
        if self.record_night_image:
            exposure_times = self.night_exposure_times
        else:
            exposure_times = self.exposure_times
        for exp_i in exposure_times:
            logging.info(f'setting exposure to {exp_i}')
            self.set_exposure(exp_i)
            while True:
                time.sleep(self.settling_time.total_seconds())
                now = datetime.now().astimezone(self.timezone_images)
                logging.info(f'taking snapshot for {now}')
                snapshot = self.url_get_snapshot()

                real_exp = self.get_real_exposure(snapshot)
                
                sampling_time_exceeded = datetime.now().astimezone(self.timezone_images) > latest_end_time_series
                
                if abs(real_exp - exp_i)/exp_i < self.rel_threshold_exposure:
                    now = datetime.now().astimezone(self.timezone_images)
                    snapshots.append({'exp_time': exp_i, 'img': snapshot, 'timestamp': now})
                    logging.info(f'snapshot taken before {now}')
                    break
                elif sampling_time_exceeded:
                    break

            if sampling_time_exceeded:
                logging.info(f'Image series for {start_time_series} could not be completed. Camera too slow.')
                break
            
        self.set_exposure(np.min(np.asarray(exposure_times)))
        for exposure in snapshots:
            self.store_img_to_path(exposure['timestamp'], exposure['img'], exposure['exp_time'])

        logging.info(f'snapshots stored before {datetime.now().astimezone(self.timezone_images)}')

    @staticmethod
    def get_real_exposure(img):
        """
        Checks the exposure time noted in the EXIF header.

        :param img: Binary encoded jpeg image with EXIF header that contains actual exposure time
        :returns exp: int, read exposure time in microseconds
        """
        with Image.open(BytesIO(img)) as img_file:
            exif_header = img_file._getexif()
        return int(exif_header[EXIF_KEYS['exposure_time']]*1e6)


class AxisSeriesReceiver(SeriesReceiver):
    """
    Handles the acquisition of image series with an AXIS camera.

    :param url_cam: see SeriesReceiver
    :param storage_path: see ImageReceiver
    :param location: see ImageReceiver
    :param user: name of user authorized to set exposure parameters
    :param password: password of user
    :param sampling_time: see ImageReceiver
    :param exposure_times: see SeriesReceiver
    :param settling_time: see SeriesReceiver
    :param user: name of user authorized to set exposure parameters
    :param password: password of user
    """

    # command appended to camera IP to receive a snapshot
    command_record = ('jpg/image.jpg?camera=1&overview=0&resolution=2016x2016&videoframeskipmode=empty&timestamp='
                      '1662794762943&Axis-Orig-Sw=true&fps=30')

    def __init__(self, url_cam, storage_path, location, user, password, **kwargs):
        """
        Initializes AxisSeriesReceiver

        Parameters see SeriesReceiver.

        :param url_cam: see SeriesReceiver
        :param storage_path: see ImageReceiver
        :param location: see ImageReceiver
        :param user: name of user authorized to set exposure parameters
        :param password: password of user
        :param sampling_time: see ImageReceiver
        :param exposure_times: see SeriesReceiver
        :param settling_time: see SeriesReceiver
        """
        SeriesReceiver.__init__(self, url_cam, storage_path, location, **kwargs)
        self.authentication = requests.auth.HTTPDigestAuth(user, password)

    def set_exposure(self, exp_set):
        """"
        Request camera to change exposure time

        :param exp_set: Exposure time in microseconds
        """
        set_exp_cmd = (rf'{self.url_cam}/axis-cgi/param.cgi?action=update&ImageSource.I0.Sensor.MaxExposureTime='
                       rf'{exp_set:.0f}&ImageSource.I0.Sensor.MinExposureTime={exp_set:.0f}')
        requests.get(set_exp_cmd, auth=self.authentication)


class MobotixSeriesReceiver(SeriesReceiver):
    """
    Handles the acquisition of image series with a Mobotix camera.

    :param url_cam: see SeriesReceiver
    :param storage_path: see ImageReceiver
    :param location: see ImageReceiver
    :param sampling_time: see ImageReceiver
    :param exposure_times: see SeriesReceiver
    :param settling_time: see SeriesReceiver
    """

    # command appended to camera IP to receive a snapshot
    command_record = 'record/current.jpg'

    def __init__(self, *args, **kwargs):
        """
        Initializes MobotixSeriesReceiver

        :param url_cam: see SeriesReceiver
        :param storage_path: see ImageReceiver
        :param location: see ImageReceiver
        :param sampling_time: see ImageReceiver
        :param exposure_times: see SeriesReceiver
        :param settling_time: see SeriesReceiver
        """
        SeriesReceiver.__init__(self, *args, **kwargs)

    def set_exposure(self, exp_set):
        """"
        Request camera to change exposure time

        :param exp_set: Exposure time in microseconds
        """
        set_exp_cmd = (f'{self.url_cam}/control/control?set&section=exposure&ca_exp_min='
                       f'{exp_set:.0f}&ca_exp_max={exp_set:.0f}')
        requests.get(set_exp_cmd)

    @staticmethod
    def get_real_exposure(img):
        """
        Checks the exposure time noted in the header of a Mobotix JPEG.

        Mobotix JPEGs possess a specific header. This is fundamental for the function.
        
        Exposure times of 1e4 seconds and more are not expected.

        :param img: Binary encoded mobotix jpeg image
        :returns: int, read exposure time in microseconds
        """
        meta_infos = img[:5000].decode('utf-8', errors='ignore')  # assume header is never longer than this
        exp = int(re.search(r"EXP=(\d{1,9})", meta_infos).groups()[0])
        logging.info(f'Based on its header, this image has an exposure time of {exp} microseconds.')
        return exp


def main():
    """
    Parse cmd line arguments and run data acquisition

    Provide the path to your config file as argument after flag -c or run the daq in a folder which contains a config
    file named 'asi_daq_cfg.yaml'
    """

    parser = ArgumentParser()
    parser.add_argument('-c', '--config', nargs=1, default=[pathlib.Path.cwd() / 'asi_daq_cfg.yaml'])
    timestamp_logfile = datetime.now()

    args = parser.parse_args()
    config.load_config(args.config[0])

    cam_cfg = config.get('Camera')
    system_cfg = config.get('System')
    optional_arguments = config.get('Process')

    if type(optional_arguments) is dict:
        if 'timezone_images' in optional_arguments.keys() and optional_arguments['timezone_images'] is not None:
            optional_arguments['timezone_images'] = pytz.timezone(optional_arguments['timezone_images'])
            timestamp_logfile = timestamp_logfile.astimezone(optional_arguments['timezone_images'])

        if 'sampling_time' in optional_arguments.keys() and optional_arguments['sampling_time'] is not None:
            optional_arguments['sampling_time'] = timedelta(seconds=optional_arguments['sampling_time'])

        if 'night_sampling_time' in optional_arguments.keys() and optional_arguments['night_sampling_time'] is not None:
            optional_arguments['night_sampling_time'] = timedelta(seconds=optional_arguments['night_sampling_time'])

        if 'settling_time' in optional_arguments.keys() and optional_arguments['settling_time'] is not None:
            optional_arguments['settling_time'] = timedelta(seconds=optional_arguments['settling_time'])

        if ('safety_deadtime_between_exposure_series' in optional_arguments.keys() and
                optional_arguments['safety_deadtime_between_exposure_series'] is not None):
            optional_arguments['safety_deadtime_between_exposure_series'] = (
                timedelta(seconds=optional_arguments['safety_deadtime_between_exposure_series']))
    else:
        optional_arguments = {}

    storage_path = fr'{system_cfg["storage_path"]}/{datetime.now():%Y}/{cam_cfg["camera_name"]}'
    os.chdir(system_cfg["daq_working_dir"])

    if not os.path.isdir('Logs'):
        os.mkdir('Logs')

    logging.basicConfig(filename=f'Logs/asi_daq_{timestamp_logfile:%Y%m%d_%H%M%S}.log',
                        level=logging.INFO, format='%(levelname)s - %(asctime)s: %(message)s', datefmt='%H:%M:%S')

    if cam_cfg['camera_model'] in KNOWN_MOBOTIX_MODELS:
        MobotixSeriesReceiver(cam_cfg['url_cam'], storage_path, cam_cfg['location'], **optional_arguments
                              ).record_continuously()
    elif cam_cfg['camera_model'][:4] == 'AXIS':
        AxisSeriesReceiver(cam_cfg['url_cam'], storage_path, cam_cfg['location'], **optional_arguments
                           ).record_continuously()
    else:
        raise Exception('Unknown camera type')


if __name__ == '__main__':
    main()

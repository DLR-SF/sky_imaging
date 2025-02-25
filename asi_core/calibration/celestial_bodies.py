# Licensed under the Apache License: http://www.apache.org/licenses/LICENSE-2.0
# For details: https://github.com/DLR-SF/sky_imaging/blob/main/NOTICE.txt

"""
Tools to detect the position of celestial bodies in images and to calculate their position in the sky astronomically

This functionality itself was introduced and described by
Niklas Blum, Paul Matteschk, Yann Fabel, Bijan Nouri, Roberto Roman, Luis F. Zarzalejo, Juan Carlos Antuna-Sanchez,
Stefan Wilbert "Geometric Calibration of All-Sky Cameras Using Sun and Moon Positions: A Comprehensive Analysis",
Solar Energy (in review).

When using the tools of this module in any publication or other work, make sure to reference this publication.
"""
import logging

import ephem
import numpy as np
import cv2
import pandas as pd
import pytz

from datetime import timedelta, datetime
from dateutil.tz import tzlocal

from asi_core import asi_analysis


THRESHOLDS_MOON_WDR = {'intensity_threshold': 240, 'aspect_ratio_tolerance': 0.2, 'circularity_threshold': 0.5,
                       'min_area': 10, 'max_area': 10000}
"""Thresholds to be used when detecting Moon in 8-bit HDR JPGs"""

THRESHOLDS_MOON = {'intensity_threshold': 100, 'aspect_ratio_tolerance': 0.1, 'circularity_threshold': 0.8,
                   'min_area': 10, 'max_area': 10000}
"""Thresholds to be used when detecting Moon in regular (LDR) JPGs"""

THRESHOLDS_SUN = {'intensity_threshold': 240, 'aspect_ratio_tolerance': 0.2, 'circularity_threshold': 0.5,
                  'min_area': 100, 'max_area': 10000}
"""Thresholds to be used when detecting Sun"""


class CelestialBodyDetector:
    """
    Detects celestial bodies in ASI images and calculates their spherical coordinates from astronomy.

    :param name: Name of the celestial body
    :param lat: Latitude of the observer/ ASI in decimal degrees north
    :param lon: Longitude of the observer/ ASI in decimal degrees east
    :param alt: Altitude above sea level
    :param center: Center of the fisheye lens in the ASI image
    :param diameter: Diameter of the exposed area in the ASI image
    :param exp_time: Required exposure time of used images
    :param thresholds: Thresholds of criteria to classify a valid contour
    """
    def __init__(self, name, lat, lon, alt, center, diameter, exp_time, thresholds=None):
        """
        Initializes CelestialBodyDetector.

        :param name: Name of the celestial body
        :param lat: Latitude of the observer/ ASI in decimal degrees north
        :param lon: Longitude of the observer/ ASI in decimal degrees east
        :param alt: Altitude above sea level
        :param center: Center of the fisheye lens in the ASI image
        :param diameter: Diameter of the exposed area in the ASI image
        :param exp_time: Required exposure time of used images
        :param thresholds: Thresholds of criteria to classify a valid contour
        """

        self.name = name
        self.lat = lat
        self.lon = lon
        self.alt = alt
        self.center = center
        self.diameter = diameter
        self.exp_time = exp_time
        self.observer = self._create_observer()
        self.body = self._create_body()
        self._set_thresholds(thresholds=thresholds)
        self.timestamps = pd.Series([], dtype='datetime64[ns]')

    def _set_thresholds(self, thresholds):
        """
        Sets thresholds for orb detection specific to a celestial body.

        :param thresholds: Dict of threshold values
        """
        self.circularity_threshold = thresholds['circularity_threshold']
        self.aspect_ratio_tolerance = thresholds['aspect_ratio_tolerance']
        self.intensity_threshold = thresholds['intensity_threshold']
        self.min_area = thresholds['min_area']
        self.max_area = thresholds['max_area']

    def _create_observer(self):
        """
        Creates and configures an ephem Observer instance.

        return: Observer instance
        """
        observer = ephem.Observer()
        observer.lat = str(self.lat)
        observer.lon = str(self.lon)
        observer.elevation = self.alt
        return observer

    def _create_body(self):
        """
        Creates a celestial body from ephem
        """
        raise NotImplementedError("Subclasses must implement _create_body method")

    def calculate_azimuth_elevation(self, timestamp):
        """
        Calculates the celestial body's spherical coordinates for the current timestamp

        :param timestamp: timezone-aware datetime to calculate orb coordinates for
        :return: azimuth and elevation angle of the orb in degree
        """
        self.observer.date = timestamp.astimezone(pytz.UTC)
        self.body.compute(self.observer)
        azimuth = self.body.az * 180 / np.pi
        elevation = self.body.alt * 180 / np.pi
        return azimuth, elevation

    def detect_celestial_body(self, img):
        """
        Detects the position of a celestial body in an image using thresholding and contour detection.

        :param img: (numpy.ndarray) image to analyze.
        :return: x and y pixel coordinates of the celestial body's center if detected or None if not detected.
        """
        if self.diameter is None:
            logging.error("Image diameter not set.")
            return None
        if self.center is None:
            logging.error("Image center not set.")
            return None

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        if self.intensity_threshold is not None:
            _, high_intensity_thresh = cv2.threshold(gray, self.intensity_threshold, 255, cv2.THRESH_BINARY)
            thresh = cv2.bitwise_and(thresh, high_intensity_thresh)

        mask = np.zeros(gray.shape, dtype=np.uint8)

        cv2.circle(mask, [int(v) for v in self.center], int(self.diameter // 2), 255, -1)

        masked_thresh = cv2.bitwise_and(thresh, thresh, mask=mask)

        contours, _ = cv2.findContours(masked_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
        else:
            logging.debug(f'No contour of sufficiently bright pixels found. (Threshold {self.intensity_threshold}), '
                          f'Found intensities in the range {np.min(gray)}...{np.max(gray)}')
            return None

        if cv2.contourArea(largest_contour) < self.min_area:
            logging.debug('Largest contour of bright pixels smaller than threshold for orb.')
            return None

        if cv2.contourArea(largest_contour) > self.max_area:
            logging.debug('Largest contour of bright pixels larger than threshold for orb.')
            return None

        is_circle_contour, aspect_ratio, circularity = asi_analysis.is_circle_contour(
            largest_contour, self.aspect_ratio_tolerance, self.circularity_threshold)

        if is_circle_contour:
            moments = cv2.moments(largest_contour)
            center_x = moments["m10"] / moments["m00"]
            center_y = moments["m01"] / moments["m00"]

            orb_props = {'exp_time': self.exp_time, 'found_x': [center_x], 'found_y': [center_y],
                         'area': cv2.contourArea(largest_contour), 'aspect_ratio': aspect_ratio,
                         'circularity': circularity}

            return orb_props
        else:
            logging.debug('Largest contour of bright pixels did not qualify as circle (circularity and aspect_ratio).')
            return None


class Moon(CelestialBodyDetector):
    """
    Detects the Moon in ASI images and calculates its spherical coordinates from astronomy.

    :param lat: Latitude of the observer/ ASI in decimal degrees north
    :param lon: Longitude of the observer/ ASI in decimal degrees east
    :param alt: Altitude above sea level
    :param center: Center of the fisheye lens in the ASI image
    :param diameter: Diameter of the exposed area in the ASI image
    :param exp_time: Required exposure time of used images
    :param min_moon_phase: Minimum moon illumination in ]0, 1[ from which timestamps are evaluated
    :param min_moon_ele: Minimum moon elevation in degree from which timestamps are evaluated
    :param max_sun_ele_night: Maximum sun elevation in degree up to which timestamps are evaluated (should be negative)
    :param thresholds: Thresholds of criteria to classify a valid contour
    """
    def __init__(self, lat, lon, alt, center, diameter, exp_time=80000, min_moon_phase=0.8, min_moon_ele=10,
                 max_sun_ele_night=-6, thresholds=None):
        super().__init__("Moon", lat, lon, alt, center, diameter, exp_time, thresholds=thresholds)
        self.min_moon_phase = min_moon_phase
        self.min_orb_ele = min_moon_ele
        self.max_sun_ele_night = max_sun_ele_night

    def _create_body(self):
        """
        Creates the celestial body Moon
        """
        return ephem.Moon()

    def _set_thresholds(self, thresholds=None):
        """
        Set thresholds for orb detection specific to the Moon

        :param thresholds: dict of thresholds or None or 'wdr'. The latter two apply predefined parameter sets
        """
        if thresholds == 'wdr':
            thresholds = THRESHOLDS_MOON_WDR
        elif thresholds is None:
            thresholds = THRESHOLDS_MOON
        super()._set_thresholds(thresholds)

    def timestamps_from_moon_period(self, timestamp_period, sampling_time=timedelta(minutes=1)):
        """
        Get the most recent full moon period before the requested timestamp

        :param timestamp_period: Requested date, moon period closest to that date will be determined
        :param sampling_time: Time difference between images of the moon period evaluated
        """
        if type(timestamp_period) is not list or len(timestamp_period) == 1:
            if type(timestamp_period) is list:
                timestamp_period = timestamp_period[0]

            most_recent_full_moon = (pytz.utc.localize(ephem.previous_full_moon(timestamp_period).datetime()).replace
                               (minute=0, second=0, microsecond=0))
            timestamp_start = most_recent_full_moon.astimezone(pytz.UTC) - timedelta(days=13)
            timestamp_end = min(datetime.now(tzlocal()).astimezone(pytz.UTC),
                                most_recent_full_moon + timedelta(days=13))

        elif len(timestamp_period) == 2:
            timestamp_start = timestamp_period[0]
            timestamp_end = timestamp_period[1]
        else:
            raise Exception('Expected one or two timestamps as first argument')

        timestamps_raw = pd.date_range(timestamp_start, timestamp_end, freq=sampling_time)

        observer = self.observer
        moon = ephem.Moon()
        sun = ephem.Sun()

        timestamps = []
        for timestamp in timestamps_raw:
            observer.date = timestamp
            moon.compute(observer)
            if moon.moon_phase < self.min_moon_phase:
                continue
            if np.rad2deg(moon.alt.real) < self.min_orb_ele:
                continue
            sun.compute(observer)
            if np.rad2deg(sun.alt.real) > self.max_sun_ele_night:
                continue
            timestamps.append(timestamp)

        if len(timestamps):
            self.timestamps = pd.DatetimeIndex(timestamps).round(sampling_time)


class Sun(CelestialBodyDetector):
    """
    Detects the Sun in ASI images and calculates its spherical coordinates from astronomy.

    :param lat: Latitude of the observer/ ASI in decimal degrees north
    :param lon: Longitude of the observer/ ASI in decimal degrees east
    :param alt: Altitude above sea level
    :param center: Center of the fisheye lens in the ASI image
    :param diameter: Diameter of the exposed area in the ASI image
    :param exp_time: Required exposure time of used images
    :param min_sun_ele_day: Minimum sun elevation in degree from which timestamps are evaluated
    :param thresholds: Thresholds of criteria to classify a valid contour
    """
    def __init__(self, lat, lon, alt, center, diameter, exp_time=160, min_sun_ele_day=10, thresholds=None):
        super().__init__("Sun", lat, lon, alt, center, diameter, exp_time, thresholds=thresholds)
        self.min_orb_ele = min_sun_ele_day

    def _create_body(self):
        """
        Creates the celestial body Sun
        """
        return ephem.Sun()

    def _set_thresholds(self, thresholds):
        """
        Set thresholds for orb detection specific to the Sun
        """
        if thresholds is None:
            thresholds = THRESHOLDS_SUN
        super()._set_thresholds(thresholds)

    def timestamps_from_daytime(self, timestamp_start, timestamp_end, sampling_time):
        """
        Gets a range of timestamps excluding any times with too low sun elevation

        :param timestamp_start: First timestamp of the period
        :param timestamp_end: Last timestamp of the period
        :param sampling_time: Time difference between images of the sun period evaluated
        """
        timestamps_raw = pd.date_range(timestamp_start.astimezone(pytz.UTC), timestamp_end.astimezone(pytz.UTC),
                                       freq=sampling_time)
        observer = self.observer
        sun = ephem.Sun()

        timestamps = []
        for timestamp in timestamps_raw:
            observer.date = timestamp
            sun.compute(observer)
            if np.rad2deg(sun.alt.real) < self.min_orb_ele:
                continue
            timestamps.append(timestamp)

        self.timestamps = pd.DatetimeIndex(timestamps).round(sampling_time)


class Venus(CelestialBodyDetector):
    def __init__(self, *args, **kwargs):
        super().__init__("Venus", args, kwargs)

    def _create_body(self):
        """
        Creates the celestial body Venus
        """
        return ephem.Venus()

    def _set_thresholds(self):
        """
        Set thresholds for orb detection specific to Venus
        """
        self.intensity_threshold = 100
        self.aspect_ratio_tolerance = 0.1
        self.circularity_threshold = 0.8


class Sirius(CelestialBodyDetector):
    def __init__(self, *args, **kwargs):
        super().__init__("Sirius", args, kwargs)

    def _create_body(self):
        """
        Creates the celestial body Sirius
        """
        return ephem.star("Sirius")

    def _set_thresholds(self):
        """
        Set thresholds for orb detection specific to Sirius
        """
        self.intensity_threshold = 100
        self.aspect_ratio_tolerance = 0.1
        self.circularity_threshold = 0.8


class Canopus(CelestialBodyDetector):
    def __init__(self, *args, **kwargs):
        super().__init__("Canopus", args, kwargs)

    def _create_body(self):
        """
        Creates the celestial body Canopus
        """
        return ephem.star("Canopus")

    def _set_thresholds(self):
        """
        Set thresholds for orb detection specific to the Canopus
        """
        self.intensity_threshold = 100
        self.aspect_ratio_tolerance = 0.1
        self.circularity_threshold = 0.8


class AlphaCentauri(CelestialBodyDetector):
    def __init__(self, *args, **kwargs):
        super().__init__("Alpha Centauri", args, kwargs)

    def _create_body(self):
        """
        Creates the celestial body Alpha Centauri
        """
        return ephem.star("Alpha Centauri")

    def _set_thresholds(self):
        """
        Set thresholds for orb detection specific to the AlphaCentauri
        """
        self.intensity_threshold = 100
        self.aspect_ratio_tolerance = 0.1
        self.circularity_threshold = 0.8


class Arcturus(CelestialBodyDetector):
    def __init__(self, *args, **kwargs):
        super().__init__("Arcturus", args, kwargs)

    def _create_body(self):
        """
        Creates the celestial body Arcturus
        """
        return ephem.star("Arcturus")

    def _set_thresholds(self):
        """
        Set thresholds for orb detection specific to the Arcturus
        """
        self.intensity_threshold = 100
        self.aspect_ratio_tolerance = 0.1
        self.circularity_threshold = 0.8


class Vega(CelestialBodyDetector):
    def __init__(self, *args, **kwargs):
        super().__init__("Vega", args, kwargs)

    def _create_body(self):
        """
        Creates the celestial body Vega
        """
        return ephem.star("Vega")

    def _set_thresholds(self):
        """
        Set thresholds for orb detection specific to the Vega
        """
        self.intensity_threshold = 100
        self.aspect_ratio_tolerance = 0.1
        self.circularity_threshold = 0.8

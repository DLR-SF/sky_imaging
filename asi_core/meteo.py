# Licensed under the Apache License: http://www.apache.org/licenses/LICENSE-2.0
# For details: https://github.com/DLR-SF/sky_imaging/blob/main/NOTICE.txt

"""
This module is for handling and organizing meteorological measurement data.
"""

import pandas as pd
import numpy as np
import pvlib
from scipy.signal import argrelextrema
import datetime
from tqdm import tqdm
from io import StringIO

from asi_core.visualization.timeseries import plot_data_distributions
from asi_core.basics import get_number_of_nans, get_temporal_resolution_from_timeseries
from asi_core.matlab_converters import map_data_columns, load_matlab_mesor_file, load_matlab_dni_classes, load_matlab_linke_turbidity_per_day

from asi_core.constants import PYTHON_LAT, PYTHON_LON, PYTHON_ALT, PYTHON_TZ, PYTHON_DT, PYTHON_PAMB
from asi_core import real_time_utility_functions
import matplotlib.pyplot as plt

class MeteoData(object):
    """Class for meteo time series data."""
    columns = ['ghi', 'dni', 'dhi', 't_amb', 'p_amb', 'rel_humid', 'wind_dir', 'wind_speed', 'sun_el', 'sun_az',
               'airmass_abs', 'linke_turbidity', 'clear_sky_ghi', 'clear_sky_dni', 'clear_sky_dhi', 'dni_var_class']
    units = {
        'ghi': 'W/m^2',
        'dni': 'W/m^2',
        'dhi': 'W/m^2',
        't_amb': '°C',
        'p_amb': 'Pa',
        'rel_humid': '%',
        'wind_dir': 'deg',
        'wind_speed': 'm/s',
        'sun_el': 'deg',
        'sun_az': 'deg',
    }
    data = pd.DataFrame(columns=columns, dtype=np.float64)

    def __init__(self, name=None, latitude=None, longitude=None, altitude=None, tz=None, data=None,
                 temporal_resolution=None):
        self.name = name
        self.latitude = latitude
        self.longitude = longitude
        self.altitude = altitude
        self.tz = tz
        if data is not None:
            self.data = data
            if self.tz is None:
                self.tz = data.index.tz
            else:
                assert self.tz == data.index.tz
        self.temporal_resolution = temporal_resolution

    def get_p_amb(self, use_default_for_nans=True):
        """Gets ambient pressure from self.data"""
        p_amb = self.data['p_amb'].copy()
        if use_default_for_nans:
            p_amb_by_altitude = pvlib.atmosphere.alt2pres(self.altitude)
            p_amb.loc[p_amb.isna()] = p_amb_by_altitude
        return p_amb

    def get_solar_pos(self, recompute=False):
        """Gets solar position from self.data"""
        all_nans = self.data[['sun_el', 'sun_az']].isna().all().all()
        if recompute or all_nans:
            return self.compute_solar_position()
        else:
            return self.data[['sun_el', 'sun_az']]

    def get_airmass(self, absolute=True, recompute=False):
        """Gets airmass from self.data"""
        all_nans = self.data['airmass_abs'].isna().all()
        if recompute or all_nans:
            airmass_abs = self.compute_airmass()
        else:
            airmass_abs = self.data['airmass_abs']
        if not absolute:
            p_amb = self.get_p_amb()
            airmass = airmass_abs / p_amb * pvlib.atmosphere.alt2pres(0)
        else:
            airmass = airmass_abs
        return airmass

    def get_linke_turbidity(self, recompute=False, method_linke_turbidity='mcclear'):
        """Gets linke turbidity from self.data"""
        all_nans = self.data['linke_turbidity'].isna().all()
        if recompute or all_nans:
            return self.compute_linke_turbidity(method_linke_turbidity=method_linke_turbidity)
        else:
            return self.data['linke_turbidity']

    def get_clear_sky_irradiance(self, recompute=False, method_clear_sky_irradiance='ineichen',
                                 method_linke_turbidity='mcclear'):
        """Gets clear sky irradiance from self.data."""
        all_nans = self.data[['clear_sky_ghi', 'clear_sky_dni', 'clear_sky_dhi']].isna().all().all()
        if recompute or all_nans:
            return self.compute_clear_sky_irradiance(method_clear_sky_irradiance=method_clear_sky_irradiance,
                                                     method_linke_turbidity=method_linke_turbidity)
        else:
            return self.data[['clear_sky_ghi', 'clear_sky_dni', 'clear_sky_dhi']]

    def get_dni_variability_classes(self):
        """Gets dni variability classes from self.data."""
        return self.data['dni_var_class']

    def remove_duplicates(self, keep='first', inplace=False):
        """Removes rows in dataframes with duplicated indexes."""
        data = self.data[~self.data.index.duplicated(keep=keep)]
        if inplace:
            self.data = data
        else:
            return data

    def insert_missing_timestamps(self, inplace=False, min_ele=None, method_solar_pos='nrel_numpy'):
        """Inserts missing timestamps"""
        data = self.data.sort_index()
        tz = self.data.index.tz
        _, ts_gaps_filled = get_missing_timestamps(
            dt=data.index,
            temporal_resolution=self.temporal_resolution,
            tz=tz,
            min_ele=min_ele,
            latitude=self.latitude,
            longitude=self.longitude,
            altitude=self.altitude,
            method_solar_pos=method_solar_pos
        )
        data = self.data.reindex(ts_gaps_filled)
        if inplace:
            self.data = data
        else:
            return data

    def compute(self):
        self.compute_solar_position()
        print('computed solar position.')
        self.compute_airmass()
        print('computed air mass.')
        self.compute_linke_turbidity()
        print('computed linke turbidity.')
        self.compute_clear_sky_irradiance()
        print('computed clear sky irradiance.')
        self.compute_dni_var_class()
        print('computed dni var classes.')

    def compute_solar_position(self, method='nrel_numpy', apparent_elevation=True):
        """Computes solar position using pvlib."""
        solar_pos = pvlib.solarposition.get_solarposition(time=self.data.index,
                                                          latitude=self.latitude,
                                                          longitude=self.longitude,
                                                          altitude=self.altitude,
                                                          method=method)
        self.data['sun_el'] = solar_pos['apparent_elevation'] if apparent_elevation else solar_pos['elevation']
        self.data['sun_az'] = solar_pos['azimuth']
        return self.data[['sun_el', 'sun_az']]

    def compute_airmass(self, method_airmass='kastenyoung1989', fill_default_pressure=True):
        """Computes airmass using pvlib."""
        p_amb = self.get_p_amb(fill_default_pressure)
        solar_pos = self.get_solar_pos()
        zenith = 90 - solar_pos['sun_el']
        airmass_rel = pvlib.atmosphere.get_relative_airmass(zenith, model=method_airmass)
        airmass_abs = airmass_rel * p_amb / pvlib.atmosphere.alt2pres(0)
        self.data['airmass_abs'] = airmass_abs
        return self.data['airmass_abs']

    def compute_linke_turbidity(self, method_linke_turbidity='mcclear', solar_constant=1361.1, limit_sun_elevation=20,
                                limit_dni=300, variability_tl_lim=0.01, tl_cloud_limit=8):
        """Computes linke turbidity using pvlib."""
        if method_linke_turbidity == 'mcclear':
            linke_turbidity = pvlib.clearsky.lookup_linke_turbidity(self.data.index, self.latitude, self.longitude,
                                                                    interp_turbidity=True)
        elif method_linke_turbidity == 'ineichen_dlr':
            linke_turbidity = linke_turbidity_ineichen_dlr(
                dni=self.data.dni,
                altitude=self.altitude,
                sun_el=self.data.sun_el,
                airmass_abs=self.data.airmass_abs,
                solar_constant=solar_constant,
                limit_sun_elevation=limit_sun_elevation,
                limit_dni=limit_dni,
                variability_tl_lim=variability_tl_lim,
                tl_cloud_limit=tl_cloud_limit
            )
        else:
           raise NotImplementedError(f'Method {method_linke_turbidity} for computing linke turbidity not implemented')
        self.data['linke_turbidity'] = linke_turbidity
        return linke_turbidity

    def compute_clear_sky_irradiance(self, method_clear_sky_irradiance='ineichen', method_linke_turbidity='mcclear'):
        """Computes clear sky irradiance using pvlib."""
        solar_pos = self.get_solar_pos()
        zenith = 90 - solar_pos['sun_el']
        airmass_abs = self.get_airmass(absolute=True)
        linke_turbidity = self.get_linke_turbidity(method_linke_turbidity=method_linke_turbidity)
        if method_clear_sky_irradiance == 'ineichen':
            df_clear_sky = pvlib.clearsky.ineichen(apparent_zenith=zenith, altitude=self.altitude,
                                                   airmass_absolute=airmass_abs, linke_turbidity=linke_turbidity)
        else:
            raise NotImplementedError(f'Method {method_linke_turbidity} not implemented.')
        self.data['clear_sky_ghi'] = df_clear_sky['ghi']
        self.data['clear_sky_dni'] = df_clear_sky['dni']
        self.data['clear_sky_dhi'] = df_clear_sky['dhi']

        self.data['clear_sky_ghi'] = self.data['clear_sky_ghi'].mask(self.data['ghi']/self.data['clear_sky_ghi']
                                                                      > 1.4, self.data['ghi'])
        self.data['clear_sky_dni'] = self.data['clear_sky_dni'].mask(self.data.dni > self.data['clear_sky_dni'],
                                                                     self.data.dni)
        self.data['clear_sky_dhi'] = (self.data['clear_sky_ghi'] - self.data['clear_sky_dni'] *
                                      np.cos(np.radians((90 - self.data.sun_el) % 360)))

        return self.data[['clear_sky_ghi', 'clear_sky_dni', 'clear_sky_dhi']]

    def compute_dni_var_class(self, time_period='15T'):
        """
        DNI variability procedure for time series in 1 min resolution  according to: Schroedter-Homscheidt et al.,
        Classifying ground-measured 1 minute temporal variability within hourly intervals for direct normal irradiances,
        Meteorologische Zeitschrift, (2018)
        :param time_period: used for classification
        :return: self.data['dni_var_class']: (pd series)
        """
        self.data['dni_var_class'] = compute_dni_variability_classes(
            dni=self.data.dni,
            clear_sky_dni=self.data.clear_sky_dni,
            time_period=time_period
        )
        return self.data['dni_var_class']

    def interpolate(self, limit, columns=None, method='index', limit_area='inside', limit_direction='both',
                    inplace=False):
        """Interpolates missing data in dataframe."""
        data = self.data if inplace else self.data.copy()
        if columns is None:
            data.interpolate(method=method, limit=limit, limit_area=limit_area, limit_direction=limit_direction,
                             inplace=True)
        else:
            for col in columns:
                data.loc[col] = data.loc[col].interpolate(method=method, limit=limit, limit_area=limit_area,
                                                          limit_direction=limit_direction)
        if 'dni_var_class' in data:
            data.loc[:, 'dni_var_class'] = np.round(data.loc[:, 'dni_var_class'])
        if not inplace:
            return data

    def filter_by_sun_el(self, min_sun_el, inplace=False):
        """Filters data by sun elevation."""
        data = self.data[self.data['sun_el'] > min_sun_el]
        if inplace:
            self.data = data
        else:
            return data

    def get_temporal_resolution(self):
        """Gets temporal resolution of dataframe"""
        if self.temporal_resolution is None:
            if isinstance(self.data, pd.DataFrame) and len(self.data) > 1:
                # Identify most frequent time between samples, round to full seconds before calculating mode value
                mode_freq = get_temporal_resolution_from_timeseries(self.data)
                # Express temporal resolution in seconds
                temp_res = f'{mode_freq.total_seconds():.0f}s'
            else:
                temp_res = 'undefined'
            self.temporal_resolution = temp_res
        return self.temporal_resolution

    def get_number_of_nans(self):
        return get_number_of_nans(self.data)

    def plot_distributions(self, columns=None, n_rows=2, df_ref=None, figsize=(20, 12)):
        if columns is None:
            columns = list(self.data.columns)
        plot_data_distributions(self.data, columns=columns, n_rows=n_rows, df_ref=df_ref, figsize=figsize)

    def reduce_temporal_resolution(self, new_freq, min_fraction=60 / 60):
        """
        Averages meteo data to lower time resolution rejecting all low-resolution periods with incomplete data.

        :param new_freq: (str, datetime.timedelta, or DateOffset), new temporal resolution, to be understood as
                   "freq" by pandas
        :param min_fraction: (float) Minimum fraction of valid readings required to retain average of a period.
        """
        self.data = aggregate_temporal_resolution(self.data, new_freq, self.temporal_resolution,
                                                  min_fraction=min_fraction)
        self.temporal_resolution = new_freq

    def check_daily_index_continuity(self):
        return check_daily_index_continuity(self.data.index)

    @classmethod
    def from_csv(cls, csv_file, **kwargs):
        """Initialize MeteoData object from csv file."""
        df_csv = pd.read_csv(csv_file, parse_dates=['timestamp'], index_col='timestamp')
        tz = df_csv.index.tz
        mode_freq = get_temporal_resolution_from_timeseries(df_csv)
        # Express temporal resolution in seconds
        temp_res = f'{mode_freq.total_seconds():.0f}s'
        return cls(data=df_csv, tz=tz, temporal_resolution=temp_res, **kwargs)

    @classmethod
    def from_mesor_txt(cls, txt_file):
        """Initialize MeteoData object from mesor txt file."""
        df_csv, meta_data = read_mesor_txt_file(txt_file)
        df_csv.rename(columns=map_data_columns(df_csv.columns), inplace=True)
        mode_freq = get_temporal_resolution_from_timeseries(df_csv)
        temp_res = f'{mode_freq.total_seconds():.0f}s'
        return cls(data=df_csv, temporal_resolution=temp_res, **meta_data)


class MesorMat(MeteoData):
    """Class for creating MeteoData objects from Mesor files."""

    def __init__(self, mesor_files: list, remove_duplicates=True, insert_missing_timestamps=True, min_sun_el=0,
                 interpolate=True, interpolate_limit=10, temporal_resolution=None, mesor_nan=-9999,
                 compute_solar_position=True, compute_clear_sky_irradiance=True, compute_dni_var_class=False,
                 time_period_dni_var_class='15T',tl_files=None, dni_var_class_files=None,
                 method_clear_sky_irradiance='ineichen', method_linke_turbidity='mcclear',
                 file_type_dni_var_class='.mat'):

        assert type(mesor_files) is list, 'First argument to MesorMat must be a list of strings (paths to mat files)'

        super(MesorMat, self).__init__(self, temporal_resolution=temporal_resolution)
        self.mesor_files = mesor_files
        for i, mat_file in enumerate(mesor_files):
            # load mesor data from mat file
            mesor_dict = load_matlab_mesor_file(mat_file, rename_params=True)
            if i == 0:
                self.latitude = mesor_dict[PYTHON_LAT]
                self.longitude = mesor_dict[PYTHON_LON]
                self.altitude = mesor_dict[PYTHON_ALT]
                self.tz = mesor_dict[PYTHON_TZ]
            else:
                assert self.latitude == mesor_dict[PYTHON_LAT], 'Inconsistent location'
                assert self.longitude == mesor_dict[PYTHON_LON], 'Inconsistent location'
                assert self.altitude == mesor_dict[PYTHON_ALT], 'Inconsistent location'
                assert self.tz == mesor_dict[PYTHON_TZ], 'Inconsistent timezone information'
            df_tmp = pd.DataFrame(mesor_dict).set_index(PYTHON_DT)
            df_tmp = df_tmp[~df_tmp.index.isna()]
            df_tmp_aligned = df_tmp.reindex(columns=self.data.columns)
            if self.data.empty:
                self.data = df_tmp_aligned
            else:
                self.data = pd.concat([self.data, df_tmp_aligned], axis=0)
        if remove_duplicates:
            self.remove_duplicates(inplace=True)
        if self.temporal_resolution is None:
            self.get_temporal_resolution()
        if insert_missing_timestamps:
            self.insert_missing_timestamps(inplace=True)
        # Convert invalid mesor values (e.g., -9999) to nan
        if mesor_nan is not None:
            convert_invalid_values_to_nan(self.data, mesor_nan, inplace=True)
        # Convert pressure from hPa to Pa
        if PYTHON_PAMB in self.data.columns:
            convert_invalid_values_to_nan(self.data, invalid_value=930, condition='min', columns=['p_amb'], inplace=True)
            self.data.loc[:, PYTHON_PAMB] = self.data[PYTHON_PAMB] * 100
        if compute_solar_position:
            self.compute_solar_position()
        if tl_files is not None:
            self.add_linke_turbidity(tl_files)
        if compute_clear_sky_irradiance:
            self.compute_clear_sky_irradiance(method_clear_sky_irradiance=method_clear_sky_irradiance,
                                              method_linke_turbidity=method_linke_turbidity)
        if dni_var_class_files is not None:
            self.add_dni_variability_classes(dni_var_class_files, file_type_dni_var_class=file_type_dni_var_class)
        if compute_dni_var_class:
            self.compute_dni_var_class(time_period=time_period_dni_var_class)
        if interpolate:
            self.interpolate(limit=interpolate_limit, inplace=True)
        if compute_solar_position:
            self.filter_by_sun_el(min_sun_el=min_sun_el, inplace=True)

    def add_linke_turbidity(self, tl_files):
        """Adds linke turbidity from mat files to data."""
        series_tl = load_matlab_linke_turbidity_per_day(tl_files=tl_files).tz_convert(self.data.index.tzinfo)
        self.data['linke_turbidity'] = pd.merge_asof(pd.DataFrame(index=self.data.index), series_tl, left_index=True,
                                                     right_index=True)

    def add_dni_variability_classes(self, dnivc_files, file_type_dni_var_class='.mat'):
        """Adds dni variability classes from mat files to data."""
        if file_type_dni_var_class == '.mat':
            dni_var_class = load_matlab_dni_classes(dnivc_files, convert_timezone=self.tz, rename_params=True)
        elif file_type_dni_var_class == '.csv':
            dni_var_class = pd.read_csv(dnivc_files[0], index_col=0).squeeze("columns")
            dni_var_class.index = pd.to_datetime(dni_var_class.index)
            dni_var_class = dni_var_class.to_frame(dni_var_class.name)
        else:
            raise ValueError('Only .mat and .csv files are supported.')

        self.data['dni_var_class'] = pd.merge(pd.DataFrame(index=self.data.index), dni_var_class, left_index=True,
                                              right_index=True)


class CSOnline(MeteoData):
    """ Class for retrieving real time meteo data from CS logger """
    def __init__(self, url_cs_logger_table: str, latitude, longitude, altitude, timezone,
                 name_desired_columns_cs_table=None):
        """
        This method initializes an instance of the CSOnline class, which is used to retrieve
        data from a CS logger table.

        :param url_cs_logger_table: (str) The URL of the CS logger table.
        :param latitude: (float) The latitude of the location.
        :param longitude: (float) The longitude of the location.
        :param altitude: (float) The altitude of the location.
        :param timezone: (str) The timezone of the location (e.g. "GMT+1").
        :param name_desired_columns_cs_table: (dict, optional) A string specifying the desired columns from the CS table.
        """
        assert type(url_cs_logger_table) is str, 'First argument to CSOnline must be a str (URL of desired CS table)'

        super(CSOnline, self).__init__(self)
        self.url_cs_logger_table = url_cs_logger_table
        # get CS online data
        self.latitude = latitude
        self.longitude = longitude
        self.altitude = altitude
        self.tz = timezone
        self.name_desired_columns_cs_table = name_desired_columns_cs_table

    def update_meteo_data_real_time(self, compute_solar_position=True, compute_clear_sky_irradiance=True):
        """
        This method updates the meteorological data stored in the object in real-time by fetching
        data from a CS logger table and optionally computing solar position and clear sky irradiance.

        :param compute_solar_position: (bool, optional) Whether to compute solar position data (default is True)
        :param compute_clear_sky_irradiance: (bool, optional) Whether to compute clear sky irradiance (default is True)
        :return:
        """
        cs_data_frame = real_time_utility_functions.parse_logger_data(self.url_cs_logger_table, self.tz,
                                                                      name_desired_columns_cs_table=
                                                                      self.name_desired_columns_cs_table)
        if self.data.empty:
            # This is needed when self.data is still empty in order to avoid this FutureWarning: The behavior of array
            # concatenation with empty entries is deprecated. In a future version, this will no longer exclude empty
            # items when determining the result dtype. To retain the old behavior, exclude the empty entries before the
            # concat operation.
            common_columns = self.data.columns.intersection(cs_data_frame.columns)
            self.data = self.data.drop(columns=common_columns)
            self.data = pd.concat([self.data, cs_data_frame], axis=1)
            self.data.index = pd.to_datetime(self.data.index)
        else:
            self.data = pd.concat([self.data, cs_data_frame], axis=0)

        # get rid of older data
        num_rows = self.data.shape[0]
        if num_rows > 1:
            self.data = self.data.tail(1)
        # Convert pressure from hPa to Pa
        if compute_solar_position:
            self.compute_solar_position()
        if compute_clear_sky_irradiance:
            self.compute_clear_sky_irradiance()


def linke_turbidity_ineichen_dlr(dni, altitude, sun_el, airmass_abs, solar_constant=1361.1, limit_sun_elevation=20,
                                 limit_dni=300, variability_tl_lim=0.01, tl_cloud_limit=8):
    """
    Computes the Linke turbidity factor using the Ineichen DLR method.
    This method estimates the Linke turbidity, which describes atmospheric turbidity based on DNI
    and atmospheric conditions, and applies various filters and adjustments to derive a daily mean turbidity.

    :param dni: (pd.Series) Direct normal irradiance (DNI) values, indexed by datetime.
    :param altitude: (float) Altitude of the location in meters above sea level.
    :param sun_el: (pd.Series) Solar elevation angle in degrees, indexed by datetime.
    :param airmass_abs: (pd.Series) Absolute airmass values, indexed by datetime.
    :param solar_constant: (float) Solar constant in W/m^2
    :param limit_sun_elevation: (float) Minimum sun elevation angle in degrees for filtering
    :param limit_dni: (float) Minimum DNI threshold for filtering
    :param variability_tl_lim: (float) Maximum allowed variability in turbidity factor
    :param tl_cloud_limit: (float) Maximum Linke turbidity threshold for clear-sky filtering
    :return: linke_turbidity: (pd.Series) Processed and filtered Linke turbidity, indexed by datetime.
    """
    dt = dni.index
    _, earth_sun_corr_factor, _ = compute_earth_sun_distance_correction_factor(dt.dayofyear)
    fh1 = np.exp(-altitude / 8000)
    bcoef = 0.664 + 0.163 / fh1
    linke_turbidity = ((1 / 0.09 * np.log(bcoef * solar_constant * earth_sun_corr_factor / dni)) /
                       airmass_abs + 1)
    linke_turbidity[linke_turbidity < 2] = (linke_turbidity[linke_turbidity < 2]
                                            - 0.25 * np.sqrt(2 - linke_turbidity[linke_turbidity < 2]))
    linke_turbidity = linke_turbidity.replace(np.inf, np.nan)
    filter_sun_elevation = sun_el >= limit_sun_elevation
    filter_low_DNI = dni >= limit_dni
    diff_linke_turbidity = np.diff(linke_turbidity, prepend=0)
    time_gradient = np.gradient((dt - dt[0]).total_seconds()) / 86400

    varib_TL = (np.abs(diff_linke_turbidity) + np.abs(np.roll(diff_linke_turbidity, -1))) / (2 * time_gradient)
    varib_TL[np.isnan(varib_TL)] = 0
    filter_varib_TL = (np.abs(varib_TL) / (60 * 24)) <= variability_tl_lim
    filter_TL_high = linke_turbidity <= tl_cloud_limit
    combined_filter = filter_sun_elevation & filter_low_DNI & filter_varib_TL & filter_TL_high

    filtered_turbidity = linke_turbidity[combined_filter]

    daily_mean_turbidity = filtered_turbidity.resample('D').mean()
    daily_mean_turbidity = daily_mean_turbidity.interpolate(method='linear').fillna(method='bfill')
    linke_turbidity = daily_mean_turbidity.reindex(linke_turbidity.index, method='ffill')
    return linke_turbidity


def compute_dni_variability_classes(dni, clear_sky_dni, time_period='15T'):
    """
    DNI variability procedure for time series in 1 min resolution  according to: Schroedter-Homscheidt et al.,
    Classifying ground-measured 1 minute temporal variability within hourly intervals for direct normal irradiances,
    Meteorologische Zeitschrift, (2018)
    :param dni: (pd.Series) Series of dni values with pd.DatetimeIndex.
    :param clear_sky_dni: (pd.Series) Series of clear-sky dni values with pd.DatetimeIndex.
    :param time_period: (str) Time period used for classification.
    :return: dni_var_class: (pd series) Series of dni variability classes with pd.DatetimeIndex
    """
    print("Start compute DNI variability classification")
    # normalized index according to: Schroedter-Homscheidt et al. (2018)
    normalized_index_k_dni_mean = 60.0
    normalized_index_k_dni_diff_mean = 111.0
    normalized_index_k_dni_diff_std = 146.0
    normalized_index_k_dni_diff_max = 32.0
    normalized_index_dni_diff_mean = 0.14
    normalized_index_dni_diff_std = 0.18
    normalized_index_dni_diff_max = 0.04
    normalized_index_UML = 0.12
    normalized_index_UMC = 0.07
    normalized_index_Vi = 0.16
    normalized_index_V = 118.0
    normalized_index_CSFD = 1.25
    normalized_index_LMA = 0.11
    # normalization factor is defined for CSFD and LMA in one hour
    if time_period != '60T':
        normalized_index_CSFD = normalized_index_CSFD / (float(time_period[:-1])/60.0)
        normalized_index_LMA = normalized_index_LMA / (float(time_period[:-1])/60.0)
    # Median values reference data according to: Schroedter-Homscheidt et al. (2018)
    median_mean_Kc = [59.716450, 57.303360, 55.608116, 47.613976, 45.294029, 28.660505, 10.690928, 0.071842760]
    median_group_low = [0.17903249, 1.5642396, 7.2099032, 18.405214, 3.5177464, 17.464592, 6.6749816, 0.12353657]
    median_group_med = [0.16895518, 2.4010856, 15.225690, 27.722343, 4.1754084, 24.160376, 9.7200699, 0.15893045]
    median_CSFD = [0, 0, 10, 18.75, 3.75, 18.75, 8.75, 0]
    median_UMC = [-0.41167247, -2.3392999, -2.4438744, -5.8918180, -14.473125, -22.455175, -43.247498, -52.265186]
    median_LMA = [93.260612, 82.621170, 78.697609, 59.032288, 64.499847, 28.969118, 10.521536, 0.064277329]
    rel_diff_clear_sky_dni = 0.15

    # calculate DNI variability indices
    dni_diff = dni.diff()
    dni_clear_sky_threshold = clear_sky_dni * rel_diff_clear_sky_dni

    significant_changes = (np.sign(dni_diff).diff().abs() > 0) & (dni_diff.abs() > dni_clear_sky_threshold)

    k_dni = dni / clear_sky_dni
    k_dni_diff = k_dni.diff()
    clear_sky_dni_diff = clear_sky_dni.diff()

    upper_envelope, lower_envelope = time_dependent_envelope_curve(dni)

    upper_envelop_sliding_int = (upper_envelope.rolling(window=time_period, on=upper_envelope.index, min_periods=1).
                         apply(trapezoidal_integral_over_time_window))
    lower_envelop_sliding_int = (lower_envelope.rolling(window=time_period, on=lower_envelope.index, min_periods=1).
                         apply(trapezoidal_integral_over_time_window))
    clear_sky_dni_sliding_int = (clear_sky_dni.rolling(window=time_period, on=clear_sky_dni.index, min_periods=1).
                         apply(trapezoidal_integral_over_time_window))

    UML = upper_envelop_sliding_int - lower_envelop_sliding_int
    UMC = upper_envelop_sliding_int - clear_sky_dni_sliding_int
    LMA = lower_envelop_sliding_int
    UML = UML[~UML.index.duplicated(keep='first')]
    UMC = UMC[~UMC.index.duplicated(keep='first')]
    LMA = LMA[~LMA.index.duplicated(keep='first')]

    variability_index_V = variability_index(k_dni_diff, time_period)
    variability_index_VI = variability_index_indicator(dni_diff, clear_sky_dni_diff, time_period)

    # normalize DNI variability indices
    k_dni_sliding_mean = k_dni.rolling(window=time_period, min_periods=1).mean() * normalized_index_k_dni_mean
    k_dni_diff_sliding_mean = (np.abs(k_dni_diff)).rolling(window=time_period, min_periods=1).mean() * normalized_index_k_dni_diff_mean
    k_dni_diff_sliding_std = (np.abs(k_dni_diff)).rolling(window=time_period, min_periods=1).std() * normalized_index_k_dni_diff_std
    k_dni_diff_sliding_max = (np.abs(k_dni_diff)).rolling(window=time_period, min_periods=1).max() * normalized_index_k_dni_diff_max
    dni_diff_sliding_mean = (np.abs(dni_diff)).rolling(window=time_period, min_periods=1).mean() * normalized_index_dni_diff_mean
    dni_diff_sliding_std = (np.abs(dni_diff)).rolling(window=time_period, min_periods=1).std() * normalized_index_dni_diff_std
    dni_diff_sliding_max = (np.abs(dni_diff)).rolling(window=time_period, min_periods=1).max() * normalized_index_dni_diff_max
    UML = UML * normalized_index_UML
    UMC = UMC * normalized_index_UMC
    variability_index_V = variability_index_V * normalized_index_V
    variability_index_VI = variability_index_VI * normalized_index_Vi
    CSFD = significant_changes.rolling(window=time_period, min_periods=1).sum() * normalized_index_CSFD
    LMA = LMA * normalized_index_LMA
    # group some indices according to Schroedter-Homscheidt et al., (2018)
    group_low = (pd.concat([k_dni_diff_sliding_mean, dni_diff_sliding_mean, UML, variability_index_VI],
                            axis=1)).mean(axis=1)

    group_med = (pd.concat([k_dni_diff_sliding_std, k_dni_diff_sliding_max, dni_diff_sliding_std,
                             dni_diff_sliding_max, variability_index_V], axis=1)).mean(axis=1)
    # Calculate the absolute difference between the observed indices and the predefined median values from
    # Schroedter-Homscheidt et al. (2018).
    diff_classes_k_dni_sliding_mean = pd.DataFrame({f'var_class_{i + 1}': np.abs(k_dni_sliding_mean - val)
                                                    for i, val in enumerate(median_mean_Kc)})
    diff_classes_group_low = pd.DataFrame({f'var_class_{i + 1}': np.abs(group_low - val)
                                                    for i, val in enumerate(median_group_low)})
    diff_classes_group_med = pd.DataFrame({f'var_class_{i + 1}': np.abs(group_med - val)
                                                    for i, val in enumerate(median_group_med)})
    diff_classes_CSFD = pd.DataFrame({f'var_class_{i + 1}': np.abs(CSFD - val)
                                                    for i, val in enumerate(median_CSFD)})
    diff_classes_UMC = pd.DataFrame({f'var_class_{i + 1}': np.abs(UMC - val)
                                                    for i, val in enumerate(median_UMC)})
    diff_classes_LMA = pd.DataFrame({f'var_class_{i + 1}': np.abs(LMA - val)
                                                    for i, val in enumerate(median_LMA)})
    # Calculate the sum of all absolute differences.
    all_diff_classes = sum([diff_classes_k_dni_sliding_mean, diff_classes_group_low, diff_classes_group_med,
                        diff_classes_CSFD, diff_classes_UMC, diff_classes_LMA])
    # Identify the class with the lowest sum difference between the observed indices and the predefined median
    # values from Schroedter-Homscheidt et al. (2018).
    min_column = all_diff_classes.idxmin(axis=1)
    column_map = {f'var_class_{i + 1}': i + 1 for i in range(8)}
    dni_var_class = min_column.map(column_map)

    return dni_var_class


def aggregate_temporal_resolution(meteo_data, new_freq, old_freq, min_fraction=60 / 60):
    """
    Averages meteo data to lower time resolution rejecting all low-resolution periods with incomplete data.

    Based on pyranocam/validation_thesis -- data_post_preprocessing.generate_dif_time_res

    :param meteo_data: (pd DataFrame) define the dataframe, which contains the data in a specific time frequency
               (defined by 'new_freq'-argument).
    :param new_freq: (str, datetime.timedelta, or DateOffset), new temporal resolution, to be understood as "freq" by
                pandas
    :param old_freq: (str, datetime.timedelta, or DateOffset), old temporal resolution, to be understood as "freq" by
                pandas
    :param min_fraction: (float) Minimum fraction of valid readings required to retain average of a period.
    """

    # Average, ignore minutes which contain NaN
    time_res_df = meteo_data.groupby(pd.Grouper(freq=new_freq, closed='right', label='right')).mean()
    # Count samples per low-res period, exclude columns which only contain NaN in all rows
    time_res_count_df = meteo_data.dropna(axis=1, how='all').groupby(pd.Grouper(freq=new_freq, closed='right',
                                                                                label='right')).count()
    # Retain only periods with as many valid sample points as expected
    expected_readings_per_period = pd.to_timedelta(new_freq) / pd.to_timedelta(old_freq)
    return time_res_df[(time_res_count_df >= min_fraction * expected_readings_per_period).all(axis=1)]


def compute_earth_sun_distance_correction_factor(day_of_year):
    """ Calculate distance between sun and earth acoording to Liou, An Introduction to Atmospheric Radiation,
    2002, p.49
    :param day_of_year: (int, np.ndarray) (vector of) day(s) of year as Integer [1,366]
    :return:
        - earth_sun_distance: Earth to Sun distance in meters
        - earth_sun_corr_factor: Correction factor for Earth to Sun distance depending on the day of the year.
        - solar_disk_radius: Angular radius of solar disk
    """
    r0 = 1.49598e11  # Mean Earth-Sun distance in meters
    as_radius = 6.96000e8  # Sun's mean radius in meters
    # Coefficients from Liou (identical to Iqbal)
    an = [1.000110, 0.034221, 0.000719]
    bn = [0, 0.001280, 0.000077]
    e = 0.017  # Orbital eccentricity
    a = r0 / (1 - e ** 2)
    day_of_year = np.asarray(day_of_year)
    t = (day_of_year - 1) * 2 * np.pi / 365
    sum_vector = (
        an[0] * np.cos(0 * t) + bn[0] * np.sin(0 * t) +
        an[1] * np.cos(1 * t) + bn[1] * np.sin(1 * t) +
        an[2] * np.cos(2 * t) + bn[2] * np.sin(2 * t))
    earth_sun_distance = a / np.sqrt(sum_vector)   # Earth-Sun distance
    earth_sun_corr_factor = (earth_sun_distance / r0) ** 2   # Earth-Sun Distance Correction Factor
    solar_disk_radius = np.degrees(np.arctan(as_radius / earth_sun_distance))   # solar disk angular radius
    return earth_sun_distance, earth_sun_corr_factor, solar_disk_radius


def time_dependent_envelope_curve(dni_series):
    """
    Get upper and lower DNI envelope curves according toSchroedter-Homscheidt et al. 2018
    :param dni_series: (pd series) dni with datetimeindex in 1 min resolution
    :return: upper_envelope: (pd series) upper envelope curve fitting to dni_series
    :return: upper_envelope: (pd series) lower envelope curve fitting to dni_series
    """
    print("Start creating evelope curves for DNI variability classification.")

    upper_envelope = []
    lower_envelope = []
    unique_days = dni_series.index.normalize().unique()
    total_days = len(unique_days)
    count = 0
    for day in tqdm(unique_days, total=total_days):
        count += 1
        day_start = day
        day_end = day + pd.Timedelta(days=1)
        daily_dni = dni_series[day_start:day_end]

        local_max = argrelextrema(daily_dni.values, np.greater)[0]  # Indices of local maxima
        local_min = argrelextrema(daily_dni.values, np.less)[0]  # Indices of local minima

        local_max_series = pd.Series(np.nan, index=daily_dni.index)
        local_min_series = pd.Series(np.nan, index=daily_dni.index)
        local_max_series.iloc[local_max] = daily_dni.iloc[local_max]
        local_min_series.iloc[local_min] = daily_dni.iloc[local_min]

        # get valid extrema according to Schroedter-Homscheidt et al. 2018
        max_series = find_valid_extrema(daily_dni, local_max_series, extrema_type='maxima')
        min_series = find_valid_extrema(daily_dni, local_min_series, extrema_type='minima')

        # create envelope curve
        upper_envelope_day = pd.Series(np.nan, index=daily_dni.index)
        lower_envelope_day = pd.Series(np.nan, index=daily_dni.index)

        upper_envelope_day.loc[max_series.index] = max_series.values
        lower_envelope_day.loc[min_series.index] = min_series.values

        upper_envelope_day.iloc[0] = daily_dni.iloc[0]
        upper_envelope_day.iloc[-1] = daily_dni.iloc[-1]
        lower_envelope_day.iloc[0] = daily_dni.iloc[0]
        lower_envelope_day.iloc[-1] = daily_dni.iloc[-1]

        upper_envelope_day = upper_envelope_day.interpolate(method='linear', limit_direction='both')
        lower_envelope_day = lower_envelope_day.interpolate(method='linear', limit_direction='both')

        upper_envelope_day = upper_envelope_day.where(upper_envelope_day >= daily_dni, daily_dni)
        lower_envelope_day = lower_envelope_day.where(lower_envelope_day <= daily_dni, daily_dni)

        upper_envelope.append(upper_envelope_day)
        lower_envelope.append(lower_envelope_day)

    upper_envelope = pd.concat(upper_envelope)
    lower_envelope = pd.concat(lower_envelope)

    return upper_envelope, lower_envelope


def find_valid_extrema(dni_series, local_extrema_series, extrema_type='maxima', lower_time_limit=4, upper_time_limit=10):
    """
    Function to detect valid maxima or minima within a lower and upper time limit for a time window. This is according
    to the procedure described in Schroedter-Homscheidt et al. 2018
    :param dni_series: (pd series) dni with datetimeindex in 1 min resolution
    :param local_extrema_series: (pd series) The series with local extrema (maxima or minima) filled.
    :param extrema_type: (string) either 'maxima' or 'minima' to specify the type of extrema to detect.
    :return: extrema_series (pd series) valid extrema
    """
    start_idx = 0
    extrema_series = pd.Series(dtype='float64').tz_localize(dni_series.index.tz)
    while start_idx < len(dni_series):
        window_start = dni_series.index[start_idx]
        window_end = window_start + pd.Timedelta(minutes=lower_time_limit)
        window_end = dni_series.index[dni_series.index.get_indexer([window_end], method='nearest')[0]]
        while window_end <= dni_series.index[-1]:
            # Check for local extrema in the current window
            window_extrema = local_extrema_series[window_start:window_end].dropna()
            if not window_extrema.empty:
                # valid extrema exist
                if extrema_type == 'maxima':
                    extrema_idx = window_extrema.idxmax()
                    extrema_value= window_extrema.max()
                elif extrema_type == 'minima':
                    extrema_idx = window_extrema.idxmin()
                    extrema_value= window_extrema.min()
                extrema_series.at[extrema_idx] = extrema_value
                break
            elif (window_end - window_start) >= pd.Timedelta(minutes=upper_time_limit):
                # no extrema in upper time limit for time window --> probably clear sky conditions --> use dni_series
                extrema_idx = dni_series.index[dni_series.index.slice_indexer(start=window_start, end=window_end)]
                extrema_value = dni_series[extrema_idx].values
                new_extrema = pd.Series(extrema_value, index=extrema_idx)
                extrema_series = pd.concat([extrema_series, new_extrema])
                break
            else:
                # No extrema found, expand the window by 1 minute
                window_end += pd.Timedelta(minutes=1)
                if window_end > dni_series.index[-1]:
                    window_end = dni_series.index[-1]
                    break
                window_end = dni_series.index[dni_series.index.get_indexer([window_end], method='nearest')[0]]
        start_idx = dni_series.index.get_loc(window_end)
        if start_idx == len(dni_series) - 1:
            break
    return extrema_series


def trapezoidal_integral_over_time_window(window):
    """
    Function to compute the trapezoidal integral over a given time window. The time window must have
    a datetime index with 1-minute resolution.

    :param window: (pd.Series) A time series with a datetime index and corresponding values for integration.
    :return: (float) The computed integral value over the time window in hours, or NaN .
    """
    window = pd.Series(window)
    window = window.dropna()

    if len(window) > 1:
        time_diff = np.array((window.index[1:] - window.index[:-1]).total_seconds()/60)
        return np.trapz(window, dx=(time_diff/60)) # The integral expects time in units of hours, so we have to divide by 60.
    else:
        return np.nan


def variability_index(k_dni_diff, time_period):
    """
    Calculates the Variability Index (V) over a specified time period, according to Coimbra et al. (2013)
    This index provides insights on the variability of direct normal irradiance (DNI) compared to clear sky conditions.

    :param k_dni_diff: (pd.Series) The difference in clearness index (K) values over time.
    :param time_period: (str) A time-based window string (e.g., '15T') used for rolling window calculations.

    :return: v (pd.Series) variability index V: according to Coimbra et al. (2013)
    """
    v = (k_dni_diff ** 2).rolling(window=time_period, min_periods=1).mean().apply(np.sqrt)
    return v


def variability_index_indicator(dni_diff, clear_sky_dni_diff, time_period):
    """
    Calculates the Variability Index Indicator (VI) over a specified time period, according to Stein et al. (2012)
    This index provides insights on the variability of direct normal irradiance (DNI) compared to clear sky conditions.

    :param dni_diff: (pd.Series) The difference in DNI values over time.
    :param clear_sky_dni_diff: (pd.Series) The difference in clear-sky DNI values over time.
    :param time_period: (str) A time-based window string (e.g., '15T') used for rolling window calculations.

    :return: vi (pd.Series) variability index indicator VI: according to Stein et al. (2012)
    """
    dt = dni_diff.index.to_series().diff().dt.total_seconds() / 60

    vi = ((dni_diff ** 2 + dt ** 2).apply(np.sqrt).rolling(window=time_period, min_periods=1).sum() /
          (clear_sky_dni_diff ** 2 + dt ** 2).apply(np.sqrt).rolling(window=time_period, min_periods=1).sum())
    return vi


def plot_series(series):
    """
    Function to plot a time series with a datetime index. It generates a simple line plot of the series
     (helpful for debugging).
    :param series: (pd.Series) A time series with a datetime index.
    """
    series.plot()
    plt.title("Time Series Plot")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.show()


def check_daily_index_continuity(ts, temporal_res=None):
    """Check if index (pd.DatetimeIndex) has gaps."""
    if temporal_res is None:
        temporal_res = get_temporal_resolution_from_timeseries(pd.Series(0, index=ts))
    all_dates = np.unique(ts.date)
    num_missing_entries = pd.Series(0, index=all_dates)
    missing_entries = pd.DatetimeIndex([])
    for date in tqdm(all_dates):
        ts_d = ts[ts.date == date]
        dt_range_d = pd.date_range(start=ts_d[0], end=ts_d[-1], freq=temporal_res)
        missing_dts_d = dt_range_d[~dt_range_d.isin(ts_d)]
        num_missing_entries.loc[date] = len(missing_dts_d)
        missing_entries = missing_entries.union(missing_dts_d)
    return num_missing_entries, pd.DatetimeIndex(missing_entries)


def read_mesor_txt_file(file_path):
    """Read meteo data from mesor txt file."""
    meta_keys = {
        'location.latitude[degN]': 'latitude',
        'location.longitude[degE]': 'longitude',
        'location.altitude[m]': 'altitude',
        'location.timezone[h]': 'tz',
    }
    indicator_columnname = '#channel'
    indicator_datasection_start = '#begindata'
    indicator_datasection_end = '#enddata'
    meta_data = {}
    data_start = False
    column_labels = []
    data_lines = []

    # Open the file and parse line by line
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()

            # Extract meta-data before data starts
            if not data_start:
                # Look for key-value pairs in meta-data
                for key, new_key in meta_keys.items():
                    if line.startswith(f"#{key}:"):
                        if 'location' in key:
                            meta_data[new_key] = float(line.split(":", 1)[1].strip())
                        else:
                            meta_data[new_key] = line.split(":", 1)[1].strip()

                # Check for the start of the data section
                if line.startswith(indicator_columnname):
                    column_labels.append(line.split(" ")[1])
                elif line == indicator_datasection_start:
                    data_start = True

            # Collect data lines once data starts until the end of the data section
            elif data_start:
                if line == indicator_datasection_end:
                    break
                data_lines.append(line)
    if column_labels[0] == 'date' and column_labels[1] == 'time':
        column_labels[0] = 'timestamp'
        column_labels.pop(1)
    else:
        raise RuntimeError('Expected first two columns to be "date" and "time".')
    # Convert the data lines into a Pandas DataFrame
    data_string = "\n".join(data_lines)
    df_data = pd.read_csv(StringIO(data_string), sep="\t", names=column_labels)
    tz = datetime.timezone(datetime.timedelta(hours=meta_data['tz']))
    meta_data['tz'] = tz
    df_data['timestamp'] = pd.DatetimeIndex(df_data['timestamp'], tz=tz)
    df_data.set_index('timestamp', inplace=True)
    convert_invalid_values_to_nan(df_data, -9999, inplace=True)

    return df_data, meta_data


def convert_invalid_values_to_nan(data, invalid_value, condition='equals', columns=None, inplace=False):
    """Converts invalid values in dataframe to nans."""
    if columns is None:
        columns = data.columns
    if not inplace:
        data = data.copy()
    for col in columns:
        if condition == 'equals':
            data.loc[data[col] == invalid_value, col] = np.nan
        elif condition == 'min':
            data.loc[data[col] <= invalid_value, col] = np.nan
        elif condition == 'max':
            data.loc[data[col] >= invalid_value, col] = np.nan
        else:
            raise ValueError(f'Value for parameter condition ({condition}) not defined.')
    if not inplace:
        return data


def get_missing_timestamps(dt, temporal_resolution, tz, min_ele=None, latitude=None, longitude=None, altitude=None,
                           method_solar_pos='nrel_numpy', bounds_from_dt=False):
    """
    Identifies missing timestamps within a given datetime range based on a specified temporal resolution.

    :param dt: Array-like of existing timestamps (pandas DatetimeIndex or list of timestamps).
    :param temporal_resolution: String representing the desired time frequency (e.g., '30S' for 30 seconds).
    :param tz: Time zone information for the generated timestamps.
    :param min_ele: (Optional) Minimum solar elevation angle to filter timestamps. Default is None.
    :param latitude: (Optional) Latitude of the location for solar position calculations. Required if `min_ele` is set.
    :param longitude: (Optional) Longitude of the location for solar position calculations. Required if `min_ele` is set.
    :param altitude: (Optional) Altitude of the location for solar position calculations. Default is None.
    :param method_solar_pos: (Optional) Method used for solar position calculations (default: 'nrel_numpy').
    :param bounds_from_dt: (Optional) If True, uses the first and last timestamps from `dt` as the range bounds.
                           If False, expands the range to include the entire day (00:00:00 to 23:59:59).

    :return:
        - dt_missing (pandas DatetimeIndex): Timestamps that are missing from `dt` within the expected range.
        - dr (pandas DatetimeIndex): The full expected timestamp range.
    """
    if bounds_from_dt:
        start = dt[0]
        end = dt[-1]
    else:
        start = pd.to_datetime(dt[0].strftime("%Y-%m-%d 00:00:00"))
        end = pd.to_datetime(dt[-1].strftime("%Y-%m-%d 23:59:59"))
    dr = pd.date_range(start=start, end=end, freq=temporal_resolution, tz=tz)
    if min_ele is not None:
        solar_pos = pvlib.solarposition.get_solarposition(time=dr,
                                                          latitude=latitude,
                                                          longitude=longitude,
                                                          altitude=altitude,
                                                          method=method_solar_pos)
        dr = dr[solar_pos['apparent_elevation'] >= min_ele]
    dt_missing = dr.difference(dt)
    return dt_missing, dr
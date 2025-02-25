# Licensed under the Apache License: http://www.apache.org/licenses/LICENSE-2.0
# For details: https://github.com/DLR-SF/sky_imaging/blob/main/NOTICE.txt

"""
This module provides functions to convert data structures from matlab into python.
"""

import pandas as pd
import numpy as np
import scipy
import itertools
import os
import pytz
import pathlib
import logging
from h5py import File

from asi_core.constants import MATLAB_TZ, MATLAB_DATENUM, MATLAB_SDN, MATLAB_DNIVARCLASS, MATLAB_TIMEDELTA
from asi_core.constants import PYTHON_RENAMING, PYTHON_DT
from asi_core.camera import RadiometricImager
from asi_core.ocam import OcamModel


def map_data_columns(matlab_column_names):
    """
    Maps MATLAB-style column names to their corresponding Python-style names using a predefined renaming dictionary.

    :param matlab_column_names: List of column names from MATLAB data.
    :return: A dictionary mapping MATLAB column names to their Python-equivalent names.
    """
    col_map_dict = {}
    for col in matlab_column_names:
        if col in PYTHON_RENAMING.keys():
            col_map_dict[col] = PYTHON_RENAMING[col]
        elif col.lower() in PYTHON_RENAMING.keys():
            col_map_dict[col] = PYTHON_RENAMING[col.lower()]
        elif col.upper() in PYTHON_RENAMING.keys():
            col_map_dict[col] = PYTHON_RENAMING[col.upper()]
        else:
            logging.warning(f'No respective python name for {col} found.')
    return col_map_dict


def load_matlab_struct(mat_file, struct_name, return_df=False):
    """Loads matlab struct from mat-file and stores it to python dict or pd.DataFrame.

    :param mat_file: full path of mat-file.
    :param struct_name: name of struct variable in mat-file.
    :param return_df: if true, parameters are renamed to new format.
    :return: python dict or pd.DataFrame of data stored in struct variable of mat-file.
    """

    data_dict = scipy.io.loadmat(mat_file)
    struct_array = data_dict[struct_name]
    if len(struct_array.dtype) <= 1:
        return struct_array
    var_names = struct_array.dtype.names
    var_dict = dict()
    for var_name in var_names:
        if struct_array[var_name][0, 0].shape[0] > 1:
            var_dict[var_name] = struct_array[var_name][0, 0].reshape(-1)
        else:
            var_dict[var_name] = struct_array[var_name][0, 0].item()
    if return_df:
        return pd.DataFrame(var_dict)
    else:
        return var_dict


def get_mat_dict(mat_file_path, suffix=None):
    """
    Import Matlab structure (.mat file) into python dictionary using scipy loadmat.

    The method is very similar to load_matlab_struct and is deprecated.

    :param mat_file_path (string): Path to Matlab file
    :param suffix (string, optional): Add suffix to key names
    :return: Dictionary containing Matlab struture data
    """
    mat = scipy.io.loadmat(mat_file_path)
    var_name_mat = list(mat.keys())[-1]

    mdata = mat[var_name_mat]
    mtype = mdata.dtype
    ndata = {n: mdata[n][0, 0] for n in mtype.names}

    mat_dict = {}

    for i in list(ndata.keys()):
        mat_dict[i] = list(itertools.chain(*mat[var_name_mat][i][0][0]))

    if suffix is not None:
        mat_dict = {k + suffix: v for k, v in mat_dict.items()}

    return mat_dict


def load_matlab_mesor_file(mat_file, struct_name='MesorDat', param_tz=MATLAB_TZ, param_datenum=MATLAB_DATENUM,
                           convert_timezone=None, rename_params=False, drop_datenum=True):
    """Loads matlab mesor file. Function for loading irradiance measurements and other meteorological measurements from
     a mat-file containing single struct ('MesorDat') to python dict or pandas dataframe.

    :param mat_file: full path of mat-file.
    :param struct_name: name of struct variable in mat-file.
    :param param_tz: name of timezone parameter.
    :param param_datenum: name of matlab datenum parameter.
    :param convert_timezone: timezone value (+/- UTC) to convert date times.
    :param rename_params: if true, parameters are renamed to new format.
    :param drop_datenum: drop matlab datenum data.
    :return: python dict  of mesor data.
    """

    mesor_dict = load_matlab_struct(mat_file, struct_name)
    assert param_datenum in mesor_dict, f'datetime parameter {param_datenum} not found.'
    timezone = mesor_dict.get(param_tz, None)
    # convert matlab datnum into datetime
    timestamps = matlab_datenum_to_pandas_datetime(mesor_dict[param_datenum],
                                                   timezone=timezone,
                                                   convert_timezone=convert_timezone)
    if drop_datenum:
        mesor_dict.pop(param_datenum, None)
    mesor_dict[PYTHON_DT] = timestamps

    if rename_params:
        mesor_dict_renamed = {PYTHON_DT: timestamps}
        for param in mesor_dict:
            if param in PYTHON_RENAMING:
                mesor_dict_renamed[PYTHON_RENAMING[param]] = mesor_dict[param]
        mesor_dict = mesor_dict_renamed
    return mesor_dict


def load_matlab_linke_turbidity_per_day(tl_files, struct_name='tl_data', param_name_tl='tl_per_day',
                                        param_name_dt='SDN_per_day'):
    """Loads daily linke turbidity values from mat-files.

    :param tl_files: list of full paths of mat-files containing turbidity data.
    :param struct_name: name of struct variable in mat-files.
    :param param_name_tl: name of daily turbididty parameter.
    :param param_name_dt: name of datetime (datenum) parameter.
    :return: pd.Series of daily Linke turbidity values.

    """
    series_tl = pd.Series(dtype='float64', name='linke_turbidity')
    for tl_file in tl_files:
        tl_data = load_matlab_struct(tl_file, struct_name)
        tl_per_day = tl_data[param_name_tl]
        pd_datetime = matlab_datenum_to_pandas_datetime(tl_data[param_name_dt], timezone=pytz.timezone(tl_data['tz']))
        assert len(tl_data) != len(pd_datetime)
        tmp_tl = pd.Series(tl_per_day, index=pd_datetime, name='linke_turbidity')
        if series_tl.empty:
            series_tl = tmp_tl
        else:
            series_tl = pd.concat([series_tl, tmp_tl])
    series_tl.sort_index(inplace=True)
    return series_tl


def load_matlab_dni_classes(mat_files, struct_name='DNIClass', param_dni_class=MATLAB_DNIVARCLASS,
                            param_dt=MATLAB_SDN, param_tz=MATLAB_TZ, convert_timezone=None, rename_params=False):
    """Loads dni variability classes from mat-files. Precomupted DNI variability classes for each measurement timestamo
    (https://doi.org/10.1127/metz/2018/0875) are loaded from mat-files and concatenated into single pandas dataframe.

    :param mat_files: list of full paths of mat-files.
    :param struct_name: name of struct variable in mat-files.
    :param param_dni_class: name of dni variability class parameter.
    :param param_dt: name of datetime (datenum) parameter.
    :param param_tz: name of timezone parameter.
    :param convert_timezone: timezone value (+/- UTC) to convert date times.
    :param rename_params: if true, parameters are renamed to new format.
    :return: pd.DataFrame of DNI variabilty classes.
    """

    if rename_params and param_dni_class in PYTHON_RENAMING:
        column_name = PYTHON_RENAMING[param_dni_class]
    else:
        column_name = param_dni_class

    df = pd.DataFrame()
    for matfile in mat_files:
        data_dict = load_matlab_struct(matfile, struct_name=struct_name)
        if param_tz in data_dict.keys():
            timezone = data_dict[param_tz]
        else:
            timezone = None
        pd_datetime = matlab_datenum_to_pandas_datetime(data_dict[param_dt],
                                                        timezone=timezone,
                                                        convert_timezone=convert_timezone)
        df = pd.concat([df, pd.DataFrame({column_name: data_dict[param_dni_class]}, index=pd_datetime, dtype=int)],
                       axis=0)
    df = df[~df.index.duplicated(keep='first')]
    return df


def load_matlab_persistence_values(mat_files, norm_max=None, norm_min=None, stand_mean=None, stand_std=None,
                                   param_dt=MATLAB_SDN, persist_col='nowGHIPer', timezone=1, convert_timezone=None):
    """Loads matlab persistence values.

    :param mat_files: list of full paths of mat-files.
    :param norm_max: max value to normalize persistence values (optional).
    :param norm_min: min value to standardize persistence values (optional).
    :param stand_mean: mean value to standardize persistence values (optional).
    :param stand_std: std value to standardize persistence values (optional).
    :param param_dt: name of datetime (datenum) parameter.
    :param persist_col: name of persistence column.
    :param timezone: timezone as integer (+/- UTC).
    :param convert_timezone: timezone value (+/- UTC) to convert date times.
    :return: pd.DataFrame of persistence values.
    """

    df = pd.DataFrame()
    for mat_file in mat_files:
        # load persistence nowcasting data from mat file
        data_dict = scipy.io.loadmat(mat_file)
        timestamps = matlab_datenum_to_pandas_datetime(data_dict[param_dt].reshape(-1),
                                                       timezone=timezone,
                                                       convert_timezone=convert_timezone)
        df_persist = pd.DataFrame(data_dict[persist_col],
                                  columns=range(1, data_dict[persist_col].shape[1]+1),
                                  index=timestamps)

        df = pd.concat([df, df_persist], axis=0)
    if np.isscalar(norm_min) and np.isscalar(norm_max):
        norm_range = (norm_max - norm_min)
        df = (df - norm_min) / norm_range
    elif np.isscalar(stand_mean) and np.isscalar(stand_std):
        df = (df - stand_mean) / stand_std
    return df


def load_matlab_ceilometer_data(mat_file, timestamp_col='time', cloud_base_height_col='cloud_base_height',
                                timezone=pytz.UTC):
    """
    Loads matlab ceilometer values (only the 3 channels for the cloud base height measurement).

    :param mat_file: full path of mat file.
    :param timestamp_col: name of column with timestamps
    :param cloud_base_height_col: name of column with cloud base height measurements
    :param timezone as pytz timezone or as integer (+/- UTC).
    """
    data_dict = scipy.io.loadmat(mat_file)
    timestamps = matlab_datenum_to_pandas_datetime(data_dict[timestamp_col].squeeze(), timezone=timezone)
    df_cloud_base_height = pd.DataFrame(np.transpose(data_dict[cloud_base_height_col]),
                                        index=timestamps,
                                        columns=['channel_1', 'channel_2', 'channel_3'])
    return df_cloud_base_height


def pandas_datetime_to_matlab_datenum(date):
    """
    Converts a Python datetime object to a MATLAB datenum.

    :param date: Pandas datetime series to be converted.
    :return: Pandas series with MATLAB datenum floats.
    """
    return date.apply(lambda dt: dt.toordinal() + 366 +  (dt.hour + dt.minute / 60.0 + dt.second / 3600.0) / 24.0)


def matlab_datenum_to_pandas_datetime(datenum, round_to='s', timezone=None, convert_timezone=None):
    """Convert matlab datenum to pandas datetime.

    :param datenum: matlab datenum values.
    :param round_to: abbreviation to round values (default: 's' -> second).
    :param timezone: implicit timezone of inserted matlab datenum timestamps or pytz.timezone. If not provided, the returned timestamps
                     will be timezone unaware: UTC --> 0; CET --> +1; CEST --> +2
    :param convert_timezone: timestamps will be returned in this timezone if specified: UTC --> 0; CET --> +1;
                     CEST --> +2
    :return: pd.DatetimeIndex of timestamps.
    """

    if isinstance(datenum, pd.Series):
        datenum = datenum.values
    if not np.issubdtype(datenum.dtype, np.floating):
        datenum = datenum.astype(float)
    pd_datetime = pd.to_datetime(datenum - MATLAB_TIMEDELTA, unit='d')
    if round_to:
        pd_datetime = pd_datetime.round(round_to)
    if timezone is not None:
        if isinstance(timezone, pytz.BaseTzInfo):
            pd_datetime = pd_datetime.tz_localize(tz=timezone)
        else:
            pd_datetime = pd_datetime.tz_localize(tz=timezone * 3600)
        if convert_timezone is not None:
            if isinstance(convert_timezone, pytz.BaseTzInfo):
                pd_datetime = pd_datetime.tz_convert(tz=convert_timezone)
            else:
                pd_datetime = pd_datetime.tz_convert(tz=convert_timezone * 3600)
    return pd_datetime


class MatlabCamera(RadiometricImager):
    """
    Used calibrations etc. are initialized via MATLAB

    This class can be useful for testing.
    """

    def __init__(self, path_matlab_workspace):
        with File(path_matlab_workspace, 'r') as matlab_workspace:
            self.latitude, self.longitude, self.altitude = \
                np.squeeze(matlab_workspace['handles']['ConfigParams']['Camera']['Location'][()])
            self.name = ''.join(map(chr, np.squeeze(matlab_workspace['handles']['ConfigParams']['Camera']['Name']
                                                        [()])))
            self.img_path_base = ''.join(map(chr, np.squeeze(matlab_workspace['handles']['ConfigParams']
                                                                        ['Project']['Dir']['RawDataDevice'][()])))
            tz_read = matlab_workspace['handles']['ConfigParams']['Project']['Timezone'][()][0][0]
            self.img_timezone = pytz.FixedOffset(tz_read*60)
            self.color_temperature = \
                matlab_workspace['handles']['ConfigParams']['CameraProp']['ColorTemperature'][()][0][0]
            self.weighting_luminosity = np.squeeze(matlab_workspace['handles']['ConfigParams']['diffIrradCam']
                                                   ['cam_model']['weightingLuminosity'][()])
            if self.color_temperature == 1e4:
                self.beta_planck = [0.3835, 0.3324, 0.2841]
            else:
                Exception()
            print('Improve the following line.')

            self.img_path_structure = r'\\129.247.24.131' + self.img_path_base[9:] + \
                                      '/{timestamp:%Y}/{camera_name}/{timestamp:%m}/{timestamp:%d}/{timestamp:%H}/' \
                                      '{timestamp:%Y%m%d%H%M%S}_{exposure_time:d}.jpg'

            self.satVal = matlab_workspace['handles']['ConfigParams']['diffIrradCam']['cam_model']['satVal'][()][0][0]
            self.base_sensitivity = matlab_workspace['handles']['ConfigParams']['diffIrradCam']['calib']\
                                                    ['baseSensitivity'][()][0][0]
            self.rel_overest_with_DNI = matlab_workspace['handles']['ConfigParams']['diffIrradCam']['calib']\
                                                        ['relOverestimationWithDNI'][()][0][0]
            self.satur_cor = matlab_workspace['handles']['ConfigParams']['diffIrradCam']['calib']\
                                                        ['saturCor'][()][0][0]

            angle_matrix_azimuth = matlab_workspace['handles']['Data']['Camera']['AngleMatrix']['AZ'][()].T
            angle_matrix_elevation = matlab_workspace['handles']['Data']['Camera']['AngleMatrix']['ELE'][()].T

            self.external_orientation = \
                np.squeeze(matlab_workspace['handles']['ConfigParams']['Camera']['Orientation'][()])
            horizon_mask = matlab_workspace['handles']['ConfigParams']['CameraProp']['HorizonMask'][()][0][0]
            ocam_path = ''.join(map(chr, np.squeeze(matlab_workspace['handles']['ConfigParams']['Camera']
                                                    ['CalibDataPath'][()])))
            config_path = ''.join(map(chr, np.squeeze(matlab_workspace['handles']['ConfigParams']['Project']['Dir']
                                                      ['ConfigDataDir'][()])))
            if config_path == 'path_matlab_workspace':
                config_path = str(pathlib.Path(path_matlab_workspace).parent)
            mask_file_path = ''.join(map(chr, np.squeeze(matlab_workspace['handles']['ConfigParams']['Project']['Path']
                                                    ['CameraMask'][()])))
            if mask_file_path == 'path_matlab_workspace':
                mask_file_path = str(pathlib.Path(path_matlab_workspace).parent)
            mask_file_name = ''.join(map(chr, np.squeeze(matlab_workspace['handles']['ConfigParams']['Camera']['Mask']
                                                    ['File'][()])))

            self.rel_exp_tol = matlab_workspace['handles']['ConfigParams']['PreProcessImage']['General']['SelectIm']\
                                                           ['ExpTimeTolerance'][()][0][0]

        self.load_ocam(os.path.join(config_path, ocam_path))

        self.camera_mask = np.array(get_mat_dict(os.path.join(mask_file_path, mask_file_name))['BW'])\
            .reshape((self.ocam_model.height, self.ocam_model.width))

        self.azimuth_mask, self.elevation_mask = \
            self.get_azimuth_elevation(self.ocam_model, horizon_mask, self.external_orientation)

        self.rel_exp_tol = 0.1

    def load_ocam(self, ocam_path):
        """Load ocam model from mat-file."""
        ocam_dict = {'ss': np.nan * np.ones((4,)), 'xc': 0, 'yc': 0, 'c': np.nan, 'd': np.nan, 'e': np.nan,
                           'width': 100000, 'height': 100000, 'pol': np.nan * np.ones((13,))}

        try:
            ocam_file = File(ocam_path)
            ocam_get = lambda k: ocam_file['ocam_model'][k][()]
        except OSError:
            from scipy.io import loadmat
            ocam_file = loadmat(ocam_path)
            ocam_get = lambda k: ocam_file['ocam_model'][k][0][0]

        for k in ocam_dict.keys():
            if not hasattr(ocam_dict[k], '__len__'):
                ocam_dict[k] = ocam_get(k)[0][0]
            elif len(ocam_dict[k]) > 1:
                ocam_dict[k] = np.squeeze(ocam_get(k))
            else:
                Exception('ocam_model not initialized properly')

        self.ocam_model = OcamModel(ocam_dict['ss'], ocam_dict['pol'], ocam_dict['xc'] - 1, ocam_dict['yc'] - 1,
                                    ocam_dict['c'], ocam_dict['d'], ocam_dict['e'], int(ocam_dict['width']),
                                    int(ocam_dict['height']))
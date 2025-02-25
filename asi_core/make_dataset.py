# Licensed under the Apache License: http://www.apache.org/licenses/LICENSE-2.0
# For details: https://github.com/DLR-SF/sky_imaging/blob/main/NOTICE.txt

"""
This module provides functions to create ASI datasets, e.g., for machine learning applications.
"""

import numpy as np
import PIL
import pandas as pd
import logging
from datetime import datetime
from pathlib import Path
import pvlib.solarposition
from fastcore.parallel import parallel

from asi_core.basics import get_image_files
from asi_core.camera import load_camera_data, AllSkyImager
from asi_core.util.mobotix import get_mobotix_meta_data

Q25_LEFT_CENTRATION_THRESHOLD = 200
Q25_RIGHT_CENTRATION_THRESHOLD = 400


def load_asi_list(csv_files, asi_root=None, col_timestamp='timestamp', col_rel_path='rel_path',
                  col_filename='file_name'):
    """Loads list of ASI from csv files. If asi_root is passed, the absolute path of each ASI image is appended to the
    resulting dataframe.

    :param csv_files: list of csv files (full paths).
    :param asi_root: root directory of asi images.
    :param col_timestamp: column name of timestamps of ASI.
    :param col_rel_path: column name of relative path of ASI wrt root directory.
    :param col_filename: column name of asi file names.
    :return: dataframe of merged csv files.
    """

    df_asi = pd.DataFrame()
    for csv_file in csv_files:
        df_temp = pd.read_csv(csv_file)
        df_asi = pd.concat([df_asi, df_temp], axis=0, ignore_index=True)
    df_asi[col_timestamp] = pd.to_datetime(df_asi[col_timestamp])
    no_nans = ~df_asi[col_rel_path].isna()
    # workaround to change windows separator to posix separator
    if '\\' in df_asi[col_rel_path].iloc[0]:
        df_asi.loc[no_nans, col_rel_path] = df_asi.loc[~df_asi[col_rel_path].isna(), col_rel_path].apply(
            lambda x: x.replace('\\', '/'))
    # if filename is not part of relative path, include it to relative path and drop column of filename
    if col_filename in df_asi.columns:
        df_asi.loc[no_nans, col_rel_path] = df_asi.loc[no_nans, col_rel_path].apply(
            lambda x: Path(x)) / df_asi.loc[no_nans, col_filename]
        df_asi.drop(columns=col_filename, inplace=True)
    if asi_root is not None:
        df_asi['abs_path'] = df_asi[col_rel_path].apply(lambda x: Path(asi_root) / Path(x))
    return df_asi


def read_asi_meta_data(filename, is_mobotix=True, name_convention='dlr', tz='UTC+0100'):
    """Extracts meta data of an all-sky image from its name.

    :param filename: file path of all-sky image.
    :param name_convention: determines naming convention of all-sky images.
    :return: meta data as dict.
    """
    filename = Path(filename)
    meta_data = {}
    # read meta data from file if it is a mobotix camera
    if is_mobotix:
        meta_data = get_mobotix_meta_data(filename)
    # else read meta data from filename
    if not is_mobotix or len(meta_data) == 0:
        if name_convention == 'dlr':
            if int(filename.name[:4]) < 2000:
                dateformat = '%y%m%d%H%M%S'
                len_datestr = 12
            else:
                dateformat = '%Y%m%d%H%M%S'
                len_datestr = 14
            meta_data['timestamp'] = pd.to_datetime(
                datetime.strptime(filename.name[:len_datestr], dateformat)).tz_localize(tz=tz)
            meta_data['exposure_time'] = int(filename.name[len_datestr+1:-len(filename.suffix)])
        else:
            raise NotImplementedError(f'Timestamp cannot be extracted for file name convention {name_convention}.')
    return meta_data


def check_asi_list(asi_list, tz=1, limit_exp_time=None, name_convention='dlr', n_workers=0):
    """Checks a list of all-sky image files for corruption and returns dataframe with additional data.

    :param asi_list: list of asi files.
    :param tz: timezone as int (+/- UTC).
    :param limit_exp_time: limit of valid exposure time.
    :param name_convention: asi file name convention.
    :param n_workers: number of workers to use for parallel processing.
    :return: meta data of asi files as dataframe.
    """

    checked_list = parallel(check_asi, asi_list, tz=tz, limit_exp_time=limit_exp_time, name_convention=name_convention,
                            n_workers=n_workers)
    df_asi = pd.DataFrame(checked_list)
    return df_asi


def check_asi(filename, tz="UTC+0100", is_mobotix=True, limit_exp_time=None, name_convention='dlr'):
    """
    Checks a single ASI (All-Sky Image) file for corruption and extracts metadata.

    :param filename: Path to the ASI image file.
    :param tz: Timezone for parsing metadata timestamps. Default is "UTC+0100".
    :param is_mobotix: Boolean indicating whether the image follows the Mobotix format. Default is True.
    :param limit_exp_time: Optional threshold for maximum exposure time. If exceeded, a warning is logged.
    :param name_convention: Naming convention used for parsing metadata. Default is 'dlr'.

    :return: A dictionary containing:
        - 'name': Extracted image name from metadata (or NaN if unavailable).
        - 'timestamp': Extracted timestamp from metadata (or NaN if unavailable).
        - 'exposure_time': Extracted exposure time from metadata (or NaN if unavailable).
        - 'illuminance': Extracted illuminance value from metadata (or NaN if unavailable).
        - 'width': Image width in pixels.
        - 'height': Image height in pixels.
        - 'corrupted': Boolean indicating if the image is corrupted (True if corrupted, False otherwise).

    :raises Warning: Logs a warning if the exposure time exceeds `limit_exp_time`.
    """
    filename = Path(filename)
    asi_dict = {}
    try:
        meta_data = read_asi_meta_data(filename, tz=tz, is_mobotix=is_mobotix, name_convention=name_convention)
        asi_dict['name'] = meta_data.get('name', np.nan)
        asi_dict['timestamp'] = meta_data.get('timestamp', np.nan)
        asi_dict['exposure_time'] = meta_data.get('exposure_time', np.nan)
        asi_dict['illuminance'] = meta_data.get('illuminance', np.nan)
        if limit_exp_time is not None and asi_dict['exposure_time'] > limit_exp_time:
            logging.warning(f'Invalid exposure time ({asi_dict["exposure_time"]}) for {filename}')
        asi = PIL.Image.open(filename)
        asi_dict['width'] = asi.width
        asi_dict['height'] = asi.height
        asi_dict['corrupted'] = False
        asi.load()
    except:
        logging.warning(f'Corrupted image {filename}. Skipping image.')
        asi_dict['corrupted'] = True
    return asi_dict


def load_transform_save_asi(rel_path, all_sky_imager, source_dir, target_dir):
    """Loads, transforms and saves transformed all-sky image.

    :param rel_path: relative file path of image.
    :param all_sky_imager: camera used to take image.
    :type all_sky_imager: AllSkyImager.
    :param source_dir: directory of raw images.
    :param target_dir: directory of transformed images.
    :return: True/False depending on success.
    """
    failed = False
    try:
        asi = all_sky_imager.load_image(source_dir / rel_path)
        tfmd_asi = all_sky_imager.transform(asi)
        all_sky_imager.save_image(tfmd_asi, image_file=target_dir / rel_path)
    except Exception as e:
        failed = True
        logging.debug(e)
        logging.error(f'Could not save transformed image {rel_path}. Skipping')
    return failed


def create_asi_list(asi_root, do_check=False, name_convention='dlr', csv_file=None, n_workers=0):
    """Gets all asi within asi_root and save the list to csv.

    :param asi_root: root folder where images are stored.
    :param do_check: if true, all images are checked for validity.
    :param name_convention: asi file name convention.
    :param csv_file: csv file to save results.
    :param n_workers: number of workers to use for parallel processing.
    :return: None.
    """

    all_asi = get_image_files(asi_root)
    logging.info(f'{len(all_asi)} images found.')
    if do_check:
        df_asi = check_asi_list(all_asi, name_convention=name_convention, n_workers=n_workers)
        logging.info(f'ASI Data checked for corrupted images.')
    else:
        df_asi = pd.DataFrame(
            columns=['timestamp', 'res_x', 'res_y', 'exposure_time', 'corrupted', 'rel_path'],
            index=range(len(all_asi)))
    df_asi['rel_path'] = pd.Series(all_asi).apply(lambda x: x.relative_to(asi_root).as_posix())
    if csv_file is not None:
        df_asi.to_csv(csv_file, header=False, index=False)
        logging.info(f'ASI Data saved to csv.')
    return df_asi


def create_asi_dataset(asi_series, source_dir, target_dir, camera_data_dir, n_workers=0, asi_tfms=None):
    """Creates an ASI dataset from all passed filenames in target_dir.

    :param asi_series: pd.Series of all-sky images, with timestamp of acquisition as index and camera name as name.
    :param source_dir: directory of raw images.
    :param target_dir: directory of transformed images.
    :param camera_data_dir: directory of yaml files containing camera data.
    :param n_workers: number of workers to use for parallel processing.
    :param kwargs: kwargs for applying transformation.
    :return: pd.Series of successfully saved (transformed) images.
    """

    if isinstance(asi_series.index, pd.MultiIndex):
        dt_index = asi_series.index.get_level_values(0)
    else:
        dt_index = asi_series.index
    assert dt_index.inferred_type == 'datetime64', 'Invalid index of series. Must be DatetimeIndex (datetime64)'
    camera_name = asi_series.name
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    camera_data = load_camera_data(camera_data_dir=camera_data_dir, camera_name=camera_name)
    success = pd.Series(False, index=asi_series.index, name='success')
    logging.info(f'Started creation of dataset with transformations: {asi_tfms}')
    for cam_id, cam_dict in camera_data.items():
        camera_data_i = camera_data[cam_id]
        cam_id_idx1 = dt_index.date >= camera_data_i['mounted'].date()
        cam_id_idx2 = dt_index.date <= camera_data_i['demounted'].date()
        part_asi_series = asi_series.loc[cam_id_idx1 & cam_id_idx2]
        if len(part_asi_series) == 0:
            continue
        try:
            all_sky_imager = AllSkyImager(camera_data_i, tfms=asi_tfms)
        except Exception:
            logging.warning(f'Could not create AllSkyImager for camera {camera_name} with cam_id {cam_id}. Skipping.')
            continue
        logging.info(f'{len(part_asi_series)} images found for camera {camera_name} with cam_id {cam_id}')
        logging.info(f'Starting parallel image transformation and storing at {target_dir}')
        fails = parallel(load_transform_save_asi, part_asi_series, all_sky_imager=all_sky_imager, source_dir=source_dir,
                         target_dir=target_dir, n_workers=n_workers)
        logging.info(f'{sum(~fails)}/{len(fails)} images transformed and saved')
        success.loc[part_asi_series.index] = ~fails
    return success


def read_asi_dataset(csv_file, img_dir=None, asi_path_col='rel_path', drop_asi_filepath=True, filter_dates=None):
    """
    Reads an ASI dataset from a CSV file and optionally filters by date.

    :param csv_file: Path to the CSV file containing ASI metadata.
    :param img_dir: Optional directory path where ASI images are stored. If provided, file paths will be adjusted accordingly.
    :param asi_path_col: Column name in the CSV that contains the relative file paths of ASI images. Default is 'rel_path'.
    :param drop_asi_filepath: Whether to drop the ASI file path column from the returned DataFrame. Default is True.
    :param filter_dates: Optional list of dates to filter the dataset. Only entries matching these dates will be retained.

    :return:
        - asi_files: A Pandas Series containing file paths to ASI images.
        - df: A Pandas DataFrame with metadata, optionally filtered and with the ASI path column removed.
    """
    df = pd.read_csv(csv_file, parse_dates=['timestamp'], index_col=['timestamp'])
    if filter_dates is not None:
        df = df[pd.DatetimeIndex(df.index.date).isin(filter_dates)]
    asi_files = df.loc[~df[asi_path_col].isna(), asi_path_col].apply(
        lambda x: Path(img_dir)/Path(x.replace('\\', '/')) if img_dir is not None else Path(x.replace('\\', '/')))
    if drop_asi_filepath:
        df.drop(columns=asi_path_col, inplace=True)
    return asi_files, df


def merge_meteo_and_asi_data(df_meteo, df_asi, temporal_resolution='30s', max_delta_t=15, parameters_to_cast=None):
    """
    Merges meteorological data with ASI data based on timestamps.

    :param df_meteo: Pandas DataFrame containing meteorological data indexed by timestamp.
    :param df_asi: Pandas DataFrame containing ASI metadata indexed by timestamp.
    :param temporal_resolution: Time rounding resolution for ASI timestamps (e.g., '30s' for 30 seconds). Default is '30s'.
    :param max_delta_t: Maximum allowed time difference (in seconds) for matching ASI data to meteorological data. Default is 15 seconds.
    :param parameters_to_cast: Optional dictionary specifying data types for certain parameters after merging.

    :return: A Pandas DataFrame with meteorological and ASI data merged, indexed by timestamp.
    """
    df_asi_mapped = map_asi_to_timestamps(df_asi, round_to=temporal_resolution, max_delta_t=max_delta_t)
    df = df_meteo.join(df_asi_mapped, how='left')
    df.rename(columns={'timestamp': 'asi_timestamp'}, inplace=True)
    df.index.name = 'timestamp'
    if parameters_to_cast is not None:
        df = df.astype(parameters_to_cast)
    df.dropna(axis=1, how='all', inplace=True)
    dt_min, dt_max = df_asi_mapped.index.min(), df_asi_mapped.index.max()
    df = df[(df.index >= dt_min) & (df.index <= dt_max)]
    return df


def map_asi_to_timestamps(df, round_to='60s', max_delta_t=10, valid_exp_times=None, max_delta_exp_time=10,
                          multi_exposure=False, inplace=False):
    """Maps asi acqusition time to a rounded timestamp.

    :param df: dataframe containing a column 'timestamp'.
    :param round_to: string of resolution to round timestamps to.
    :param max_delta_t: maximal allowed deviation to rounded timestamp in seconds.
    :param valid_exp_times: tuple of valid exposure times to be considered.
    :param inplace: if true, overwrites existing dataframe.
    :return: dataframe with rounded timestamp as index.
    """

    if not inplace:
        df = df.copy()
    # define series of rounded timestamps and corresponding time delta
    df['rounded_ts'] = df['timestamp'].dt.round(round_to)
    delta_t = (df['timestamp'] - df['rounded_ts']).round('s')
    df['delta_t'] = np.abs(delta_t)
    if multi_exposure:
        # remove all rows where delta_t is negative (do not use images that were taken before timestamp)
        df = df[delta_t >= pd.Timedelta(0)]
    # remove data deviating too much from timestamp
    df = df[df['delta_t'] <= pd.Timedelta(max_delta_t, 'sec')]
    # remove corrupted data
    df = df[~df['corrupted']]
    # remove invalid exposure times if applicable
    if valid_exp_times is not None:
        delta_exp_time = np.min(np.abs(np.asarray([df['exposure_time'] - exp_time for exp_time in valid_exp_times])), axis=0)
        df = df[delta_exp_time < max_delta_exp_time]

    # drop remaining duplicates (the ones with highest deviation from timestamp
    if multi_exposure:
        # sort by rounded timestamp and minimal deviation of timestamps
        df.sort_values(by=['rounded_ts', 'exposure_time', 'delta_t'], inplace=True)
        df.drop_duplicates(subset=['rounded_ts', 'exposure_time'], inplace=True)
        df.set_index(['rounded_ts', 'exposure_time'], inplace=True)
    else:
        df.sort_values(by=['rounded_ts', 'delta_t'], inplace=True)
        df.drop_duplicates(subset=['rounded_ts'], inplace=True)
        df.set_index('rounded_ts', inplace=True)
    return df


def select_by_dni_var_classes(dni_var_classes, selected_classes, include_by='H'):
    """Selects timestamp by dni variability class. A timestamp is selected if the timestamp itself or the included time
    frame has a dni variability class contained in selected_classes.

    :param dni_var_classes: pd.Series of dni var classes with DatetimeIndex
    :param selected_classes: list of dni var classes to filter by.
    :param include_by: determines size of time frame (e.g., 'H' means hour)
    :return: selected timestamps as DatetimeIndex.
    """
    high_var_idx = dni_var_classes[dni_var_classes.isin(selected_classes)].index
    idx_selected = dni_var_classes[dni_var_classes.index.round(include_by).isin(
        high_var_idx.round(include_by).unique())].index
    return idx_selected


def check_Q25_asi_cropping(rel_path_to_image, asi_root):
    """This function can be used to determine how asis from the Q25 all sky imager have been cropped for custom
    resolution. The function is used for images of the Kontas camera in the interval from '20160920' to '20190612'.

    :param rel_path_to_image: path to asi image, relative to asi root directory
    :param asi_root: absolute path to asi root directory
    :return string specifying how image has been cropped (left, center, right). If the image can't be opened the
    function returns nan.
    """
    filename = Path(asi_root, rel_path_to_image)
    try:
        # load image
        asi = PIL.Image.open(filename)
        # initialize numpy array from pil image
        asi_numpy = np.array(asi)
        # sum up rgb channels
        rgb = asi_numpy.sum(axis=2)
        # find the widest row
        image_widths = np.sum(rgb > 0, axis=1)
        widest_row_index = np.argmax(image_widths)
        widest_row = rgb[widest_row_index, :]
        # find the thinnest left border width
        border_width = np.nonzero(widest_row)[0][0]
        # decide how image has been cropped by camera chip
        if border_width > Q25_RIGHT_CENTRATION_THRESHOLD:
            crop = 'right'
        elif (border_width > Q25_LEFT_CENTRATION_THRESHOLD) and (border_width < Q25_RIGHT_CENTRATION_THRESHOLD):
            crop = 'center'
        else:
            crop = 'left'
    except:
        logging.warning(f'Can not open image {filename}. Skipping image.')
        crop = float('nan')
    return crop


def get_dates_from_csv(csv_file, col_name='date'):
    """
    Extracts unique dates from a specified column in a CSV file.

    :param csv_file: Path to the CSV file containing date information.
    :param col_name: Column name in the CSV that contains date values. Default is 'date'.

    :return: A NumPy array of unique dates extracted from the specified column.
    """
    dates = pd.read_csv(csv_file)
    dates = pd.DatetimeIndex(dates[col_name].values).date
    return dates


def filter_timestamps_by_sun_elevation(ts, min_el, sun_el=None, latitude=None, longitude=None, altitude=None):
    """
    Filters timestamps based on minimum solar elevation.

    :param ts: Pandas DatetimeIndex of timestamps to be filtered.
    :param min_el: Minimum solar elevation angle (in degrees) required for timestamps to be retained.
    :param sun_el: Optional Pandas Series containing precomputed solar elevations for the timestamps.
                   If None, solar elevation will be computed using latitude, longitude, and altitude.
    :param latitude: Latitude of the location (required if sun_el is not provided).
    :param longitude: Longitude of the location (required if sun_el is not provided).
    :param altitude: Altitude of the location in meters (optional, used when computing solar elevation).

    :return: A filtered Pandas DatetimeIndex containing only timestamps where solar elevation exceeds min_el.
    """
    if sun_el is None:
        df_solarpos = pvlib.solarposition.get_solarposition(ts, latitude, longitude, altitude)
        sun_el = df_solarpos['apparent_elevation']
    return ts[sun_el > min_el]
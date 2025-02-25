# Licensed under the Apache License: http://www.apache.org/licenses/LICENSE-2.0
# For details: https://github.com/DLR-SF/sky_imaging/blob/main/NOTICE.txt

"""
This module provides basic helper functions.
"""

from pathlib import Path
import os
import pandas as pd
import numpy as np
import cv2
from h5py import File, Group
import glob
from datetime import date, datetime
import re
import pytz
import shutil
import logging
from fastcore.parallel import parallel
from scipy import stats

from asi_core.constants import IGNORED_PATTERNS_SUBDAY, IMAGE_EXTENSIONS, FSTRING_RE


def ifnone(var, otherwise):
    """If a variable's original value is None, replace it with an alternative value"""
    return var if var is not None else otherwise


def get_absolute_path(filepath, root=None, as_string=False):
    """Combine root and relative path and resolve to absolute path."""
    absolute_path = Path(filepath)
    if root is not None:
        absolute_path = Path(root) / filepath
    absolute_path = absolute_path.resolve()
    if as_string:
        absolute_path = str(absolute_path)
    return absolute_path


def replace_double_backslashes_with_slashes_in_path(str_path, root_dir=None):
    """Replace double backslashes from windows paths with slashes."""
    if root_dir is None:
        return Path(str_path.replace('\\', '/'))
    else:
        return Path(root_dir)/Path(str_path.replace('\\', '/'))


def _get_files(p, fs, extensions=None, substring=None):
    """Get all files in path with 'extensions' and a name containing a 'substring'."""
    p = Path(p)
    res = [p / f for f in fs if not f.startswith('.')
           and ((not extensions) or f'.{f.split(".")[-1].lower()}' in extensions)
           and ((not substring) or substring in f)]
    return res


def get_files(path, extensions=None, substring=None, recursive=True, folders=None, followlinks=True):
    """Get all files in `path` with optional `extensions` or `substring`, optionally `recursive`, only in `folders`,
    if specified."""
    path = Path(path)
    folders = list(ifnone(folders, []))
    extensions = set(ifnone(extensions, []))
    extensions = {e.lower() for e in extensions}
    if recursive:
        res = []
        for i, (p, d, f) in enumerate(os.walk(path, followlinks=followlinks)):
            if len(folders) != 0 and i == 0:
                d[:] = [o for o in d if o in folders]
            else:
                d[:] = [o for o in d if not o.startswith('.')]
            if len(folders) != 0 and i == 0 and '.' not in folders:
                continue
            res += _get_files(p, f, extensions, substring)
    else:
        f = [o.name for o in os.scandir(path) if o.is_file()]
        res = _get_files(path, f, extensions, substring)
    return list(res)


def get_image_files(path, recursive=True, folders=None, extensions=IMAGE_EXTENSIONS, substring=None):
    """Get image files in `path` recursively, only in `folders` and with `substring`, if specified."""
    return get_files(path, extensions=extensions, substring=substring, recursive=recursive, folders=folders)


def copy_file(src_file, tgt_file=None, tgt_dir=None, create_parents=False):
    """
    Copies a file to a specified target file or directory.

    :param src_file: Path to the source file.
    :param tgt_file: Path to the target file (optional if tgt_dir is provided).
    :param tgt_dir: Path to the target directory (optional if tgt_file is provided).
    :param create_parents: Whether to create parent directories if they do not exist (default: False).
    """
    assert not (tgt_file is None and tgt_dir is None), 'Either tgt_file or tgt_dir must be specified.'
    if tgt_file is None:
        tgt_file = Path(tgt_dir) / Path(src_file).name
    if create_parents:
        Path(tgt_file).parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy(str(src_file), str(tgt_file))
        logging.info(f"Successfully copied {src_file} to {tgt_file}")
    except FileNotFoundError:
        logging.error(f"Error: File {src_file} not found.")
    except IsADirectoryError:
        logging.error(f"Error: Destination {tgt_file} is a directory, not a file.")
    except Exception as e:
        logging.info(f"Error: {e}")


def copy_file_relative_to_directories(rel_filepath, src_dir, tgt_dir):
    """
    Copies a file from the source directory to the target directory while preserving its relative path.

    :param rel_filepath: File path relative to the source directory.
    :param src_dir: Root source directory.
    :param tgt_dir: Root target directory.
    """
    abs_src_file = get_absolute_path(rel_filepath, root=src_dir)
    abs_tgt_file = get_absolute_path(rel_filepath, root=tgt_dir)
    Path(abs_tgt_file).parent.mkdir(parents=True, exist_ok=True)
    copy_file(abs_src_file, abs_tgt_file)


def parallel_copy_files(filepaths, src_dir, tgt_dir, keep_dir_structure=False, num_workers=0):
    """
    Copies multiple files in parallel, optionally preserving directory structure.

    :param filepaths: List of file paths to copy.
    :param src_dir: Source directory (used to determine relative paths).
    :param tgt_dir: Target directory where files will be copied.
    :param keep_dir_structure: Whether to maintain the original directory structure (default: False).
    :param num_workers: Number of parallel workers for file copying (default: 0, meaning sequential execution).
    """
    relative_paths = False if src_dir is None else True
    assert Path(tgt_dir).is_dir(), 'Target directory does not exist.'
    if keep_dir_structure:
        if not relative_paths:
            filepaths = pd.Series(filepaths).apply(lambda x: Path(x).relative_to(src_dir))
        parallel(copy_file_relative_to_directories, filepaths, src_dir=src_dir, tgt_dir=tgt_dir,
                             n_workers=num_workers)
    else:
        if relative_paths:
            filepaths = pd.Series(filepaths).apply(lambda x: get_absolute_path(filepath=x, root=src_dir))
        parallel(copy_file, filepaths, tgt_dir=tgt_dir)


def get_number_of_nans(df):
    """
    Count number of nans in a dataframe columns-wise and total number of rows with a NaN value in at least one column.
    """
    df_nan = pd.DataFrame({col: df[col].isna().sum() for col in df.columns}, index=['Num of NaN rows'])
    df_nan['total'] = len(df[df.isna().any(axis=1)])
    return df_nan


def get_ETC_GMT_timezone(desired_timezone="GMT"):
    """
    Return a `pytz` timezone object corresponding to the desired Etc/GMT timezone.

    Example: UTC+2: get_ETC_GMT_timezone('GMT+2') returns pytz timezone object for "Etc/GMT-2"

    :param desired_timezone: (str) A string specifying the desired GMT timezone. It should be in the format 'GMT+1'
        (in the case of UTC+1). If a '+' sign is used, it will be replaced with '-' to follow the Etc/GMT convention
        (https://en.wikipedia.org/wiki/Tz_database#Area).
    :return: pytz.timezone, Etc/GMT timezone
    """
    if '+' in desired_timezone:
        desired_timezone = desired_timezone.replace('+', '-')
    elif '-' in desired_timezone:
        desired_timezone = desired_timezone.replace('-', '+')
    return pytz.timezone('Etc/' + desired_timezone)


def parse_datetime(dt_string, datetime_format="%Y%m%d%H%M%S"):
    """Extracts timestamp from filename."""
    pattern = datetime_format.replace('%Y', r'\d{4}')
    pattern = pattern.replace('%y', r'\d{2}')
    pattern = pattern.replace('%m', r'\d{2}')
    pattern = pattern.replace('%d', r'\d{2}')
    pattern = pattern.replace('%H', r'\d{2}')
    pattern = pattern.replace('%M', r'\d{2}')
    pattern = pattern.replace('%S', r'\d{2}')
    match = re.search(pattern, dt_string)
    if match:
        timestamp_str = match.group(0)
        timestamp = datetime.strptime(timestamp_str, datetime_format)
        return timestamp
    else:
        raise ValueError(f"Could not extract timestamp from filename: {dt_string}")


def get_temporal_resolution_from_timeseries(data):
    """Get temporal resolution from a pandas DataFrame with a DatetimeIndex."""
    return pd.Timedelta(stats.mode(data.index.to_series().diff().round('1s'), keepdims=False).mode)


def assemble_path(path_structure, camera_name, timestamp, set_subday_to_wildcard=False, exposure_time=None):
    """Assemble path to images or other files and replace timestamp and camera name 'tags' with actual values

    :param path_structure: (str) path to each image, containing {camera_name} where the camera name should be inserted
        and {timestamp:...} (e.g. {timestamp:%Y%m%d%H%M%S%f}) where the evaluated timestamp should be inserted
    :param camera_name: (str) Name of the camera as specified in config file and used in (image) folder structure
    :param timestamp: (datetime, tz-aware) Timestamp for which an (image) file is requested
    :param set_subday_to_wildcard: (bool) If True, replace formatters indicating hours, minutes etc. with wildcards
    :param exposure_time: (int) exposure time of images has to be set if set_subday_to_wildcard is False
    :return assemble_path: (str) assembled path
    """
    if set_subday_to_wildcard:
        # replace hour with wildcard
        for replace in IGNORED_PATTERNS_SUBDAY:
            path_structure = re.sub(replace['pattern'], replace['substitution'], r'{}'.format(path_structure))
        assemble_path = path_structure.format(camera_name=camera_name, timestamp=timestamp, exposure_time=exposure_time)
    else:
        if exposure_time is None:
            assemble_path = path_structure.format(camera_name=camera_name, timestamp=timestamp)
        else:
            assemble_path = path_structure.format(camera_name=camera_name, timestamp=timestamp,
                                                  exposure_time=exposure_time)

    return assemble_path


def fstring_to_re(string):
    """Convert from f-string syntax to regular expression syntax

    Only a limited set of formatters supported so far, FSTRING_RE should be extended as needed.
    """
    # make sure not to confuse re and f-string curly brackets
    string = re.sub(r'{(.+?)}', lambda m: '__{__' + m.groups()[0] + '__}__', string)
    for replace in FSTRING_RE:
        string = re.sub(r'(__{__.*?)(' + replace['formatter'] + r')(.*?__}__)',
                        lambda m: m.groups()[0] + replace['re'] + m.groups()[2], string)
    string = re.sub(r'__{__(\w+):(.+?)__}__', lambda m: r'(?P<' + m.groups()[0] + '>' + m.groups()[1] + ')', string)
    string = re.sub(r'__{__(\w+)__}__', lambda m: r'(?P<' + m.groups()[0] + '>*)', string)
    return string


class DailyH5:
    """Base class for DailyH5 file manipulation"""

    def __init__(self, products_path, meta_infos={}):
        """
        Initializes a writer of daily h5 files

        :param products_path: Basepath to which daily h5 files will be stored
        :param meta_infos: Dict of meta infos valid for a whole daily h5 file. Stored once in the h5 file.
        """

        self.products_path = Path(products_path)
        self.meta_infos = meta_infos

        self.daily_h5 = {'date': date(1980, 1, 1), 'path': None}

    def get_file(self, timestamp):
        """
        Get the path to the current daily h5 file. Initialize if not done yet.

        :param timestamp: Timestamp of the data to be stored
        :return: Path to the current daily h5 file
        """

        if self.daily_h5['date'] != timestamp.date():
            self.daily_h5 = {'date': date(1980, 1, 1), 'path': None}
            self.init_h5file(timestamp)

        return self.daily_h5['path']

    def init_h5file(self, timestamp):
        """
        Initialize daily h5 file for reading or writing.

        :param timestamp: Timestamp of the current data to be stored
        """
        raise Exception('Not implemented')

    def process_entry(self, timestamp, mode, data=None, timestamp_forecasted=None):
        """
        Stores a dataset of one timestamp to the daily h5 file

        :param timestamp: Timestamp based on which dataset was created
        :param mode: character, r/w/a, i.e. read, write or append
        :param data: Dataset to be saved either dataset which can be stored by h5py or dict of such datasets
        :param timestamp_forecasted: (Optional) timestamp forecasted by the dataset
        """

        target_file = self.get_file(timestamp)
        label_entry = f'{timestamp:%H%M%S}'
        if timestamp_forecasted is not None:
            if timestamp_forecasted.date() != timestamp.date():
                raise Exception('Only intraday forecasts expected!')

            label_entry += f'_{timestamp_forecasted:%H%M%S}'

        with File(target_file, mode) as f:
            data_out = self.process_sub_entry(f, label_entry, data)
        return data_out

    def process_sub_entry(self, label, data=None):
        """
        Defines the read/ write operation to be applied recursively

        :param label: Label of the current data to be stored/ read
        :param data: Data to be processed
        """
        raise Exception('Not implemented')


class DailyH5Writer(DailyH5):

    def store_entry(self, timestamp, data, timestamp_forecasted=None):
        """
        Stores a dataset of one timestamp to the daily h5 file

        :param timestamp: Timestamp based on which dataset was created
        :param timestamp_forecasted: (Optional) timestamp forecasted by the dataset
        :param data: Dataset to be saved either dataset which can be stored by h5py or dict of such datasets
        """
        self.process_entry(timestamp, 'a', data=data, timestamp_forecasted=timestamp_forecasted)

    def init_h5file(self, timestamp, do_not_overwrite=True):
        """
        Initialize daily h5 file, create folders if required, store meta infos to a new h5 file.

        :param timestamp: Timestamp of the current data to be stored
        :param do_not_overwrite: If called and a daily file already exists, create additional file instead of
                                 overwriting the previous one.
        """

        if not os.path.isdir(self.products_path):
            os.makedirs(self.products_path)

        file_counter = 0
        h5_path = str(self.products_path / f'{timestamp:%Y%m%d}')
        if do_not_overwrite:
            file_counter = len(glob.glob(h5_path + '*.h5'))
        if file_counter:
            h5_path += f'_{file_counter+1}'
        h5_path += '.h5'

        with File(h5_path, 'w') as f:
            data = self.process_sub_entry(f, 'meta', self.meta_infos)

        self.daily_h5 = {'date': timestamp.date(), 'path': h5_path}

    def process_sub_entry(self, handle, label, data):
        """
        Recursively store all datasets in data

        :param handle: Handle to an h5file or a group in an h5 file
        :param label: Label under which data will be stored
        :param data: dataset or dict of datasets
        """

        if type(data) is dict:
            handle.create_group(label)
            for k, v in data.items():
                self.process_sub_entry(handle[label], k, v)
        else:
            handle[label] = data
        return None


class DailyH5Reader(DailyH5):
    @staticmethod
    def list_entries(h5_path):
        """
        Generate a dataframe of the keys and corresponding timestamps and forecasted timestamps in the h5 file.

        :param h5_path: Path of the h5 file, the ke
        :return: Dataframe with columns key, timestamp, forecasted_timestamp
        """

        filename = Path(h5_path).stem

        entries = pd.DataFrame({'key': [], 'timestamp': [], 'timestamp_forecasted': []})

        with File(h5_path, 'r') as f:
            for k in f.keys():
                timestamp = None
                forecasted_timestamp = None
                fmt_1 = re.search(r'(\d{6})_(\d{6})', k)
                fmt_2 = re.search(r'(\d{6})', k)
                if type(fmt_1) is re.Match:
                    timestamp = datetime.strptime(filename + fmt_1.groups()[0], '%Y%m%d%H%M%S')
                    forecasted_timestamp = datetime.strptime(filename + fmt_1.groups()[1], '%Y%m%d%H%M%S')
                elif type(fmt_2) is re.Match:
                    timestamp = datetime.strptime(filename + fmt_2.groups()[0], '%Y%m%d%H%M%S')
                new_entry = pd.DataFrame({'key': [k], 'timestamp': [timestamp],
                                          'timestamp_forecasted': [forecasted_timestamp]})
                if not entries.empty:
                    entries = pd.concat([entries, new_entry], ignore_index=True)
                else:
                    entries = new_entry
        return entries

    @staticmethod
    def init_from_path(timestamp, h5_path):
        """
        Create a DailyH5Reader instance and initializes it from a specific h5 file

        :param timestamp: Timestamp based on which dataset was created
        :param h5_path: Path of the h5 file to be read (naming can deviate from convention of DailyH5Writer)
        :return: DailyH5Reader instance
        """

        reader = DailyH5Reader('')
        with File(h5_path, 'r') as f:
            reader.meta_infos = reader.process_sub_entry(f, 'meta')

        reader.daily_h5 = {'date': timestamp.date(), 'path': h5_path}
        return reader

    def get_entry(self, timestamp, timestamp_forecasted=None):
        """
        Stores a dataset of one timestamp to the daily h5 file

        :param timestamp: Timestamp based on which dataset was created
        :param timestamp_forecasted: (Optional) timestamp forecasted by the dataset

        :return: Dataset to be saved either dataset which can be stored by h5py or dict of such datasets
        """
        return self.process_entry(timestamp, 'r', timestamp_forecasted=timestamp_forecasted)

    def process_sub_entry(self, handle, label, data=None):
        """
        Recursively get all datasets in handle

        :param handle: Handle to an h5file or a group in an h5 file
        :param label: Label under which data will be stored
        :param data: dataset or dict of datasets
        """
        sub_handle = handle[label]
        if type(sub_handle) is Group:
            data = {}
            for k in sub_handle.keys():
                data[k] = self.process_sub_entry(sub_handle, k)
        else:
            data = sub_handle[()]
        return data

    def init_h5file(self, timestamp, file_counter=0):
        """
        Initialize daily h5 file, load meta data.

        :param timestamp: Timestamp of the current data to be stored
        :param file_counter: Appends counter suffix to file name. May be useful if multiple files created for a day.
        """

        h5_path = str(self.products_path / f'{timestamp:%Y%m%d}')
        if file_counter:
            h5_path += f'_{file_counter+1}'
        h5_path += '.h5'

        with File(h5_path, 'r') as f:
            self.meta_infos = self.process_sub_entry(f, 'meta')

        self.daily_h5 = {'date': timestamp.date(), 'path': h5_path}


def adjust_gamma(image, gamma=1.0):
    """
    Only for improved visibility, for radiometric evaluations reconsider

    Taken from https://pyimagesearch.com/2015/10/05/opencv-gamma-correction/

    Build a lookup table mapping the pixel values [0, 255] to their adjusted gamma values and apply it.

    :param image: Input RGB image
    :param gamma: Gamma scalar parameter
    :return: Gamma-corrected image
    """
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


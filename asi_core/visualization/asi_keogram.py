# Licensed under the Apache License: http://www.apache.org/licenses/LICENSE-2.0
# For details: https://github.com/DLR-SF/sky_imaging/blob/main/NOTICE.txt

"""
This module provides functions to create keograms from all-sky images.
"""

import logging
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import math
from pathlib import Path
import pandas as pd
from PIL import Image

from asi_core.matlab_converters import load_matlab_ceilometer_data
from asi_core.basics import get_image_files, parse_datetime, ifnone


def get_full_hour_indices(datetimes):
    """
    Get the indices of the full hours in a list of datetimes.

    :param datetimes: A list of datetimes.
    :type datetimes: list
    :param sampling_rate: The sampling rate used to round the datetimes. Default is '30s'.
    :type sampling_rate: str
    :return: An array containing the indices of the full hours in the list of datetimes.
    :rtype: numpy.ndarray
    """
    dt_index = pd.DatetimeIndex(datetimes)
    all_sampling_rates = dt_index.to_series().diff().value_counts()
    most_common_sampling_rate = all_sampling_rates.idxmax().total_seconds()
    dt_index_rounded = dt_index.round(f'{most_common_sampling_rate:.0f}s')
    full_hour_indices = np.argwhere((dt_index_rounded.minute == 0) & (dt_index_rounded.second == 0))[:, 0]
    return full_hour_indices


def get_image_slice(image, slice_pos, strip_size):
    """
    Get a slice of the image based on the given slice position and strip size.

    :param image: The image to slice.
    :type image: ndarray
    :param slice_pos: The position of the slice.
    :type slice_pos: int
    :param strip_size: The size of the strip.
    :type strip_size: int
    :return: The sliced image.
    :rtype: ndarray
    """
    half_strip_left = math.floor(strip_size / 2)
    half_strip_right = math.ceil(strip_size / 2)
    slice_img = image[:, slice_pos - half_strip_left:slice_pos + half_strip_right, :]
    return slice_img


def compose_keogram(image_files, slice_pos, strip_size):
    """
    Compose a keogram from a list of image files.

    :param image_files: A list of paths to image files.
    :type image_files: List[str]
    :param slice_pos: The position of the slice to extract from each image.
    :type slice_pos: int
    :param strip_size: The width of the image slice.
    :type strip_size: int
    :return: A tuple containing the concatenated image slices as a NumPy array and the timestamps as a Pandas DatetimeIndex.
    :rtype: tuple
    :raises Exception: If an error occurs while reading or processing an image file.
    """
    # Get the shape of the first image to determine the height (assuming all images are the same size)
    first_image = Image.open(image_files[0])
    img_height = first_image.size[1]  # Height of the image
    first_image.close()

    # Pre-allocate a NumPy array to hold all the slices
    num_images = len(image_files)
    keogram = np.zeros((img_height, num_images * strip_size, 3), dtype=np.uint8)
    timestamps = pd.Series([pd.NaT] * num_images, dtype="datetime64[ns]")

    for i, image_file in enumerate(image_files):
        try:
            img_timestamp = parse_datetime(image_file.name)
            timestamps.iloc[i] = img_timestamp
            with Image.open(image_file) as img:
                img = np.array(img)
                img_slice = get_image_slice(img, slice_pos, strip_size)
                keogram[:, i * strip_size:(i + 1) * strip_size, :] = img_slice
        except:
            logging.warning(f'{image_file} is invalid')
    return keogram, pd.DatetimeIndex(timestamps)


def plot_keogram(keogram, datetimes_slices, strip_size, camera_name='', title=None, ax=None):
    """
    Plot a keogram.

    :param keogram: The keogram data to be plotted.
    :type keogram: ndarray
    :param datetimes_slices: The datetime slices corresponding to each column of the keogram.
    :type datetimes_slices: ndarray
    :param strip_size: The size of each strip in the keogram.
    :type strip_size: int
    :param camera_name: The name of the camera. (optional)
    :type camera_name: str, default=''
    :param title: The title of the plot. (optional)
    :type title: str, default=None
    :param ax: The matplotlib axes to plot on. (optional)
    :type ax: AxesSubplot, default=None

    :return: The matplotlib figure object. If `ax` is provided, returns None.
    :rtype: Figure
    """
    full_hour_indices = get_full_hour_indices(datetimes_slices)
    xticks_pos = full_hour_indices * strip_size
    xticks = datetimes_slices[full_hour_indices].strftime("%H:%M")
    title = ifnone(title, camera_name + " " + datetimes_slices[0].strftime("%d.%m.%Y"))
    tz = ifnone(datetimes_slices[0].tz, 'tz unknown')

    if ax is None:
        fig, ax = plt.subplots(1,1)
    else:
        fig = None
    ax.imshow(keogram)
    ax.get_yaxis().set_visible(False)
    ax.set_xticks(xticks_pos, xticks, rotation=45)
    ax.set_xlabel(f'Time of the day in HH:MM ({tz})')
    ax.set_title(title)

    return fig


def plot_ceilometer_data(ceilometer_data_path, start_time, end_time, ax=None, tz_convert=None):
    """
    Plots ceilometer data within a specified time range.

    :param ceilometer_data_path: The path to the ceilometer data file.
    :type ceilometer_data_path: str
    :param start_time: The start time of the time range.
    :type start_time: datetime
    :param end_time: The end time of the time range.
    :type end_time: datetime
    :param ax: The axes to plot on. If not provided, a new figure and axes will be created.
    :type ax: matplotlib.axes.Axes, optional
    :param tz_convert: The timezone to convert the data to.
    :type tz_convert: str, optional

    :return: The figure containing the plot.
    :rtype: matplotlib.figure.Figure
    """
    ceilo_data = load_matlab_ceilometer_data(mat_file=ceilometer_data_path)
    ceilo_data[ceilo_data < 0] = np.nan
    if tz_convert is not None:
        ceilo_data = ceilo_data.tz_convert(tz=tz_convert)
    # Filter the data by timestamps
    ceilo_data_interval = ceilo_data.between_time(
        start_time=start_time.astimezone(tz=ceilo_data.index.tz).time(),
        end_time=end_time.astimezone(tz=ceilo_data.index.tz).time()).copy()
    # Workaround to plot correct time on x-axis if no cloud was detected in the beginning or end of the time range
    if np.all(ceilo_data_interval.iloc[0].isna()):
        ceilo_data_interval.iloc[0, 0] = 0
    if np.all(ceilo_data_interval.iloc[-1].isna()):
        ceilo_data_interval.iloc[-1, 0] = 0

    if ax is None:
        fig, ax = plt.subplots(1, 1)
    else:
        fig = None
    xticks_pos = get_full_hour_indices(ceilo_data_interval.index)
    xticks = ceilo_data_interval.index[xticks_pos].round('min')
    xticks_labels = xticks.strftime("%H:%M")
    ax.plot(ceilo_data_interval.index, ceilo_data_interval, marker='.', markersize=1, linestyle='none')
    ax.legend(['channel 1', 'channel_2', 'channel_3'])
    ax.margins(x=0)
    ax.set_title('Ceilometer: Cloud base height')
    ax.set_xlabel(f'Time of the day in HH:MM ({ceilo_data.index.tz})')
    ax.set_xticks(xticks, xticks_labels, rotation=45)
    ax.set_ylabel('[m]')
    ax.set_ylim([0, 12500])
    ax.grid(visible=True, axis='both', which='both')
    return fig


def create_and_save_keogram(image_path, path_save, tz="+0100", camera_name="resolve", strip_size=7, dpi_fig=1200,
                            slice_pos=1056, increase_brightness=30, exposure_time=160, ceilometer_data_path=None):
    """
    Create and save a keogram image.

    :param image_path: The path to the directory containing the images.
    :type image_path: [str, pathlib.Path]
    :param path_save: The path to save the keogram image.
    :type path_save: [str, pathlib.Path]
    :param tz: The time zone to localize the timestamps. (default: "+0100")
    :type tz: str, optional
    :param camera_name: The name of the camera. (default: "resolve")
    :type camera_name: str, optional
    :param strip_size: The size of the strip in the keogram. (default: 7)
    :type strip_size: int, optional
    :param dpi_fig: The DPI (dots per inch) of the saved keogram image. (default: 1200)
    :type dpi_fig: int, optional
    :param slice_pos: The position of the slice in the keogram. (default: 1056)
    :type slice_pos: int, optional
    :param increase_brightness: The amount to increase the brightness of the keogram image. (default: 30)
    :type increase_brightness: int, optional
    :param ceilometer_data_path: The path to the ceilometer data file. (default: None)
    :type ceilometer_data_path: str, optional
    :raises AssertionError: If the image_path is not a directory.
    :returns: None
    """
    image_path = Path(image_path)
    path_save = Path(path_save)

    # check if image path exists
    assert image_path.is_dir(), f'{image_path} is not a directory.'

    # get camera name from image path
    if camera_name == "resolve":
        camera_name = image_path.parent.parent.name
    image_files = get_image_files(image_path, extensions=['.jpg'], substring=f'{exposure_time}.')

    # compose keogram
    keogram_array, ts_slices = compose_keogram(image_files=image_files, slice_pos=slice_pos, strip_size=strip_size)

    # set time zone
    if tz is not None:
        ts_slices = ts_slices.tz_localize(tz)

    # make image brighter
    if increase_brightness > 0:
        keogram_array = keogram_array.astype('uint16')
        keogram_array += increase_brightness
        keogram_array[keogram_array > 255] = 255
        keogram_array = keogram_array.astype('uint8')

    # plot and save keogram
    if ceilometer_data_path is not None:
        fig, ax = plt.subplots(nrows=2, ncols=1, sharex=False)
        plot_keogram(keogram_array, ts_slices, strip_size, camera_name, ax=ax[0])
        plot_ceilometer_data(ceilometer_data_path, ts_slices[0], ts_slices[-1], ax=ax[1], tz_convert=tz)
        plt.tight_layout()
    else:
        fig = plot_keogram(keogram_array, ts_slices, strip_size, camera_name)
    # Save figure
    path_save.mkdir(parents=True, exist_ok=True)
    path_save_final = path_save / f'{ts_slices[0].strftime("%Y%m%d")}_{camera_name}_Keogram.png'
    fig.savefig(path_save_final, bbox_inches='tight', dpi=dpi_fig, format='png')
    plt.close(fig)


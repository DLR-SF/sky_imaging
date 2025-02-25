# Licensed under the Apache License: http://www.apache.org/licenses/LICENSE-2.0
# For details: https://github.com/DLR-SF/sky_imaging/blob/main/NOTICE.txt

"""
Utility functions for real time applications.
"""
import requests
import io, subprocess
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from datetime import datetime
import time
from asi_core import basics


def parse_logger_data(url_cs_logger_table, timezone, name_desired_columns_cs_table=None):
    """
    This function access the table of a campbell scientific logger via its URL. All channels and the time stamp are
    retrieved from the HTML string and placed in a data frame. If desired some channels can be renamed.
    :param url_cs_logger_table: (str) url of table as string
    :param timezone: (str) desired time zone (e.g. "GMT+1" = UTC+1)
    :param name_desired_columns_cs_table: (dict) List with 2 columns and n rows. First column holds the original names of channels
                                         which shall be renamed. The second column holds the new names for the channels.
    :return df: data frame single row with n columns for each channel + timestamp index as datetime
    """
    response = requests.get(url_cs_logger_table)
    if response.status_code == 200:
        html_string = response.text
    else:
        print(f"Failed to retrieve data from CS logger. Status code: {response.status_code}")
        return
    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(html_string, 'html.parser')

    # Find the data within the table
    table = soup.find('table')
    rows = table.find_all('tr')

    # Initialize lists to store data
    columns = []
    values = []
    for row in rows:
        # Extract column names (headers) and values
        th = row.find('th')
        td = row.find('td')

        if th and td:
            column = th.get_text()
            value = float(td.get_text())  # Convert value to float if needed

            columns.append(column)
            values.append(value)

    # get time stamp
    b_element = soup.find('b', string='Record Date: ')
    timestamp = b_element.find_next_sibling(string=True)
    timestamp = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S.%f')
    timestamp = pd.Timestamp(timestamp).tz_localize(tz=basics.get_ETC_GMT_timezone(timezone))
    # Create a DataFrame
    data_dict = {'Timestamp': [timestamp]}
    data_dict.update({col: [value] for col, value in zip(columns, values)})
    df = pd.DataFrame(data_dict)
    df.set_index("Timestamp", inplace=True)

    if name_desired_columns_cs_table is not None:
        logger_keys = list(name_desired_columns_cs_table["header_logger"])
        pyranocam_values = list(name_desired_columns_cs_table["header_PyranoCam"])
        columns_mapping = list(zip(logger_keys, pyranocam_values))
        for old_name, new_name in columns_mapping:
            df.rename(columns={old_name: new_name}, inplace=True)
    return df



def parse_latest_image_ssh(asi_img_fmt, host, interval=30, verbose=False):
    """
       Function to load latest images from remote server
       :param asi_img_fmt: (str) filename format of image to load from server (absolute path).
                                Requires identifier for timestamp (dt, e.g. {dt:%Y%m%d%H%M%S})
       :param host: (str) host name to be accessed via ssh
       :param interval: (int, optional, default=30) Interval for image acquisition
       :return img, dt: image + timestamp (datetime64)
    """

    log = print if verbose else basics.ignore

    # wait for image initiation time
    now = np.datetime64('now')

    # Now wait for the slot
    while now.item().second % interval > 0:
        time.sleep(1)
        now = np.datetime64('now')
        continue
    dt = now

    # Load raw asi image from contabo
    host_img = asi_img_fmt.format(dt=dt.item())
    log(f'Wait for image {host_img}')
    exist_img = False

    # Allow some waiting time for image to be downloaded at Host
    for i in range(interval - 5):
        try:
            image_distorted_bytes = io.BytesIO(subprocess.check_output(['ssh', host, 'cat', f'{host_img}']))
            return image_distorted_bytes, dt
        except subprocess.CalledProcessError:
            time.sleep(1)
            continue
        except Exception as e:
            print(host_img, e)

    if not exist_img:
        log('No image')
        return None, None


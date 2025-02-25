# Licensed under the Apache License: http://www.apache.org/licenses/LICENSE-2.0
# For details: https://github.com/DLR-SF/sky_imaging/blob/main/NOTICE.txt

"""
Configuration of logging for asi_core.
"""

import logging


def configure_logging(log_file=None, log_level=logging.INFO):
    """Set basic config of logging module."""
    if log_file is None:
        logging.basicConfig(
            format="{asctime} - {levelname} - {message}",
            style="{",
            datefmt="%Y-%m-%d %H:%M",
            level=log_level,
        )
    else:
        logging.basicConfig(
            format="{asctime} - {levelname} - {message}",
            style="{",
            datefmt="%Y-%m-%d %H:%M",
            level=log_level,
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_file, mode='w', encoding='utf-8')
            ]
        )
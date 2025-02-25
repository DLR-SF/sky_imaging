# Licensed under the Apache License: http://www.apache.org/licenses/LICENSE-2.0
# For details: https://github.com/DLR-SF/sky_imaging/blob/main/NOTICE.txt

"""
Configuration for ASI tasks

The configuration is expected to be stored as `pyranocam.yaml` in the current directory.
It should be formatted as YAML file.
"""
import pathlib
import typing as t

import yaml


_config = None


def load_config(filename: t.Union[str, pathlib.Path, None] = None) -> None:
    """
    Load PyranoCam configuration from the given filename or the default location.

    :param filename: Where to load configuration from. If nothing is given, ``pyranocam.yaml`` is asserted.
    """
    if filename is None:
        filename = "pyranocam.yaml"

    with open(filename, "r") as config_file:
        global _config
        _config = yaml.load(config_file, yaml.Loader)


def get(key: t.Optional[str] = None, default: t.Any = None) -> t.Any:
    """
    Retrieve a configuration value.

    If configuration is not yet loaded, this is done automatically.

    :param key: Name of the configuration section to load. If nothing is passed, the whole config dict will be returned.
    :param default: Default value to return if the key is not present.
    :return: The requested configuration section.
    """
    if _config is None:
        load_config()

    if key is None:
        return _config
    else:
        return _config.get(key, default)

# Licensed under the Apache License: http://www.apache.org/licenses/LICENSE-2.0
# For details: https://github.com/DLR-SF/sky_imaging/blob/main/NOTICE.txt

"""
Global constants of asi_core
"""

## General assumptions

# if you request all images or other files from a folder structure according to our convention, replace these patterns
# with wildcards. May need to be extended if exotic folder structure is used
IGNORED_PATTERNS_SUBDAY = [{'pattern': '{timestamp:%H}', 'substitution': r'*'},
                           {'pattern': r'({timestamp:[%\w-]+)%H[%MSf_-]*(})',
                            'substitution': lambda m: m.groups()[0] + '*' + m.groups()[1]},
                           {'pattern': '{exposure_time:d}', 'substitution': '[0-9]*'}]

FSTRING_RE = [{'formatter': r':d', 're': r':\d+'},
              {'formatter': r'\d+?d', 're': r'\d+'},
              {'formatter': '%Y', 're': r'\d{4}'},
              {'formatter': '%y', 're': r'\d{2}'},
              {'formatter': '%m', 're': r'\d{2}'},
              {'formatter': '%d', 're': r'\d{2}'},
              {'formatter': '%H', 're': r'\d{2}'},
              {'formatter': '%M', 're': r'\d{2}'},
              {'formatter': '%S', 're': r'\d{2}'}]

# image files are expected to have one of these file endings
IMAGE_EXTENSIONS = ('.png', '.jpg')


## Matlab parameter names
# Matlab datenum is a floating point number counting days since year 0
# whereas numerical unix time is represented as seconds since 01.01.1970
MATLAB_TIMEDELTA = 719529  # number of days that need to be subtracted for unix time conversion
MATLAB_SUNEL = 'EL'
MATLAB_SUNAZ = 'AZ'
MATLAB_LAT = 'latitudeDegN'
MATLAB_LON = 'longitudeDegE'
MATLAB_ALT = 'altitude'
MATLAB_TZ = 'timezone'
MATLAB_TAMB = 'Tamb'
MATLAB_PAMB = 'p'
MATLAB_BP = 'BP'
MATLAB_RH = 'relHum'
MATLAB_WS = 'WS'
MATLAB_WD = 'WD'
MATLAB_DATENUM = 'numericDate'
MATLAB_SDN = 'SDN'
MATLAB_DNI = 'DNI'
MATLAB_DNI2 = 'DHI_Ref'
MATLAB_GHI = 'GHI'
MATLAB_GHI2 = 'GHI_Ref'
MATLAB_GHI_CALIB = 'GHI_CALIB'
MATLAB_DHI = 'DHI'
MATLAB_DHI2 = 'DHI_Ref'
MATLAB_GTI = 'GTI'
MATLAB_GTI_S30 = 'GTI_t30_S'
MATLAB_GTI_S40 = 'GTI_t40_S'
MATLAB_GTI_E45 = 'GTI_t45_E'
MATLAB_Albedo = 'Albedo'
MATLAB_DNIVARCLASS = 'varClass'

## Python parameter names
PYTHON_DT = 'timestamp'
PYTHON_LAT = 'latitude'
PYTHON_LON = 'longitude'
PYTHON_ALT = 'altitude'
PYTHON_SUNEL = 'sun_el'
PYTHON_SUNAZ = 'sun_az'
PYTHON_TZ = 'timezone'
PYTHON_TAMB = 't_amb'
PYTHON_PAMB = 'p_amb'
PYTHON_RH = 'rel_humid'
PYTHON_WS = 'wind_speed'
PYTHON_WD = 'wind_dir'
PYTHON_DNI = 'dni'
PYTHON_GHI = 'ghi'
PYTHON_GHI_CALIB = 'ghi_calib'
PYTHON_DHI = 'dhi'
PYTHON_GTI = 'gti'
PYTHON_GTI_S30 = 'gti_t30_south'
PYTHON_GTI_S40 = 'gti_t40_south'
PYTHON_GTI_E45 = 'gti_t45_east'
PYTHON_Albedo = 'albedo'
PYTHON_DNIVARCLASS = 'dni_var_class'

PYTHON_RENAMING = {
    MATLAB_LAT: PYTHON_LAT,
    MATLAB_LON: PYTHON_LON,
    MATLAB_ALT: PYTHON_ALT,
    MATLAB_SUNEL: PYTHON_SUNEL,
    MATLAB_SUNAZ: PYTHON_SUNAZ,
    MATLAB_TZ: PYTHON_TZ,
    MATLAB_TAMB: PYTHON_TAMB,
    MATLAB_PAMB: PYTHON_PAMB,
    MATLAB_BP: PYTHON_PAMB,
    MATLAB_RH: PYTHON_RH,
    MATLAB_WS: PYTHON_WS,
    MATLAB_WD: PYTHON_WD,
    MATLAB_DNI: PYTHON_DNI,
    MATLAB_DNI2: PYTHON_DNI,
    MATLAB_GHI: PYTHON_GHI,
    MATLAB_GHI2: PYTHON_GHI,
    MATLAB_GHI_CALIB: PYTHON_GHI_CALIB,
    MATLAB_DHI: PYTHON_DHI,
    MATLAB_DHI2: PYTHON_DHI,
    MATLAB_GTI: PYTHON_GTI,
    MATLAB_GTI_S30: PYTHON_GTI_S30,
    MATLAB_GTI_S40: PYTHON_GTI_S40,
    MATLAB_GTI_E45: PYTHON_GTI_E45,
    MATLAB_Albedo: PYTHON_Albedo,
    MATLAB_DNIVARCLASS: PYTHON_DNIVARCLASS,
}
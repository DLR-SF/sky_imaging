# Quick start guide on URL image receiver data acquisition program

## Requirements

This DAQ needs to be installed on a computer (linux or windows) located in the same network as the 
camera you want to control. (The camera must be accessible e.g. in your browser.)

## Install the conda env

- Make sure you have miniforge or similar installed
- Create the asi-core conda environment (location of the asi-core repository may be different on your computer):
  
    `conda env create -f C:\git\sfpt_meteo_nowcasting\asi-core\environment.yml`

    Using a conda environment is not strictly needed. Feel free to install the package 
    in another environment with python 3.10 installed.

- Activate the conda environment 

  `conda activate asi_core`

- Install the asi-core package

    `pip install -e C:\git\sfpt_meteo_nowcasting\asi-core`


## Set your local configuration

- Create config file for your data acquisition task based on this template

  `C:\git\sfpt_meteo_nowcasting\asi-core\asi_core\daq\asi_daq_cfgs\template.yaml`

- Set the URL of the camera in your local network e.g.

  `url_cam: 'http://10.21.202.145'`

 
- Set the path to which you want to store your results:
 
  `storage_path: 'C:/data/test_daq/server_folder'`


- Set the working directory of the DAQ program. In this folder log files will be stored
 
  `daq_working_dir: 'C:/data/test_daq/asi-core/log_folder'`


- Set the coordinates of the camera e.g.

  `location: {'lat': 37.09415129, 'lon': -2.35478069, 'alt': 500}`



## Run the DAQ

- Activate the conda environment 

  `conda activate asi_core`


- Run the daq providing your config file as argument, e.g.:

  `python C:\git\sfpt_meteo_nowcasting\asi-core\asi_core\daq\url_image_receiver.py -c "C:\git\sfpt_meteo_nowcasting\asi-core\asi_core\daq\asi_daq_cfgs\Cloud_Cam_Metas.yaml"`


- You can adapt the following script to do the job

  `C:\git\sfpt_meteo_nowcasting\asi-core\asi_core\daq\run_all_GUIs.bat`


## Further settings

The DAQ uses a number of default settings which can be changed if needed when initializing a Receiver instance.
Comment in parameters in the Process section of the config file if needed. The default parameters are suited for Q26.

# How to run the ASI self-calibration?


## Installation
- Make sure you have miniforge or similar installed
- Clone the asi_core or sky_imaging repository
- Change the working directory into the repository folder 
- Create the asi-core conda environment:
  
    `conda env create -f environment.yml`

    Using a conda environment is not strictly needed. Feel free to install the package 
    in another environment with python 3.10 installed.

- Activate the conda environment 

  `conda activate asi_core`

- Install the asi-core package

    `pip install -e .`



## Preparation

Open the conda environment in which asi-core is installed.

Create a documentation folder in a suited location.

Copy the following file to your documentation folder:

`asi_core/calibration/self_calibration_cfg.yaml`

Change the working dir to the documentation folder.



## Prepare a preliminary "camera_data" file

In asi_core, each camera "sample" is described by an own camera_data file. Template see:

`asi_core/camera_data/ASI_Template.yaml`

Prepare a camera_data file based on this template with the information available before the calibration. 
- Make sure the camera name matches with your camera. 
- Make sure the period between the dates 'mounted' and 'demounted' includes your calibration period. 
- Copy `internal_calibration` from a camera of the same type e.g. Mobotix Q26. If you are using a completely new camera 
type, the polynomial coefficients "ss" taken from another camera type should be scaled to the new camera's image 
resolution. For some cameras it might be required to perform the calibration according to Scaramuzza for one sample of 
that camera model to receive suited start values for cameras of that type. In this case, you would notice large 
deviations remaining after the calibration.
- Make sure the specified width and height correspond to the actual width and height of the sky images. Diameter can be estimated as the smaller image side length
- The center of the camera lens (xc, yc) should be estimated as half of the image width and height respectively
- If possible each camera should be installed horizontally leveled, with the Sun in the upper image part at solar noon.
In that case set 

  `external_orientation: [0.,   3.14,   1.57]` (northern hemisphere) or

  `external_orientation: [0.,   3.14,   -1.57]` (southern hemisphere).

  (This is only a precaution. The calibration procedure should be able also to work with strongly 
deviating external orientations.)
- Specify your camera's exposure times under exposure settings exposure_settings/exposure_times.
  - day and night indicate the exposure times (as list) used during day and night time respectively
  - if taking image series, you can specify multiple exposure times. Images of the lowest exposure time specified will 
    be used for the calibration.
  - if you use WDR imaging or no fixed exposure time, set [0] as exposure time. 
  - In any case avoid that the tool will find multiple images (of the same exposure time if applicable) for the same 
    timestamp (see img_path_structure in the config file section below). 
    exposure_settings/tolerance_timestamp controls the tolerance between requested and found image timestamp 
    (from image name) 
- Specify the path to the camera mask (see infos on mask below)

As best practice, create a folder camera_cata in the documentation folder for your calibration. 
Store your camera_data file in that folder.


### Mask creation
You will need to create a camera mask. For this task, an automatic tool exists at DLR. With this
calibration tool only a simple manual tool based on a graphical user interface is provided to create the camera mask:

`asi_core/calibration/mask_creation.py`

To use the tool copy the following config file to your working directory:

`asi_core/calibration/mask_creation_cfg.yaml`

and in the file specify the path to an image of the current camera in under:

`ObstacleMaskDetection/img_path`

Install opencv **with head** (by default our pyproject.toml installs
"opencv-python-headless", i.e. GUI needed here not installed to save resources)

Then run the mask creation tool, on a computer with desktop:

`python <path to repository>/asi_core/calibration/mask_creation.py -c mask_creation_cfg.yaml`

The usage of the tool is described at the top of the GUI window. The basic idea of the tool is to draw polygons which 
indicate areas that are added or removed from an initial rough guess of the camera mask.

In your working directory, you will receive 2 files:
- `mask_*.mat`
- `masked_*.jpg` (only to check/ document the result visually)

As best practice, store the .mat file in your camera_data folder in a new sub folder camera_masks. 
Specify the relative path from camera_data file to camera mask in the camera_data file (entry `camera_mask_file`)


## Config file

Adapt `self_calibration_cfg.yaml` to your current calibration task.

The config file contains comments to be self-explanatory. (If not contact niklas.blum@dlr.de) Usually, you will need to
adapt the following:
- camera_name
- camera_data_dir -- comment this in to use the camera data folder you created in your working dir
- img_path_structure -- path to the image of each timestamp and exposure time with placeholders for date/ time and 
- mode -- defines what task is performed calibration and validation or only one of the two. Additionally, if orb positions can either be detected from images are be taken from a csv file which was created in advance. `calibrate_validate_from_images` should be used as default.
- last_timestamp
- last_timestamp_validation
- moon_detection/number_days
- sun_validation/number_days
- The remaining parameters will usually remain unchanged at least for Mobotix.

The period included in the calibration should be long enough to have orb observations in a wide 
range of sky areas regarding azimuth and zenith angle. Note that depending on location and season this can sometimes be
difficult. In the best case use one of the following:
- Moon positions from at least half a year between summer and winter solstice
- Moon positions from at least one moon phase in winter
- Moon positions from at least one moon phase in summer AND sun positions from one month in summer

A sampling time of 10 minutes will usually be enough to get all orb positions. If you want to save resource use a larger
interval. If you work in very cloudy conditions you may want to detect Sun and Moon in every short cloud-free period, 
in that case consider reducing the sampling time. The visualization received from the calibration will help to estimate
if a sufficient number of orb positions distributed rather homogeneously over a wide range of azimuth and zenith angles
has been detected.



## Execute calibration program

In the terminal run (path will be different on linux)

`python <path_to_repository>/asi_core/calibration`

If you don't include a field 'last_timestamp' in the config file, the self-calibration will use the previous full-moon date + 6 days as last timestamp. 

The calibration will take some time to run depending on the data connection to your image storage location, 
your computer's ressources and the number of timestamps included in the calibration. While the calibration is running a
log file `geometric_calib_processed_<start date and time>.log` is created. Check this file to see how the calibration
progresses. You will see that the calibration's orb detection loops through all timestamps once for the calibration once
for the validation. Additionally, you will see that the iterative center detection usually takes some time.

## Check the results

In the documentation folder you will receive the following files:

| Name                                                       | Meaning                                                                                                                                          |
|------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------|
| `calib_<camera_name>_<start>_<end>.yaml`                   | contains all results of the calibration. These should be transfered to the camera_data yaml`                                                     |
| `calib_<camera_name>_<start>_<end>.mat`                    | same as the yaml file above. Stored for legacy matlab tools.                                                                                     |
| `moon_observations_<start>_<end>.csv`                      | contains all moon observations included in the validation or calibration. The observations can be used to reproduce the calibration/ validation. | 
| `sun_<start>_<end>.csv`                                    | contains all sun observations included in the validation or calibration. The observations can be used to reproduce the calibration/ validation.  | 
| `calibrated_observations_<start>_<end>.csv`                | contains all orb observations included in the calibration. In this file the final calibration parameters have already been applied.              | 
| `validation_observations_<start>_<end>.csv`                | contains all orb observations included in the validation. In this file the final calibration parameters have already been applied.               | 
| `calibrated_observations_<start>_<end>.png`                | visualizes the coincidence of expected and found orb positions in the calibration.                                                               |
| `validation_observations_<start>_<end>.png`                | visualizes the coincidence of expected and found orb positions in the validation.                                                                |
| `azimuth_matrix_<camera_name>_<timestamp_processed>.npy`   | contains a matrix of the azimuth angle viewed by each pixel                                                                                      |
| `elevation_matrix_<camera_name>_<timestamp_processed>.npy` | contains a matrix of the elevation angle viewed by each pixel                                                                                    |

`<start>`, `<end>` indicate the timestamps bounding the period included in the calibration.

In this case Moon positions were used for the calibration and Sun positions for the validation.
To check the results, open both image files:

![calibrated_observations_20230205000000_20230804000000.png](media_self_calib/calibrated_observations_20230205000000_20230804000000.png)

You should see that most moon positions (expected and detected) coincide well. This is expressed by the red and blue 
dots and by the RMSD values printed. Usually we have a small number of outliers. In outlier cases surrounding lights 
were detected as Moon. After filtering for outliers (1% of the data points), you should see a very small RMSD of 
typically less than 2 pixels. RMSD values larger than 4 pixels will indicate a rather low quality of your calibration. 
At the same time you should see a large number of visualized data points spread over a wide range of azimuth and 
elevation angles in one half of the hemisphere. If these conditions are not fulfilled, your calibration may be 
over-fitted to the sky region from which the observations were received.

![validation_observations_20230807063000_20240202163000.png](media_self_calib/validation_observations_20230807063000_20240202163000.png)



Accordingly, you can evaluate the visualization of the Sun positions. In this case Sun positions were used for 
validation. This means they were only included to check if also for Sun positions small deviations between astronomic 
expectation and image processing are attested. In this case slightly larger deviations are possible if your validation 
interval includes a high fraction of turbid or cloudy situations in which the sun disk may still appear roundish while 
being disturbed by these influences. Usually you should receive an RMSD of around 3 pixels.


## Refine the results

If you calibrate using Sun positions, stronger deviations in the orb positions from image processing are possible. 
This is caused in particular by lens soiling, increased turbidity, clouds near the sun, cirrus clouds in general. If 
calibrating with sun positions, it might sometimes be required to manually filter out low quality images.

For this you can run the calibration once more adapting as follows:

````
Calibration:
	mode: calibrate_from_csv  
	sort_out_imgs_manually: True
	path_orb_observations: '<path_to_orb_observations_for_calibration>.csv'  # specify path to csv with orb observations from first run here
````

This will lead you through a dialog which requests you to delete invalid images from the subfolder used_imgs in your 
working directory (copied from the original image folder). Erased images will then be excluded from the calibration.


## Store results 

- Store results on the server: \\asfs03/Group/PROJECTS/MeteoProj/Cameras/CloudCamera/CloudCamCalibration
- create a folder for each camera (e.g. Cloud_Cam_PVotSky)
- create a folder for each calibration (e.g. 202402_XX)
- Store all calibration results here (see table ## Check the results)
- If the calibration is suited, add it to camera_data  

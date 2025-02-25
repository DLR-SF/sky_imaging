# Licensed under the Apache License: http://www.apache.org/licenses/LICENSE-2.0
# For details: https://github.com/DLR-SF/sky_imaging/blob/main/NOTICE.txt

"""
This module provides functions to create videos from all-sky images.
"""

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from fastcore.parallel import parallel

from asi_core.camera import AllSkyImager, load_camera_data
from asi_core.io import load_image, load_images
from asi_core.visualization.masking import create_saturation_mask_image


OPENCV_VIDEO_CODECS = {
    '.mp4': 'mp4v',
    '.avi': 'XVID'
}


class AllSkyVideo():
    """
    Handles the creation of videos from all-sky images.
    """
    def __init__(self, output_dir, asi_files, camera_name=None, fps=10, format='.mp4', asi_tfms=None):
        """
        Initialize the AllSkyVideo class.

        :param output_dir: Directory where the output video will be saved.
        :param asi_files: Pandas Series containing image file paths indexed by timestamps.
        :param camera_name: Name of the camera used (optional).
        :param fps: Frames per second for the video.
        :param format: Video file format (e.g., '.mp4', '.avi').
        :param asi_tfms: Optional transformations for all-sky images.
        """
        self.output_dir = Path(output_dir)
        self.asi_files = asi_files.sort_index()
        self.timestamps = asi_files.index
        self.camera_name = camera_name
        self.fps=fps
        self.format = format
        self.fourcc=cv2.VideoWriter_fourcc(*OPENCV_VIDEO_CODECS[self.format])
        self.asi_tfms = asi_tfms
        self.sky_imager = None

    def update_sky_imager_config(self, timestamp):
        """Updates the sky imager configuration based on the given timestamp."""
        if self.camera_name is None:
            pass
        elif (self.sky_imager is None or
              not (self.sky_imager.start_recording < timestamp < self.sky_imager.end_recording)):
            camera_data = load_camera_data(camera_name=self.camera_name, timestamp=timestamp)
            self.sky_imager = AllSkyImager(camera_data, tfms=self.asi_tfms)

    def create_video(self, filename, timestamps=None):
        """Creates a video from the given image timestamps."""
        assert Path(filename).suffix == self.format, f'Incorrect file format, should be {self.format}.'
        if timestamps is not None:
            asi_files = self.asi_files.loc[timestamps]
        else:
            asi_files = self.asi_files
        images = self.process_images(asi_files)
        create_video_from_images(
            images=images,
            output_path=str(self.output_dir / filename),
            fps=self.fps,
            fourcc=self.fourcc
        )

    def process_images(self, asi_files):
        """
        Processes images by applying transformations and preparing them for video creation.

        :param asi_files: Pandas Series containing image file paths.
        :return: List of processed images.
        """
        self.update_sky_imager_config(timestamp=asi_files.index[0])
        images = load_images(img_files=asi_files)
        if self.sky_imager is not None:
            assert self.sky_imager.end_recording > asi_files.index[-1], \
                'Timespan covers multiple sky imager configurations.'
            images = self.sky_imager.transform(images)
        return images

    def create_daily_videos(self, filename_prefix='', dates=None, num_workers=0):
        """
        Creates daily videos based on available timestamps.

        :param filename_prefix: Prefix for the output video filename.
        :param dates: List of dates to generate videos for (optional).
        :param num_workers: Number of parallel workers for processing.
        """
        if dates is None:
            dates = np.unique(self.timestamps.date)
        if num_workers > 0:
            tasks = [{'filename': f'{filename_prefix}_{date}{self.format}',
                      'timestamps': self.timestamps[self.timestamps.date == date]} for date in dates]
            for _ in tqdm(
                    parallel(self._create_video_task, tasks, n_workers=min(len(tasks), num_workers)),
                    desc="Overall Progress"):
                pass
        else:
            for date in tqdm(dates, total=len(dates)):
                timestamps = self.timestamps[self.timestamps.date == date]
                filename = f'{filename_prefix}_{date}{self.format}'
                self.create_video(filename=filename, timestamps=timestamps)

    def _create_video_task(self, task):
        """Helper method to unpack task and call create_video"""
        self.create_video(task['filename'], task.get('timestamps'))


class SaturationMaskSkyVideo(AllSkyVideo):
    """
    Generates videos of all-sky images with a mask overlay of saturated pixels.
    """
    def __init__(self, output_dir, asi_files, camera_name, fps=10, format='.mp4', asi_tfms=None):
        """Initialize instance as for AllSkyVideo class"""
        super().__init__(output_dir, asi_files, camera_name=camera_name, fps=fps, format=format, asi_tfms=asi_tfms)

    def process_images(self, asi_files):
        """
        Processes images by applying a saturation mask overlay.

        :param asi_files: Pandas Series containing image file paths.
        :return: List of processed images with saturation masks applied.
        """
        self.update_sky_imager_config(timestamp=asi_files.index[0])
        assert self.sky_imager.end_recording > asi_files.index[-1], \
            'Timespan covers multiple sky imager configurations.'
        camera_mask = self.sky_imager.camera_mask
        if camera_mask is not None:
            camera_mask = self.sky_imager.transform(camera_mask)
        images = []
        for asi_file in tqdm(asi_files, total=len(asi_files)):
            img = load_image(asi_file)
            img = self.sky_imager.transform(img)
            font_size = int(round(img.shape[1] / 20))
            img = create_saturation_mask_image(img, camera_mask, font_size=font_size, asarray=True)
            images.append(img)
        return images


def create_video_from_images(images, output_path, fps=10, fourcc=0, rgb_format=True):
    """
    Creates video from images.

    :param images: list of images in numpy format.
    :param output_path: path of output video.
    :param fps: frames per second.
    """
    height, width, layers = images[0].shape
    video = cv2.VideoWriter(filename=output_path, fourcc=fourcc, fps=fps, frameSize=(width, height))
    for image in images:
        if rgb_format:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        video.write(image)
    video.release()

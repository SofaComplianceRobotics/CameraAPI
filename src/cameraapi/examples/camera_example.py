#!/usr/bin/env -S uv run --script

import time
import logging
import os
import sys

import cameraapi

sys.path.append(os.path.dirname(os.path.realpath(__file__))+'/..')
from cameraapi import Camera
from cameraapi._logging_config import logger


def main(camera: Camera):

    # emio.calibrate()  # calibrate the camera if needed

    while camera.is_running:
        try:
            camera.update() # update the camera frame and trackers

            print("-"*20)
            logger.info(f"Camera parameters: {camera.parameters}")
            logger.info(f"Camera show: {camera.show_frames}")
            logger.info(f"Camera tracking: {camera.track_markers}")
            logger.info(f"Camera compute point cloud: {camera.compute_point_cloud}")
            logger.info(f"Camera is running: {camera.is_running}")
            logger.info(f"Count tracker: {len(camera.trackers_pos)}")
            logger.info(f"Trackers positions: {camera.trackers_pos}")
            logger.info(f"Point cloud shape: {camera.point_cloud.shape}")
            logger.info(f"HSV Frame shape: {camera.hsv_frame.shape}")
            logger.info(f"Mask Frame shape: {camera.mask_frame.shape}")

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received.")
            break
        except Exception as e:
            logger.exception(f"Error during communication: {e}")
            break


if __name__ == "__main__":
    try:
        logger.info("Starting EMIO Camera test...")

        logger.info("List of available cameras\n"+str(cameraapi.listCameras()))

        logger.info("Opening and configuring EMIO Camera...")

        emio = Camera(show=True, track_markers=True, compute_point_cloud=True)
        emio.fps = 30 # sets the fps to 30. Default is 60 and can only be one of 30. 60 or 90fps
        emio.depth_max = 600 # sets the maximum depth to 600mm. Default is 430mm
        emio.depth_min = 0 # sets the minimum depth to 0mm. Default is 2mm

        if emio.open(): # This will open the first available Realsense camera

            logger.info(f"Emio camera {emio.camera_serial} opened.")
            logger.info("Running main function...")
            main(emio)

            logger.info("Main function completed.")
            logger.info("Closing Emio API...")

            emio.close()

            logger.info("EMIO API closed.")
    except Exception as e:
        logger.exception(f"An error occurred: {e}")

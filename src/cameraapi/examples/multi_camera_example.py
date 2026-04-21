#!/usr/bin/env -S uv run --script

import time
import logging
import os
import sys

sys.path.append(os.path.dirname(os.path.realpath(__file__))+'/..')
from cameraapi import Camera
from cameraapi._logging_config import logger


def main(cam1: Camera,cam2: Camera):

    # cam1.calibrate()  # calibrate the camera if needed

    while cam1.is_running:
        try:
            cam1.update() # update the camera frame and trackers
            cam2.update() # update the camera frame and trackers

            print("-"*20)
            logger.info(f"Count tracker: {len(cam1.trackers_pos)},{len(cam2.trackers_pos)}")
            logger.info(f"Trackers positions: {cam1.trackers_pos},{cam2.trackers_pos}")
            
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received.")
            break
        except Exception as e:
            logger.exception(f"Error during communication: {e}")
            break


if __name__ == "__main__":
    try:
        logger.info("Starting Camera test...")

        logger.info("List of available cameras\n"+str(Camera.listCameras()))

        logger.info("Opening and configuring Camera...")

        cam1 = Camera(show=True, track_markers=True, compute_point_cloud=True)
        cam1.fps = 30 # sets the fps to 30. Default is 60 and can only be one of 30. 60 or 90fps
        cam1.depth_max = 600 # sets the maximum depth to 600mm. Default is 430mm
        cam1.depth_min = 0 # sets the minimum depth to 0mm. Default is 2mm


        cam2 = Camera(show=False, track_markers=True, compute_point_cloud=True)
        cam2.fps = 30 # sets the fps to 30. Default is 60 and can only be one of 30. 60 or 90fps
        cam2.depth_max = 600 # sets the maximum depth to 600mm. Default is 430mm
        cam2.depth_min = 0 # sets the minimum depth to 0mm. Default is 2mm
        cameras = cam1.listCameras()


        if cam1.open(cameras[1]): # This will open the first available Realsense camera
            logger.info(f"Camera 1 {cam1.camera_serial} opened.")
            logger.info("Running main function...")
            cam2.open(cameras[0])
            logger.info(f"Camera 2 {cam2.camera_serial} opened.")
            logger.info("Running main function...")
            
            main(cam1,cam2)

            logger.info("Main function completed.")
            logger.info("Closing API...")

            cam1.close()

            logger.info("API closed.")
    except Exception as e:
        logger.exception(f"An error occurred: {e}")

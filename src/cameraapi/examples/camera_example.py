#!/usr/bin/env -S uv run --script

import time
import logging
import os
import sys

from cameraapi.depthcamera import DepthCamera, list_cameras

sys.path.append(os.path.dirname(os.path.realpath(__file__))+'/..')
from cameraapi._logging_config import logger


def main(camera: DepthCamera):

    print("Camera parameters:", camera.parameters)

    while camera:
        try:
            camera.update() # update the camera frame and trackers

            print("-"*20)
            logger.info(f"Camera parameters: {camera.parameters}")
            logger.info(f"Camera show: {camera.show_video_feed}")
            logger.info(f"Camera tracking: {camera.tracking}")
            logger.info(f"Camera compute point cloud: {camera.compute_point_cloud}")
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

        logger.info("List of available cameras\n"+str(list_cameras()))

        logger.info("Opening and configuring DepthCamera...")


        import numpy as np
        corners = np.array([[ -49.497475, -230.      ,    0.      ],
                            [   0.      , -230.      ,  -49.497475],
                            [  49.497475, -230.      ,    0.      ],
                            [   0.      , -230.      ,   49.497475]])

        cam = DepthCamera(show_video_feed=True, tracking=True, compute_point_cloud=True)
        cam.fps = 30 # sets the fps to 30. Default is 60 and can only be one of 30. 60 or 90fps
        cam.depth_max = 600 # sets the maximum depth to 600mm. Default is 430mm
        cam.depth_min = 0 # sets the minimum depth to 0mm. Default is 2mm

        if cam.open(): # This will open the first available Realsense camera

            logger.info(f"Depth camera {cam.camera_serial} opened.")
            logger.info("Running main function...")
            main(cam)

            logger.info("Main function completed.")
            logger.info("Closing API...")

            cam.close()

            logger.info("API closed.")
        else:
            logger.error("Failed to open camera.")
    except Exception as e:
        logger.exception(f"An error occurred: {e}")

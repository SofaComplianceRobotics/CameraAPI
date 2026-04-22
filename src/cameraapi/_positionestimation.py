import logging

import numpy as np
import cv2 as cv
import json
import os
import csv
from pathlib import Path
from shutil import copyfile

from cameraapi._logging_config import logger


CONFIG_DIR = Path.home().joinpath(".config", "cameraapi")

DEFAULT_CONFIG_FILE = Path(__file__).parent.joinpath("config/cameraparameter.json")
CONFIG_FILENAME = CONFIG_DIR.joinpath("config/cameraparameter.json")
if not CONFIG_FILENAME.exists():
    # copy the default config file from the package to the config directory
    CONFIG_DIR.joinpath("config").mkdir(parents=True, exist_ok=True)
    copyfile(DEFAULT_CONFIG_FILE, CONFIG_FILENAME)

DEFAULT_CALIBRATION_FILE = Path(__file__).parent.joinpath("config/camera_2d_points.csv")
CALIBRATION_FILENAME = CONFIG_DIR.joinpath("config/camera_2d_points.csv")
if not CALIBRATION_FILENAME.exists():
    # copy the default calibration file from the package to the config directory
    CONFIG_DIR.joinpath("config").mkdir(parents=True, exist_ok=True)
    copyfile(DEFAULT_CALIBRATION_FILE, CALIBRATION_FILENAME)

ARUCO_MARKER_DEFAULT_ID = 672

COUNT_CALIBRATION_POINTS = 9 # Number of points in the calibration board (4 corners + 4 middle points + 1 center)


def compute_transform_from_pointclouds(image_cloud:np.ndarray, absolute_cloud:np.ndarray)  -> tuple[np.ndarray, np.ndarray]:
    """
    Find the translation vector and the rotation matrix between 2 points clouds

    Based on https://nghiaho.com/?page_id=671
    

    Args:
        image_cloud: numpy.ndarray
            The points cloud in the camera space

        absolute_cloud: numpy.ndarray
            The points cloud in the our frame space space

    Return:
        R: numpy.ndarray
            The rotation matrix between the clouds

        t: numpy.ndarray 
            The translation vector between the clouds
    """
    assert image_cloud.shape == absolute_cloud.shape
    N = image_cloud.shape[0]

    centroid_A = np.mean(image_cloud, axis=0)
    centroid_B = np.mean(absolute_cloud, axis=0)

    AA = image_cloud - centroid_A
    BB = absolute_cloud - centroid_B

    H = AA.T @ BB
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = centroid_B.T - R @ centroid_A.T
    return R, t

def image_pixel_to_mm(depth: float, pixel_x: int, pixel_y: int, camera_intrinsics:object) -> list[float]:
    """
    Convert the depth and image point information to metric coordinates in camera space.

    Based on https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html

    Args:
        depth: double
            The depth value of the image point

        pixel_x: double
            The x value of the image coordinate

        pixel_y: double
            The y value of the image coordinate

        camera_intrinsics : object
            The intrinsic values of the realsesnse camera See https://intelrealsense.github.io/librealsense/python_docs/_generated/pyrealsense2.intrinsics.html

    Return:
    X : double
            The x value in mm
    Y : double
            The y value in mm
    Z : double
            The z value in mm

    """

    X = ((pixel_x - camera_intrinsics.ppx) / camera_intrinsics.fx) * depth
    Y = ((pixel_y - camera_intrinsics.ppy) / camera_intrinsics.fy) * depth
    return [X, Y, depth]
   

class PositionEstimation:
    """
    This class is used to calculate the real world coordinates based on the pixel coordinates and the depth

    The calibration process is based on the detection of a single ArUco marker with a known position in the world space and position.
    You need to provide the intrinsic parameters of the camera, the ID of the ArUco marker to be detected and the size and position of the marker in the world space.
    The default marker ID is 672, but it can be changed at initialization. 
    The calibration process will run until the marker is detected and the camera-to-world transform is computed, 
    or until a timeout of 5 minutes is reached.

    A calibration file is created during the calibration process, and the camera-to-world transform is computed 
    based on the last calibration data.

    The calibraiton points are defined as the 4 corners of the marker, the middle points between the corners and 
    the center of the marker.

    During calibration, the corners of the marker are detected and their positions in the camera space are 
    calculated based on their pixel coordinates and depth values.
    
    You can choose to aggregate the corners positions over multiple frames, the mean value is then used for 
    the calibration, or to use only the corners positions of the current frame for the calibration.

    """

    def __init__(self, cameraintrinsinc, aruco_corners: np.ndarray=None, aruco_marker_id=ARUCO_MARKER_DEFAULT_ID, calibration_file: str=CALIBRATION_FILENAME, config_file: str=CONFIG_FILENAME) -> None:
        """
        Initialize the position estimation class
        
        Args:
            cameraintrinsinc: object
                The intrinsic values of the realsesnse camera See https://intelrealsense.github.io/librealsense/python_docs/_generated/pyrealsense2.intrinsics.html

            aruco_corners: np.ndarray
                The corners of the ArUco marker in the world space, defined as a 4x3 array (4 corners with x,y,z coordinates). 
                The order of the corners should be the same as the order of the corners detected in the image (top-left, top-right, bottom-right, bottom-left)

            aruco_marker_id: int
                The ID of the ArUco marker to be detected for calibration (default is 672)
        """
        self.calibration_points=np.zeros((COUNT_CALIBRATION_POINTS, 3))
        self.R=np.zeros((9,3))
        self.t=np.zeros((3))
        self.intr= cameraintrinsinc if cameraintrinsinc else None
        self.aruco_corners = aruco_corners
        self.aruco_marker_id = aruco_marker_id
        self.calibration_file = calibration_file
        self.config_file = config_file

        self.points = []
        self.trackers_pos = []
        self.initialized = False
        self.count_calibration_frames = 0

        # calibration
        if self.aruco_corners is not None and self.aruco_corners.shape == (4, 3):
            self.calibration_points[0] = aruco_corners[0]  # top-left corner of the marker
            self.calibration_points[1] = aruco_corners[1]  # top-right corner of the marker
            self.calibration_points[2] = aruco_corners[2]  # bottom-right corner of the marker
            self.calibration_points[3] = aruco_corners[3]  # bottom-left corner of the marker
            # Calculate the middle points of the edges
            self.calibration_points[4] = (self.calibration_points[0] + self.calibration_points[1]) /2.0
            self.calibration_points[5] = (self.calibration_points[1] + self.calibration_points[2]) /2.0
            self.calibration_points[6] = (self.calibration_points[2] + self.calibration_points[3]) /2.0
            self.calibration_points[7] = (self.calibration_points[3] + self.calibration_points[0]) /2.0
            # Add the center of the calibration board
            self.calibration_points[8] = aruco_corners.mean(axis=0)  # center of the marker

            logger.info(f"Calibration Points: {self.calibration_points}")

        elif aruco_corners is None or aruco_corners.shape != (4, 3):
            logger.error("aruco_corners must be a 4x3 numpy array representing the corners of the ArUco marker in world space")

        # Tracking parameters
        try:
            with open(CONFIG_FILENAME, 'r') as fp:
                self.parameter = json.load(fp)

        except FileNotFoundError:
            self.parameter = {'hue_h': 90,
                                    'hue_l': 36,
                                    'sat_h': 255,
                                    'sat_l': 100,
                                    'value_h': 255,
                                    'value_l': 35,
                                    'erosion_size': 1,
                                    'area': 1,
                                    }
            

    @property
    def aruco_center_position(self):
        """Get the position of the center of the ArUco marker in the world space"""
        return self.aruco_corners.mean(axis=0)
    
    @property
    def aruco_size(self):
        """Get the size of the ArUco marker in the world space, defined as the distance between the top-left and top-right corners"""
        return np.linalg.norm(self.aruco_corners[0] - self.aruco_corners[1])
            
    
    def mask_area(self, corners:np.ndarray, frame:np.ndarray) -> np.ndarray:
        """
        Create a mask of an area defined by the corners of a polygon

        Args:
            corners: numpy.ndarray
                The list of corners of the area 
            frame:numpy.ndarray
                The last color frame of the camera 

        Return:
            mask: numpy.ndarray                                        
                The masked frame with only the area visible, rest is black
        """
        frame_shape=frame.shape
        mask = np.zeros(frame_shape[:2], dtype=np.uint8)
        corners = np.array(corners, dtype=np.int32)
        cv.fillPoly(mask, [corners], 255)

        return mask
    

    def compute_camera_to_world_transform(self) -> bool:
        """
        Initialize the rotation matrix and the translation vector based on the last calibration process
                                            
        Return:
            True if the initialization process is successful, False otherwise
        """
        self.initialized = False
        self.trackers_pos = []
        self.points = []

        # If the calibration step is not required, read the values of the last calibration process
        with open(self.calibration_file, 'r') as file:
            reader = csv.reader(file)
            next(reader)  # skipp header

            # Read ground truth points in world space and their corresponding points in camera space from the calibration file
            for i in range(COUNT_CALIBRATION_POINTS):
                row = next(reader)
                logger.debug(f"Calibration point {i}: {row}")
                self.points.append((int(row[0]), int(row[1])))
                self.trackers_pos.append(
                    image_pixel_to_mm(
                        float(row[2]),  # depth
                        int(row[0]),  #  X coordinate
                        int(row[1]),  #  Y coordinate
                        self.intr
                    ))
                self.calibration_points[i] = row[4:7]  # world coordinates of the calibration points
            self.initialized = True
        
            logger.debug(f"Trackers positions from config file: {self.trackers_pos}")

        if not self.initialized:
            return False
        
        self.trackers_pos = np.array(self.trackers_pos)
        self.R, self.t = compute_transform_from_pointclouds( self.trackers_pos, self.calibration_points)
        
        self.initialized = True
        return True
    
    
    def calibrate(self, frame, depth_image, aggregate, window=None)-> bool:
        """
        Calibrate the camera by detecting a single marker and calculating the rotation matrix and translation vector.
        This method averages the corners positions of the marker and stores them in a CSV file.

        Args:
            frame: numpy.ndarray
                The color image returned by the camera

            depth_image: numpy.ndarray         
                The depth image returned by the camera

            aggregate: bool
                If True, the corners positions are aggregated over multiple frames, the mean value is then used for the calibration. If False, only the corners positions of the current frame are used for the calibration.

            window: CameraFeedWindow
                The window to display the camera feed
        Return:
            True if the calibration process is successful, False otherwise
        """
        dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_ARUCO_ORIGINAL)
        parameters =  cv.aruco.DetectorParameters()
        detector = cv.aruco.ArucoDetector(dictionary, parameters)
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        _,thresh_image = cv.threshold(gray,95,255,cv.THRESH_TOZERO)

        (corners, ids, rejected) = detector.detectMarkers(thresh_image)

        if ids is None:
            logger.error(f"Frame {self.count_calibration_frames}: No Aruco markers detected")
            return False
        if len(ids)>1:
            logger.error(f"Frame {self.count_calibration_frames}: More than one Aruco marker detected")
            return False
        if ids[0] != self.aruco_marker_id:
            logger.error(f"Frame {self.count_calibration_frames}: Aruco marker ID is not 672: {ids}")
            return False
        
        if not aggregate:
            self.trackers_pos = np.zeros((COUNT_CALIBRATION_POINTS, 3))
            self.points = np.zeros((COUNT_CALIBRATION_POINTS, 2))  # Initialize points array with 5 points and 2 coordinates (x, y)
            self.count_calibration_frames = 0

        # Add the corners positions of the marker to the 2D points and trackers_pos lists
        temp_points = np.zeros((COUNT_CALIBRATION_POINTS, 2))
        temp_trackers_pos = np.zeros((COUNT_CALIBRATION_POINTS, 3))
        for i in range(len(corners[0][0])):
            corner=corners[0][0][i]
            depth = depth_image[int(corner[1])][int(corner[0])]
            if depth == 0:
                logger.debug(f"Skipping frame: Depth value is 0 for corner {i} at position ({corner[0]}, {corner[1]})")
                return False

            temp_points[i] = [corner[0], corner[1]]
            temp_trackers_pos[i] = image_pixel_to_mm(depth, corner[0], corner[1], self.intr)
        
        # Add the the middle points between the corners
        for i in range(4):
            next_i = (i + 1) % 4
            temp_points[4 + i] = [
            (temp_points[i][0] + temp_points[next_i][0]) / 2,
            (temp_points[i][1] + temp_points[next_i][1]) / 2
            ]
            temp_trackers_pos[4 + i] = [
            (temp_trackers_pos[i][0] + temp_trackers_pos[next_i][0]) / 2,
            (temp_trackers_pos[i][1] + temp_trackers_pos[next_i][1]) / 2,
            (temp_trackers_pos[i][2] + temp_trackers_pos[next_i][2]) / 2
            ]
        
        # Replace the last dimension with the actual depth
        for i in range(4, 8):
            depth = depth_image[int(temp_points[i][1])][int(temp_points[i][0])]
            if depth == 0:
                logger.debug(f"Skipping frame: Depth value is 0 for corner {i} at position ({temp_points[i][0]}, {temp_points[i][1]})")
                return False
            temp_trackers_pos[i][2] = depth

        # Average the corners positions
        x, y = np.mean(corners[0][0], axis=0)
        x = int(x)
        y = int(y)

        # Adds the center of the marker to the points and trackers_pos lists
        depth= depth_image[y][x]
        if depth == 0:
                logger.debug(f"Skipping frame: Depth value is 0 for corner {i} at position ({corner[0]}, {corner[1]})")
                return False
        temp_points[-1] = [x, y]
        temp_trackers_pos[-1] = image_pixel_to_mm(depth, x, y, self.intr)


        # If the calibration is not aggregated, reset the points and trackers_pos lists, else add the new points and trackers_pos to the existing lists
        self.points = temp_points if not aggregate else self.points + temp_points
        self.trackers_pos = temp_trackers_pos if not aggregate else self.trackers_pos + temp_trackers_pos

        self.count_calibration_frames += 1

        # Write the information in a CSV file for the next calibration processes
        points_2d = [(int(self.points[i][0]/self.count_calibration_frames), 
                      int(self.points[i][1]/self.count_calibration_frames), 
                      self.trackers_pos[i][2]/self.count_calibration_frames, 
                      ids[0][0]) for i in range(len(self.points))]
        
        with open(CALIBRATION_FILENAME, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['X', 'Y', 'Depth', 'id'])  # En-tête
            writer.writerows(points_2d)
            logger.debug(f"Calibration data written to {CALIBRATION_FILENAME}: {points_2d}")

         # Draw the detected markers and the corners on the frame
        cv.aruco.drawDetectedMarkers(frame, corners, ids, borderColor=(255, 0, 0))
        cv.circle(frame, (int(corners[0][0][1][0]), int(corners[0][0][1][1])), 2, (0, 0, 255), -1)
        cv.circle(frame, (int(corners[0][0][2][0]), int(corners[0][0][2][1])), 2, (0, 255, 0), -1)
        cv.circle(frame, (int(corners[0][0][3][0]), int(corners[0][0][3][1])), 2, (0, 255, 255), -1)
        # draw 2D points on the frame
        [cv.circle(frame, (int(points_2d[i][0]), int(points_2d[i][1])), 5, (0, 0, 255), 1) for i in range(len(points_2d))]
        [cv.putText(frame, f"{i} ({int(corners[0][0][i][0])}, {int(corners[0][0][i][1])}, {depth_image[int(corners[0][0][i][1]),int(corners[0][0][i][0])]}) ", 
                        (int(corners[0][0][i][0]), int(corners[0][0][i][1])), 
                        cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1) for i in range(len(corners[0][0]))]
        frame = cv.putText(frame, f"Calibration progress: {self.count_calibration_frames}/200", (10, 30), 
                            cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        if window:
            window.set_frame(frame)

        return True

    
    def camera_to_world(self, x: int, y: int, depth: float) -> list[float]:
        """
        Calculate the position of the object in our frame space

        Args
        x,y: int
            The pixel coordinates

        depth: float
            The depth of the pixel

        Return:
            position: numpy.ndarray
                The real world coordinates of the object in the Emio frame space
        """
        position=np.zeros((3))
        p = image_pixel_to_mm(depth, x, y, self.intr)
        position = self.R@p
        position += self.t
        return [position[0], position[1], position[2]]



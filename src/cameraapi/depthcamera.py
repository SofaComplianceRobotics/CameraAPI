import os
import json
from time import sleep
import time
from enum import Enum

import numpy as np
import cv2 as cv

from ._camera_interface import DepthCameraInterface
from ._realsense_camera import RealSenseCamera, list_realsense_cameras
from ._camerafeedwindow import CameraFeedWindow
from ._positionestimation import PositionEstimation, image_pixel_to_mm, CONFIG_FILENAME
from cameraapi._logging_config import logger

DEFAULT_CAMERA_PARAMS = {"hue_h": 90, "hue_l": 36, "sat_h": 255, "sat_l": 138, "value_h": 255, "value_l": 35, "erosion_size": 0, "area": 100}

class CalibrationStatusEnum(Enum):
    NOT_CALIBRATED = 0,
    CALIBRATING = 1,
    CALIBRATED = 2


def compute_contour_center(contour):
    M = cv.moments(contour)
    cX = 0
    cY = 0
    if M['m00'] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    return cX, cY


def compute_median_depth(contour, depth_image):
    image = np.zeros_like(depth_image)
    # Fills the area bounded by the contours if thickness < 0
    cv.drawContours(image, contours=[contour], contourIdx=0, color=255, thickness=-1)
    points = np.where(image == 255)
    depth_values = depth_image[points[0], points[1]].flatten()
    valid_depth_values = depth_values[depth_values > 0]
    if len(valid_depth_values) > 0:
        return np.median(valid_depth_values)
    else:
        return 0


def list_cameras() -> list:
    """List all available depth cameras (currently RealSense only)"""
    return list_realsense_cameras()


# region DepthCamera class

class DepthCamera:

    _depth_max: int = 430
    _depth_min: int = 2
    calibration_status: CalibrationStatusEnum = CalibrationStatusEnum.NOT_CALIBRATED
    """The current calibration status of the camera. 
    The camera is considered calibrated when the camera-to-world transform is successfully computed."""
    
    # Camera interface instance
    _camera: DepthCameraInterface = None
    
    # Processing state
    _parameters: dict = {}
    _position_estimator: PositionEstimation = None
    compute_point_cloud: bool = False
    """ If True, the point cloud will be computed at each frame update. Default is False. """
    point_cloud: np.ndarray = None
    """ The point cloud of the current frame, as a numpy array of shape (N, 3) containing the 3D coordinates of the N points in the camera space. """
    tracking: bool = False
    """ If True, the tracking of the markers will be enabled. Default is False. """
    trackers_pos: list = []
    """ The positions of the tracked markers in the world space, as a list of lists of 3 floats (x, y, z) for each marker. """
    intr: object = None
    """ The camera intrinsic parameters, as an implementation-specific object returned by the camera interface. For RealSense cameras, this is a pyrealsense2.intrinsics object. """
    parameters_file: str = None

    # calibration
    aruco_marker_id: int = 672
    """The ID of the ArUco marker used for calibration. Defautl is 672"""
    aruco_corners: np.ndarray = None
    """
    The corners of the ArUco marker in the world space, defined as a numpy array of shape (4, 3) containing the 3D coordinates of the 4 corners of the marker in the world space. 
    The order of the corners should be consistent with the order of the corners detected in the image (top-left, top-right, bottom-right, bottom-left).
    """
    calibration_file: str = None
    """
    Path to the calibration file. If None, the default path will be used. 
    The file is used to compute the camera-to-world transform
    """
    
    # UI
    _mask_window: CameraFeedWindow = None
    _frame_window: CameraFeedWindow = None
    _hsv_window: CameraFeedWindow = None
    _depth_window: CameraFeedWindow = None
    _root_window: CameraFeedWindow = None
    show_video_feed: bool = False
    """ If True, the video feed will be shown in separate windows. Default is False. """

    # Frame data
    frame: np.ndarray = None
    """ The color image of the current frame, as a numpy array in BGR format. """
    hsv_frame: np.ndarray = None
    """ The HSV image of the current frame, as a numpy array. """
    mask_frame: np.ndarray = None
    """ The binary mask of the current frame, as a numpy array. The mask is computed based on the HSV values and the depth values, and is used for tracking the markers. """
    depth_frame: np.ndarray = None
    """ The depth image of the current frame, as a numpy array. The depth values are in millimeters. """
    _depth_frame_native = None  # Native frame object from camera


    def __init__(self,
                 camera_serial: str=None,
                 parameters: dict=None,
                 compute_point_cloud: bool=False,
                 show_video_feed: bool=False,
                 tracking: bool=True,
                 camera_type: str="realsense",
                 width: int=640,
                 height: int=480,
                 fps: int=30,
                 aruco_corners: np.ndarray=None,
                 aruco_marker_id: int=672,
                 calibration_file: str=None,
                 parameters_file: str=None
                 ) -> None:
        """
        Initialize the camera and the parameters.

        Args:
            camera_serial : str
                Serial number of the camera to use (None for default)
            parameter : dict
                The parameters for the camera. If None, the default parameters will be used.
            compute_point_cloud : bool
                If True, the point cloud will be computed.
            show_video_feed : bool
                If True, the video feed will be shown.
            tracking : bool
                If True, the tracking will be enabled.
            camera_type : str
                Type of camera to use. Currently only "realsense" is supported.
            aruco_corners : np.ndarray
                The corners of the ArUco marker in the world space, defined as a numpy array of shape (4, 3) containing the 3D coordinates of the 4 corners of the marker in the world space. 
                The order of the corners should be consistent with the order of the corners detected in the image (top-left, top-right, bottom-right, bottom-left).
            aruco_marker_id : int
                The ID of the ArUco marker used for calibration. Default is 672.
        """
        super().__init__()

        self.tracking = tracking
        self.show_video_feed = show_video_feed
        self.compute_point_cloud = compute_point_cloud
        self.aruco_marker_id = aruco_marker_id
        self.aruco_corners = aruco_corners
        self.calibration_file = calibration_file
        self.parameters_file = parameters_file
        
        # Initialize camera interface
        if camera_type == "realsense":
            self._camera = RealSenseCamera(
                camera_serial=camera_serial,
                width=width,
                height=height,
                fps=fps
            )
        else:
            raise ValueError(f"Unsupported camera type: {camera_type}")

        self._camera.initialized = True

        self.trackers_pos = []

        if parameters:
            self._parameters = parameters
        else:
            try:
                with open(CONFIG_FILENAME, 'r') as fp:
                    json_parameters = json.load(fp)
                    self._parameters.update(json_parameters)
                    logger.info(f'Config file {CONFIG_FILENAME} found. Using parameters {self._parameters}')

            except FileNotFoundError:
                logger.warning(f'Config file {CONFIG_FILENAME} not found. Using default parameters {DEFAULT_CAMERA_PARAMS}')
                self._parameters.update(DEFAULT_CAMERA_PARAMS)

        default_param = self._parameters.copy()

        self._camera.initialized = True

        if self.show_video_feed:
            self._create_feed_windows()

    ##########################
    #  PROPERTIES
    ##########################

#region properties

    @property
    def fps(self):
        """Get the camera frames per second (fps)"""
        return self._camera.fps if self._camera else None

    @fps.setter
    def fps(self, value):
        """Set the camera frames per second (fps)"""
        if self._camera is not None:
            self._camera.fps = value

    @property
    def width(self):
        """Get the camera frame width"""
        return self._camera.width if self._camera else None
    
    @property
    def height(self):
        """Get the camera frame height"""
        return self._camera.height if self._camera else None

    @property
    def depth_max(self):
        """Get the maximum depth value in millimeters"""
        return self._depth_max

    @depth_max.setter
    def depth_max(self, value):
        """Set the maximum depth value in millimeters. Depth values above this value will be ignored in the mask and the tracking."""
        if value <= 0:
            raise ValueError("depth_max must be greater than 0")
        self._depth_max = value

    @property
    def depth_min(self):
        """Get the minimum depth value in millimeters"""
        return self._depth_min

    @depth_min.setter
    def depth_min(self, value):
        """Set the minimum depth value in millimeters. Depth values below this value will be ignored in the mask and the tracking."""
        if value < 0:
            raise ValueError("depth_min must be greater than or equal to 0")
        self._depth_min = value

    @property
    def camera_serial(self) -> str:
        """
        Returns the serial of the camera as str
        """
        return self._camera.camera_serial if self._camera else None
    
    @property
    def parameters(self) -> dict:
        """
        Get the camera parameters in a dict object:
            - `hue_h`: int: The upper hue value.
            - `hue_l`: int: The lower hue value.
            - `sat_h`: int: The upper saturation value.
            - `sat_l`: int: The lower saturation value.
            - `value_h`: int: The upper value value.
            - `value_l`: int: The lower value value.
            - `erosion_size`: int: The size of the erosion kernel.
            - `area`: int: The minimum area of the detected objects.
        Returns:
            dict: The camera parameters.
        """
        return self._parameters if self._parameters else DEFAULT_CAMERA_PARAMS


    @parameters.setter
    def parameters(self, value: dict):
        """
        Set the camera tracking parameters from the dict object:
            - `hue_h`: int: The upper hue value.
            - `hue_l`: int: The lower hue value.
            - `sat_h`: int: The upper saturation value.
            - `sat_l`: int: The lower saturation value.
            - `value_h`: int: The upper value value.
            - `value_l`: int: The lower value value.
            - `erosion_size`: int: The size of the erosion kernel.
            - `area`: int: The minimum area of the detected objects.

        :::warning
        - The camera parameters are not saved to a file. You need to save them manually.
        - The paramters are set when opening the camera. To change the parameters programatically, you need to close the camera and open it again with the wanted parameters.
        :::

        Args:
            value: dict: The new camera parameters.
        """
        self._parameters = value

#endregion

##################################
# METHODS
##################################

# region methods

    def _create_feed_windows(self):
        import tkinter as tk
        from tkinter import ttk
        self._root_window = tk.Tk()
        self._root_window.resizable(False, False)

        self._root_window.title("Camera Feed Manager")
        ttk.Button(self._root_window, text="Close Windows", command=self.quit).pack(side=tk.BOTTOM, padx=5, pady=5)
        ttk.Button(self._root_window, text="Save", command=lambda: json.dump(self._parameters, open(CONFIG_FILENAME, 'w'))).pack(side=tk.BOTTOM, padx=5, pady=5)
        ttk.Button(self._root_window, text="Mask Window", command=self._create_mask_window).pack(side=tk.BOTTOM, padx=5, pady=5)
        ttk.Button(self._root_window, text="Frame Window", command=self._create_frame_window).pack(side=tk.BOTTOM, padx=5, pady=5)
        ttk.Button(self._root_window, text="HSV Window", command=self._create_HSV_window).pack(side=tk.BOTTOM, padx=5, pady=5)
        ttk.Button(self._root_window, text="Depth Window", command=self._createDepthWindow).pack(side=tk.BOTTOM, padx=5, pady=5)

        self._create_mask_window()
        self._create_frame_window()

        self._root_window.protocol("WM_DELETE_WINDOW", self.quit)
        self._root_window.update_idletasks()

    def _create_mask_window(self):
        if self._mask_window is None or not self._mask_window.running:
            self._mask_window = CameraFeedWindow(rootWindow=self._root_window, trackbarParams=self._parameters, name='Mask')

    def _create_frame_window(self):
        if self._frame_window is None or not self._frame_window.running:
            self._frame_window = CameraFeedWindow(rootWindow=self._root_window, name='RGB Frame')

    def _create_HSV_window(self):
        if self._hsv_window is None or not self._hsv_window.running:
            self._hsv_window = CameraFeedWindow(rootWindow=self._root_window, name='HSV Frame')

    def _createDepthWindow(self):
        if self._depth_window is None or not self._depth_window.running:
            self._depth_window = CameraFeedWindow(rootWindow=self._root_window, name='Depth Frame')

    def quit(self):
        for window in [self._mask_window, self._frame_window, self._hsv_window, self._depth_window]:
            if window is not None:
                window.closed()
        self._root_window.destroy()
        self.show_video_feed = False
        self._root_window = None

    def init_realsense(self):
        """Deprecated: This method is kept for backward compatibility. Use open() instead."""
        return self.open()

    def open(self):
        """
        Open and initialize the camera. This method will initialize the camera interface, get the camera intrinsics,
         and initialize the position estimation module by reading the calibration file if it exists. 
         If the calibration file does not exist or is invalid, the position estimation will be initialized
           with default parameters and the camera will be considered not calibrated.
        """
        try:
            if self._camera is None:
                raise Exception('Camera interface not initialized')
            self._camera.open()
            self.intr = self._camera.get_intrinsics()

        except Exception as err:
            self._camera.initialized = False
            logger.error('Could not open depthcamera', str(err))
            return False
        try:            
            # Initialize the position estimation by reading the calibration file
            self._position_estimator = PositionEstimation(self.intr, self.aruco_corners, self.aruco_marker_id, self.calibration_file, self.parameters_file)
            self._position_estimator.intr = self.intr

            if self._position_estimator.compute_camera_to_world_transform():
                self.calibration_status = CalibrationStatusEnum.CALIBRATED
                logger.info(f"Camera {self.camera_serial} calibration loaded from file. Camera is calibrated.")
            else:
                self.calibration_status = CalibrationStatusEnum.NOT_CALIBRATED
                logger.warning(f"Camera {self.camera_serial} is not calibrated.")

            if not self._position_estimator.initialized:
                logger.exception('Position estimation initialization failed. Please check the camera calibration. Tracking will be unavailable.')
                self.calibration_status = CalibrationStatusEnum.NOT_CALIBRATED
                self.tracking = False

        except Exception as err:
            logger.error(f"Error during position estimation initialization, no tracking available: {err}")
            self._position_estimator = None
            self.calibration_status = CalibrationStatusEnum.NOT_CALIBRATED
            self.tracking = False
        return True


    def calibrate(self,):
        """
        Calibrate the camera using an ArUco marker with ID given at initialization (default is 672). 
        The marker should be placed in the camera field of view and the calibration process will run until 
        the marker is detected and the camera-to-world transform is computed, or until a timeout of 
        5 minutes is reached. During calibration, a window will show the camera feed with the detected marker 
        and the calibration status.

        See the [ documentation](https://docs-support.compliance-robotics.com/docs/next/Users//getting-started-with-/).
        """

        starttime = time.time()
        first = False
        success = False
        self.calibration_status = CalibrationStatusEnum.CALIBRATING

        # Create the windows to display the binary mask and the HSV frame
        calibration_window = CameraFeedWindow(rootWindow=self._root_window, name='Calibration')

        if self._position_estimator is not None:
            while self._position_estimator.count_calibration_frames < 200 and time.time() - starttime < 300:
                self._position_estimator.intr= self.intr
                _, color_image, depth_image, _ = self.get_frame()
                success = self._position_estimator.calibrate(color_image, depth_image, first, calibration_window)
                first = success if not first else first
                if self.show_video_feed:
                    self._root_window.update()

        if success:
            self._position_estimator.compute_camera_to_world_transform()
            logger.info(f"Camera {self.camera_serial} successfully calibrated.")

        # Close the calibration window
        calibration_window.closed()

        self.calibration_status = CalibrationStatusEnum.CALIBRATED if success else CalibrationStatusEnum.NOT_CALIBRATED
        return success


    def get_frame(self) -> tuple[bool, np.ndarray, np.ndarray, any]:
        """
            Get a frame from the camera interface
            
         Returns:
            Tuple containing:
            - bool: Success flag
            - np.ndarray: Color image (BGR format)
            - np.ndarray: Depth image
            - any: Native depth frame
        """
        success, color_image, depth_image, depth_frame_native = self._camera.get_frame()
        self._depth_frame_native = depth_frame_native
        return success, color_image, depth_image, depth_frame_native


    def update(self):
        """
            Update the camera frames and tracking elements (markers and point cloud)
        """
        ret, self.frame, self.depth_frame, depth_rsframe = self.get_frame()

        if ret is False:
            return
        # if frame is read correctly ret is True

        self.hsv_frame = cv.cvtColor(self.frame, cv.COLOR_BGR2HSV)

        # color definition
        red_lower = np.array([self._parameters['hue_l'], self._parameters['sat_l'], self._parameters['value_l']])
        red_upper = np.array([self._parameters['hue_h'], self._parameters['sat_h'], self._parameters['value_h']])

        # red color mask (sort of thresholding, actually segmentation)
        mask = cv.inRange(self.hsv_frame, red_lower, red_upper)
        mask2 = cv.inRange(self.depth_frame, self.depth_min, self.depth_max)

        mask = cv.bitwise_and(mask, mask2, mask=mask)

        erosion_shape = cv.MORPH_RECT
        erosion_size = self._parameters['erosion_size']
        element = cv.getStructuringElement(erosion_shape, (2 * erosion_size + 1, 2 * erosion_size + 1),
                                           (erosion_size, erosion_size))

        mask = cv.erode(mask, element, iterations=3)
        mask = cv.dilate(mask, element, iterations=3)

        self.mask_frame = cv.bitwise_and(self.frame, self.frame, mask=mask)

        if self.tracking:
            contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            if len(contours) != 0:
                areas = [cv.contourArea(cnt) for cnt in contours]

                self.trackers_pos = []
                for i, a in enumerate(areas):
                    if a > self._parameters['area']:
                        x, y = compute_contour_center(contours[i])
                        marker_mask = np.zeros_like(mask)

                        depth = compute_median_depth(contours[i], self.depth_frame) if self.depth_frame[y, x] == 0 else self.depth_frame[y, x]
                        worldx, worldy, worldz = self._position_estimator.camera_to_world(x, y, depth)
                        self.trackers_pos.append([worldx, worldy, worldz])

                        cv.drawContours(marker_mask, [contours[i]], -1, color=255, thickness=-1)
                        for frame in [self.hsv_frame, self.frame]:
                            cv.circle(frame, (x, y), 2, color=255, thickness=-1)
                            cv.putText(frame, f"{i} ({x}, {y}, {depth})", (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                            cv.putText(frame, f"{i} ({worldx:.2f}, {worldy:.2f}, {worldz:.2f})", (x, y + 15), cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

                        if self.show_video_feed:
                            cv.drawContours(self.frame, [contours[i]], -1, (255, 255, 0), 3)

        if self.compute_point_cloud:
            self.point_cloud = self._camera.calculate_point_cloud(self._depth_frame_native)

        if self.show_video_feed:
            if self._root_window is None:
                self._create_feed_windows()

            if self._mask_window is not None and self._mask_window.running:
                self._mask_window.set_frame(self.mask_frame)

            if self._frame_window is not None and self._frame_window.running:
                self._frame_window.set_frame(self.frame)

            if self._hsv_window is not None and self._hsv_window.running:
                self._hsv_window.set_frame(self.hsv_frame)

            if self._depth_window is not None and self._depth_window.running:
                colorized = self._camera.colorize_depth_frame(self._depth_frame_native)
                self._depth_window.set_frame(colorized)

            self._root_window.update()


    def camera_to_world(self, x: int, y: int, depth: float = None) -> list[float]:
        """
        Get the 3D point in the world reference frame from the pixels and depth

        Args:
            x, y: int: the horizontal and vertical position in the image/frame

        Returns:
            a list of float of the corresponding 3D point in the world reference frame
        """
        if self.is_running:
            if depth is None:
                depth = self._camera.depth_frame[y][x]
            return self._camera.position_estimator.camera_to_world(x, y, depth)

        return None


    def close(self):
        """Close the camera and the windows"""
        try:
            self._camera.initialized = False
            if self._camera:
                self._camera.close()
            if self._root_window:
                self._root_window.destroy()
        except:
            pass


    def run_loop(self):
        """Run the camera update loop and the video feed windows if enabled"""
        while True:
            if self._root_window is None or not self._root_window.winfo_exists():
                break
            if self.show_video_feed:
                self._root_window.update()
            self.update()

        self.close()
# endregion
# endregion
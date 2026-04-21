"""RealSense camera implementation"""
import numpy as np
import pyrealsense2 as rs
from typing import Tuple, Any
from ._camera_interface import DepthCameraInterface
from ._positionestimation import PositionEstimation
from src.cameraapi._logging_config import logger


def list_realsense_cameras() -> list:
    """List all available RealSense camera serial numbers"""
    context = rs.context()
    return [d.get_info(rs.camera_info.serial_number) for d in context.devices]


class RealSenseCamera(DepthCameraInterface):
    """RealSense depth camera implementation"""
    
    def __init__(self, camera_serial: str = None, width: int = 640, height: int = 480, fps: int = 30):
        """
        Initialize RealSense camera.
        
        Args:
            camera_serial: Serial number of the camera (None for default)
            width: Frame width in pixels
            height: Frame height in pixels
            fps: Frames per second (30, 60, or 90)
        """
        self._serial = camera_serial
        self._width = width
        self._height = height
        self.fps = fps
        
        # RealSense-specific objects
        self.pipeline = None
        self.rsconfig = None
        self.pipeline_wrapper = None
        self.pipeline_profile = None
        self.device = None
        self.profile = None
        self._intrinsics = None
        self.pc = None  # Point cloud processor
    
    def open(self) -> bool:
        """Open and initialize the RealSense camera"""
        try:
            self._init_realsense()
            return True
        except Exception as err:
            logger.error(f'Could not open RealSense camera: {err}')
            raise
    
    def _init_realsense(self):
        """Configure and start the RealSense pipeline"""
        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        self.rsconfig = rs.config()
        self.pc = rs.pointcloud()

        if self._serial is not None:
            self.rsconfig.enable_device(self._serial)

        # Get device product line for setting a supporting resolution
        self.pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        self.pipeline_profile = self.rsconfig.resolve(self.pipeline_wrapper)

        self.device = self.pipeline_profile.get_device()

        self.rsconfig.enable_stream(rs.stream.depth, self._width, self._height, rs.format.z16, self.fps)
        self.rsconfig.enable_stream(rs.stream.color, self._width, self._height, rs.format.bgr8, self.fps)

        depth_sensor = self.device.first_depth_sensor()
        depth_sensor.set_option(rs.option.depth_units, 0.001)

        cfg = self.pipeline.start(self.rsconfig)

        self.profile = cfg.get_stream(rs.stream.depth)
        self._intrinsics = self.profile.as_video_stream_profile().get_intrinsics()
    
    def close(self):
        """Stop the RealSense pipeline"""
        try:
            if self.pipeline:
                self.pipeline.stop()
        except Exception as err:
            logger.error(f'Error closing RealSense camera: {err}')
    
    def get_frame(self) -> Tuple[bool, np.ndarray, np.ndarray, Any]:
        """
        Capture a frame from the RealSense camera.
        
        Returns:
            Tuple of (success, color_image, depth_image, depth_frame_native)
        """
        try:
            frames = self.pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            if not depth_frame or not color_frame:
                return False, None, None, None

            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            return True, color_image, depth_image, depth_frame
        except Exception as err:
            logger.error(f'Error getting frame: {err}')
            return False, None, None, None
    
    def get_intrinsics(self):
        """Get camera intrinsic parameters"""
        return self._intrinsics
    
    def calculate_point_cloud(self, depth_frame: Any) -> np.ndarray:
        """
        Calculate point cloud from a RealSense depth frame.
        
        Args:
            depth_frame: RealSense depth frame object
            
        Returns:
            Point cloud as Nx3 numpy array (xyz)
        """
        if self.pc is None or depth_frame is None:
            return np.array([])
        
        points = self.pc.calculate(depth_frame)
        v = points.get_vertices()
        return np.asanyarray(v).view(np.float32).reshape(-1, 3)
    
    def colorize_depth_frame(self, depth_frame: Any) -> np.ndarray:
        """
        Colorize a RealSense depth frame for visualization.
        
        Args:
            depth_frame: RealSense depth frame object
            
        Returns:
            Colorized depth image as numpy array
        """
        if depth_frame is None:
            return np.array([])
        
        colorizer = rs.colorizer()
        colorized = np.asanyarray(colorizer.colorize(depth_frame).get_data())
        return colorized
    
    @property
    def camera_serial(self) -> str:
        """Get the camera serial number"""
        return self.device.get_info(rs.camera_info.serial_number) if self.device else self._serial
    
    @property
    def width(self) -> int:
        """Get the frame width"""
        return self._width
    
    @property
    def height(self) -> int:
        """Get the frame height"""
        return self._height

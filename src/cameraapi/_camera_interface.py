"""Abstract interface for depth cameras"""
from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, Any


class DepthCameraInterface(ABC):
    """Abstract base class for depth camera implementations"""

    _height: int = 480
    _width: int = 640
    _fps: int = 30

    _initialized = False

    #####################
    # Camera parameters
    #####################

    @property
    def camera_serial(self) -> str:
        """Get the camera serial number"""
        pass
    
    @property
    def height(self) -> int:
        """Get the camera frame height"""
        return self._height
    
    @height.setter
    def height(self, value: int):
        """Set the camera frame height"""
        self._height = value
    
    @property
    def width(self) -> int:
        """Get the camera frame width"""
        return self._width

    @width.setter
    def width(self, value: int):
        """Set the camera frame width"""
        self._width = value

    @property
    def fps(self) -> int:
        """Get the camera frames per second (fps)"""
        return self._fps
    
    @fps.setter
    def fps(self, value: int):
        """Set the camera frames per second (fps)"""
        self._fps = value

    @property
    def initialized(self) -> bool:
        """Check if the camera is initialized"""
        return self._initialized
    
    @initialized.setter
    def initialized(self, value: bool):
        """Set the camera initialized state"""
        self._initialized = value

    ##########################
    # Camera control methods
    ##########################
    
    @abstractmethod
    def open(self) -> bool:
        """
        Open and initialize the camera.
        
        Returns:
            bool: True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def close(self):
        """Close and cleanup the camera"""
        pass
    
    @abstractmethod
    def get_frame(self) -> Tuple[bool, np.ndarray, np.ndarray, Any]:
        """
        Get a frame from the camera.
        
        Returns:
            Tuple containing:
            - bool: Success flag
            - np.ndarray: Color image (BGR format)
            - np.ndarray: Depth image
            - Any: Native frame object (implementation-specific, used for advanced features like point cloud)
        """
        pass
    
    @abstractmethod
    def get_intrinsics(self):
        """
        Get camera intrinsic parameters.
        
        Returns:
            Camera intrinsics object (implementation-specific format)
        """
        pass
    
    @abstractmethod
    def calculate_point_cloud(self, depth_frame: Any) -> np.ndarray:
        """
        Calculate point cloud from depth frame.
        
        Args:
            depth_frame: Native depth frame object
            
        Returns:
            np.ndarray: Point cloud as Nx3 array (xyz coordinates)
        """
        pass
    
    @abstractmethod
    def colorize_depth_frame(self, depth_frame: Any) -> np.ndarray:
        """
        Colorize a depth frame for visualization.
        
        Args:
            depth_frame: Native depth frame object
            
        Returns:
            np.ndarray: Colorized depth image
        """
        pass


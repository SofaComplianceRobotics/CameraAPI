"""Abstract interface for depth cameras"""
from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, Any


class DepthCameraInterface(ABC):
    """Abstract base class for depth camera implementations"""
    
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
    
    @property
    @abstractmethod
    def camera_serial(self) -> str:
        """Get the camera serial number"""
        pass
    
    @property
    @abstractmethod
    def width(self) -> int:
        """Get the camera frame width"""
        pass
    
    @property
    @abstractmethod
    def height(self) -> int:
        """Get the camera frame height"""
        pass

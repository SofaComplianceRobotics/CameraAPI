import sys

from cameraapi._depthcamera import CalibrationStatusEnum 
from cameraapi.camera import Camera
from cameraapi._logging_config import logger

def calibrate():
    """
    Calibrate the camera of the first Emio camera found
    """
    camera = Camera(show=True)
    print("Available cameras:", Camera.listCameras())

    if camera.open():
        print(f"Camera {camera.camera_serial} opened.")
        if camera.calibration_status == CalibrationStatusEnum.NOT_CALIBRATED:
            camera.calibrate()
        else:
            print("Camera is already calibrated.")

        while camera.is_running:
            try:
                camera.update() # update the camera frame and trackers
                logger.info(camera.trackers_pos)
            except KeyboardInterrupt: 
                logger.info("Keyboard interrupt received.")
                break
            except Exception as e:
                logger.exception(f"Error during communication: {e}")
                break

        camera.close()
    else:
        print("Failed to open camera.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == 'calibrate':
        calibrate()
    else:
        print("Available functions from cameraAPI tool:")
        for name in dir(sys.modules[__name__]):
            if not name.startswith("_"):
                attr = getattr(sys.modules[__name__], name)
                if callable(attr):
                    print(f"  {name} - {attr.__doc__}")
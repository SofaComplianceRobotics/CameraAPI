# CameraAPI

## Description

CameraAPI is a library for interacting with camera devices through a simple API.

## Installation and Usage

To install CameraAPI, run:

```
pip install git+https://github.com/SofaComplianceRobotics/CameraAPI.git@main
```

Basic usage:

```python
from cameraapi import Camera

cam = Camera()
cam.open()
camera.update() # to get a frame and process it
```

## For Developers

Clone the repository and install dependencies using uv:

```
uv sync
```
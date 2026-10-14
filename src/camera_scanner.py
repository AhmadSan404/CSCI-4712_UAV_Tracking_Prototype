"""Finding the cameras (webcams, OBS Virtual Camera) that can be used as a live source."""
from dataclasses import dataclass

import cv2

CAMERA_BACKEND = cv2.CAP_DSHOW  # DirectShow: OBS Virtual Camera registers itself here
FALLBACK_PROBE_COUNT = 6  # without the name lister, try this many camera numbers


@dataclass(frozen=True)
class CameraInfo:
    """One camera: the number OpenCV opens it by, and its device name."""

    index: int
    name: str


def list_cameras() -> list[CameraInfo]:
    """Lists the cameras by name. Falls back to probing numbers ('Camera 0', ...) if names cannot be read."""
    try:
        from cv2_enumerate_cameras import enumerate_cameras

        return [CameraInfo(camera.index, camera.name) for camera in enumerate_cameras(CAMERA_BACKEND)]
    except (ImportError, OSError):
        return _probe_cameras()

def find_camera_index(name:str) -> int | None:
    """The current number of the camera with this name, or None if it is not connected."""
    return next((camera.index for camera in list_cameras() if camera.name == name), None)

def _probe_cameras() -> list[CameraInfo]:
    """Tries to open the first few camera numbers and keeps the ones that open."""
    cameras = []
    for index in range(FALLBACK_PROBE_COUNT):
        capture = cv2.VideoCapture(index, CAMERA_BACKEND)
        if capture.isOpened():
            cameras.append(CameraInfo(index, f"Camera {index}"))
        capture.release()
    return cameras

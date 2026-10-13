"""Vehicle classes, the detection array format, and the mapping from model class names to vehicle classes.

Detections travel through the program as a NumPy array of shape (N, 6), one row per vehicle, with the columns named
below. For example, `detections[:, CONFIDENCE]` is the confidence of every detection.
"""

from enum import Enum

import numpy as np


class VehicleClass(Enum):
    """The vehicle kinds this project detects."""

    CAR = "car"
    TRUCK = "truck"
    BUS = "bus"
    MOTORCYCLE = "motorcycle"
    PERSON = "person"  # only detected when "Track people" is on

    @property
    def display_name(self) -> str:
        """The name shown on screen and used in settings."""
        return self.value


VEHICLE_CLASSES = tuple(VehicleClass)  # the position in this tuple is the CLASS_INDEX stored in detections
VEHICLE_ONLY_CLASSES = tuple(c for c in VehicleClass if c is not VehicleClass.PERSON)  # the menu's class checkboxes
PERSON_CLASS_INDEX = VEHICLE_CLASSES.index(VehicleClass.PERSON)

# Column layout of a detections array. Boxes are in pixels.
X1, Y1, X2, Y2, CONFIDENCE, CLASS_INDEX = range(6)
NUM_COLUMNS = 6

# Different datasets (COCO, DOTA, Objects365, VisDrone) name their vehicle classes differently. This maps those
# names onto our own classes, so COCO and aerial (OBB) models both work without code changes.
MODEL_CLASS_NAME_TO_VEHICLE_CLASS = {
    "car": VehicleClass.CAR,
    "sports car": VehicleClass.CAR,
    "suv": VehicleClass.CAR,
    "van": VehicleClass.CAR,
    "small vehicle": VehicleClass.CAR,
    "truck": VehicleClass.TRUCK,
    "pickup truck": VehicleClass.TRUCK,
    "large vehicle": VehicleClass.TRUCK,
    "bus": VehicleClass.BUS,
    "motorcycle": VehicleClass.MOTORCYCLE,
    "motorbike": VehicleClass.MOTORCYCLE,
    "motor": VehicleClass.MOTORCYCLE,
    "person": VehicleClass.PERSON,
    "pedestrian": VehicleClass.PERSON,
    "people": VehicleClass.PERSON,
}


def empty_detections() -> np.ndarray:
    """A detections array with no rows."""
    return np.empty((0, NUM_COLUMNS), dtype=np.float32)


def find_vehicle_class(display_name: str) -> VehicleClass | None:
    """Looks up a vehicle class by its display name, or returns None if there is no such class."""
    for vehicle_class in VehicleClass:
        if vehicle_class.display_name == display_name:
            return vehicle_class
    return None

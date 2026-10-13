"""A tracked vehicle in one frame."""

from dataclasses import dataclass

import numpy as np


@dataclass
class Track:
    """One tracked vehicle.

    Attributes:
        track_id: Unique identifier. IDs are never reused.
        box: Bounding box as [x1, y1, x2, y2] in frame pixels.
        confidence: Detection confidence of the track.
        class_index: Index into `vehicle_types.VEHICLE_CLASSES`.
        age_frames: Number of frames since the track started.
        coasting_frames: Frames the track has gone without a matching detection (0 while it is matched). While
            coasting, the box is the motion model's prediction.
    """

    track_id: int
    box: np.ndarray
    confidence: float
    class_index: int
    age_frames: int
    coasting_frames: int = 0

    @property
    def is_coasting(self) -> bool:
        """True while the track follows its motion model instead of a real detection."""
        return self.coasting_frames > 0

    @property
    def center(self) -> tuple[float, float]:
        """The (x, y) center of the box."""
        return (self.box[0] + self.box[2]) / 2, (self.box[1] + self.box[3]) / 2

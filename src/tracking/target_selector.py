"""Choosing the target: the car being followed. Automatic by default, or locked by the user."""

import math

from src.tracking.track import Track

# How much each factor counts towards a track's "prominence" score (the weights add up to 1.0).
WEIGHT_CLOSE_TO_CENTER = 0.45
WEIGHT_LARGE_BOX = 0.30
WEIGHT_CONFIDENCE = 0.15
WEIGHT_LONG_LIVED = 0.10

FULL_SIZE_FRACTION = 0.3  # a box whose side is 30% of the frame (sqrt of area share) counts as "fully large"
FULL_AGE_FRAMES = 24  # a track this many frames old counts as "fully established"


class TargetSelector:
    """Picks the target. Automatic mode prefers a large, central, confident, long-lived track and sticks with it
    until that track ends. Locking a track (click or Tab) overrides this until that track ends.

    Attributes:
        target_id: Track ID of the current target, or None.
        is_locked: True when the user chose the target, False when it was chosen automatically.
    """

    def __init__(self, excluded_class_indices: tuple[int, ...] = ()) -> None:
        """Args:
            excluded_class_indices: Classes that automatic mode never picks (the user can still lock onto them).
        """
        self._excluded_class_indices = excluded_class_indices
        self.target_id: int | None = None
        self.is_locked = False

    def lock(self, track_id: int) -> None:
        """Locks the target onto a track chosen by the user."""
        self.target_id = track_id
        self.is_locked = True

    def release(self) -> None:
        """Clears the target, so automatic selection takes over on the next `update`."""
        self.target_id = None
        self.is_locked = False

    def update(self, tracks: list[Track], frame_shape: tuple[int, ...]) -> int | None:
        """Returns the current target ID, choosing a new one automatically if the old target is gone."""
        if self.target_id is not None and any(track.track_id == self.target_id for track in tracks):
            return self.target_id

        self.is_locked = False  # the locked track ended (or there never was one)
        frame_height, frame_width = frame_shape[:2]
        candidates = [track for track in tracks if not track.is_coasting and track.class_index not in self._excluded_class_indices]
        if candidates:
            best = max(candidates, key=lambda track: self._score_prominence(track, frame_width, frame_height))
            self.target_id = best.track_id
        else:
            self.target_id = None
        return self.target_id

    @staticmethod
    def _score_prominence(track: Track, frame_width: int, frame_height: int) -> float:
        """Scores (0 to 1) how likely a track is to be the car the drone is following."""
        center_x, center_y = track.center
        half_width, half_height = frame_width / 2, frame_height / 2
        # 0 at the exact center of the frame, 1 in a corner.
        distance_from_center = math.hypot((center_x - half_width) / half_width, (center_y - half_height) / half_height)
        distance_from_center /= math.sqrt(2)

        box_width = track.box[2] - track.box[0]
        box_height = track.box[3] - track.box[1]
        area_share = box_width * box_height / (frame_width * frame_height)
        size_score = min(1.0, math.sqrt(area_share) / FULL_SIZE_FRACTION)
        age_score = min(1.0, track.age_frames / FULL_AGE_FRAMES)

        return (
            WEIGHT_CLOSE_TO_CENTER * (1 - distance_from_center)
            + WEIGHT_LARGE_BOX * size_score
            + WEIGHT_CONFIDENCE * track.confidence
            + WEIGHT_LONG_LIVED * age_score
        )

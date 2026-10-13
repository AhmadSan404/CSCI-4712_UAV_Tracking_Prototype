"""Multi-object tracking with BoT-SORT, using camera motion that we measure ourselves."""

import numpy as np
from ultralytics.engine.results import Boxes
from ultralytics.trackers.bot_sort import BOTSORT
from ultralytics.trackers.utils.stracks import multi_gmc
from ultralytics.utils import IterableSimpleNamespace

from src.tracking.track import Track

TRACK_MEMORY_SECONDS = 2.5  # how long a missed track is remembered, so it can be matched again if the car reappears
COASTING_SECONDS = 1.0  # how long a missed track is still reported, at its predicted position

BOTSORT_SETTINGS = dict(
    tracker_type="botsort",
    track_high_thresh=0.3,  # detections above this take part in the first association
    track_low_thresh=0.1,  # weaker detections can still extend existing tracks (ByteTrack's second stage)
    new_track_thresh=0.35,  # minimum confidence to start a new track
    match_thresh=0.8,  # max association cost (1 - IoU)
    fuse_score=False,  # confidence-weighted costs reject the weak detections typical of tiny top-down cars
    gmc_method="none",  # BoT-SORT's own motion compensation is off: we supply the camera warp instead
    proximity_thresh=0.5,
    appearance_thresh=0.8,
    with_reid=False,
    model="auto",
)


class BotSortWithCameraMotion(BOTSORT):
    """BoT-SORT whose camera-motion compensation comes from the caller instead of being computed inside.

    Our estimate masks out the vehicles, so a car the drone keeps centered does not bias it, and the same warp keeps
    the drawn trails consistent with the tracker's motion model.

    Attributes:
        camera_warp: The 2x3 transform to apply on the next `update` (None for no compensation).
    """

    def __init__(self, settings: IterableSimpleNamespace) -> None:
        super().__init__(settings)
        self.camera_warp: np.ndarray | None = None

    def _pre_first_associate(self, strack_pool: list, unconfirmed: list, image: object, results_high: object) -> None:
        """Moves the existing tracks by the camera warp before they are matched against new detections."""
        if self.camera_warp is not None:
            multi_gmc(strack_pool, self.camera_warp)
            multi_gmc(unconfirmed, self.camera_warp)


class VehicleTracker:
    """Follows every vehicle from frame to frame, keeping each one's ID."""

    def __init__(self, video_fps: float) -> None:
        settings = {**BOTSORT_SETTINGS, "track_buffer": max(1, round(TRACK_MEMORY_SECONDS * video_fps))}
        self._botsort = BotSortWithCameraMotion(IterableSimpleNamespace(**settings))
        self._coasting_frame_limit = max(1, round(COASTING_SECONDS * video_fps))

    def update(self, detections: np.ndarray, frame_shape: tuple[int, ...], camera_warp: np.ndarray | None) -> list[Track]:
        """Advances the tracker by one frame.

        Args:
            detections: (N, 6) detections of this frame, see `vehicle_types`.
            frame_shape: Shape of the frame, (height, width, ...).
            camera_warp: 2x3 camera motion since the previous frame, or None.

        Returns:
            The tracks matched in this frame, followed by briefly missed tracks at their predicted position (those
            have `coasting_frames` above zero).
        """
        botsort = self._botsort
        botsort.camera_warp = camera_warp
        botsort.update(Boxes(detections, frame_shape[:2]), None)

        tracks = [
            Track(state.track_id, state.xyxy.copy(), float(state.score), int(state.cls), botsort.frame_id - state.start_frame)
            for state in botsort.tracked_stracks
            if state.is_activated
        ]
        for lost in botsort.lost_stracks:
            frames_lost = botsort.frame_id - lost.end_frame
            if 0 < frames_lost <= self._coasting_frame_limit:
                age = botsort.frame_id - lost.start_frame
                tracks.append(Track(lost.track_id, lost.xyxy.copy(), float(lost.score), int(lost.cls), age, frames_lost))
        return tracks

    def clear(self) -> None:
        """Drops all tracks, for example at a scene cut. IDs keep counting up, so they are never reused."""
        botsort = self._botsort
        botsort.tracked_stracks, botsort.lost_stracks, botsort.removed_stracks = [], [], []

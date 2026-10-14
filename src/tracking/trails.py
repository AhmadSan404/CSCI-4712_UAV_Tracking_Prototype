"""Trails: the recent path of each tracked car, kept pinned to the ground as the camera moves."""
from collections import deque

import numpy as np

from src.tracking.track import Track

TRAIL_SECONDS = 2.0  # how much history each trail shows
FORGET_SECONDS = 1.0  # a trail is deleted this long after its car was last seen
MIN_POINTS_TO_DRAW = 2


class Trails:
    """Recent center points per track, in current-frame pixels.

    A following drone keeps the car near the same spot in the image, so raw positions would barely move. Moving the
    old points with each frame's camera warp keeps them on the ground, so a trail shows the path actually driven.
    """

    def __init__(self, video_fps:float) -> None:
        self._max_points = round(TRAIL_SECONDS * video_fps)
        self._forget_after_frames = round(FORGET_SECONDS * video_fps)
        self._points_by_track: dict[int, deque] = {}
        self._frames_unseen_by_track: dict[int, int] = {}

    def clear(self) -> None:
        """Deletes every trail."""
        self._points_by_track.clear()
        self._frames_unseen_by_track.clear()

    def update(self, tracks:list[Track], camera_warp:np.ndarray|None) -> None:
        """Moves the old points with the camera, then adds the center of every matched track.

        Coasting tracks add no point. Trails unseen for too long are deleted.
        """
        if camera_warp is not None and not np.allclose(camera_warp, np.eye(2, 3)):
            for track_id, points in self._points_by_track.items():
                moved = np.asarray(points) @ camera_warp[:, :2].T + camera_warp[:, 2]
                self._points_by_track[track_id] = deque(map(tuple, moved), maxlen=self._max_points)

        seen_ids = set()
        for track in tracks:
            if not track.is_coasting:
                points = self._points_by_track.setdefault(track.track_id, deque(maxlen=self._max_points))
                points.append(track.center)
                seen_ids.add(track.track_id)

        for track_id in list(self._points_by_track):
            if track_id in seen_ids:
                self._frames_unseen_by_track[track_id] = 0
            else:
                self._frames_unseen_by_track[track_id] = self._frames_unseen_by_track.get(track_id, 0) + 1
            if self._frames_unseen_by_track[track_id] > self._forget_after_frames:
                del self._points_by_track[track_id], self._frames_unseen_by_track[track_id]

    def get_points(self, track_id:int) -> np.ndarray | None:
        """The trail of one track as an (N, 2) array, oldest first, or None if there is too little to draw."""
        points = self._points_by_track.get(track_id)
        if points is None or len(points) < MIN_POINTS_TO_DRAW:
            return None
        return np.asarray(points, dtype=np.float32)

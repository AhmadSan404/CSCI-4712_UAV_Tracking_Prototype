"""Drawing the tracking result on the video: boxes with track IDs, the target marker and trails."""

import cv2
import numpy as np

from src.detection.vehicle_types import VEHICLE_CLASSES
from src.display.drawing import (
    calculate_ui_scale,
    darken,
    draw_corner_marks,
    draw_dashed_rect,
    draw_label,
    draw_trail,
    to_pixel,
)
from src.tracking.track import Track
from src.tracking.trails import Trails

TARGET_COLOR = (64, 64, 255)  # BGR red: the car being followed
TRACK_COLORS = [(255, 210, 0), (80, 255, 120), (255, 120, 255), (0, 220, 255), (255, 160, 70), (170, 120, 255)]


def pick_track_color(track_id: int) -> tuple[int, int, int]:
    """A display color for a track. The same ID always gets the same color."""
    return TRACK_COLORS[track_id % len(TRACK_COLORS)]


class Overlay:
    """Draws tracks and trails onto a frame that has already been resized to the display size.

    Attributes:
        show_trails: Whether trails are drawn.
    """

    def __init__(self, show_trails: bool) -> None:
        self.show_trails = show_trails

    def draw_tracks(
        self,
        image: np.ndarray,
        tracks: list[Track],
        trails: Trails,
        target_id: int | None,
        is_target_locked: bool,
        scale: float,
    ) -> None:
        """Draws trails, then every track with its ID, with the target on top.

        Args:
            image: Frame at display size, modified in place.
            tracks: The tracks to draw, in source-frame pixels.
            trails: Trail history in source-frame pixels.
            target_id: Track ID of the target, drawn in the target style, or None.
            is_target_locked: Whether the user locked the target (changes its label).
            scale: Ratio of display pixels to source-frame pixels.
        """
        ui_scale = calculate_ui_scale(image)

        if self.show_trails:
            for track in tracks:
                points = trails.get_points(track.track_id)
                if points is not None:
                    color = TARGET_COLOR if track.track_id == target_id else pick_track_color(track.track_id)
                    draw_trail(image, points * scale, color, ui_scale)

        # Sorting by "is target" puts the target last (False < True), so it is drawn on top of the others.
        for track in sorted(tracks, key=lambda track: track.track_id == target_id):
            top_left = to_pixel(track.box[:2] * scale)
            bottom_right = to_pixel(track.box[2:] * scale)
            if track.track_id == target_id:
                self._draw_target(image, track, top_left, bottom_right, is_target_locked, ui_scale)
            else:
                self._draw_ordinary_track(image, track, top_left, bottom_right, ui_scale)

    @staticmethod
    def _draw_ordinary_track(
        image: np.ndarray, track: Track, top_left: tuple[int, int], bottom_right: tuple[int, int], ui_scale: float
    ) -> None:
        """A thin box with corner marks and a 'class #id confidence' label; dashed and dimmed while coasting."""
        color = pick_track_color(track.track_id)
        name = f"{VEHICLE_CLASSES[track.class_index].display_name} #{track.track_id}"

        if track.is_coasting:
            dim_color = darken(color)
            draw_dashed_rect(image, top_left, bottom_right, dim_color, max(1, round(1.5 * ui_scale)))
            draw_label(image, f"{name} lost", top_left, dim_color, ui_scale)
            return

        cv2.rectangle(image, top_left, bottom_right, darken(color, 0.6), 1, cv2.LINE_AA)
        draw_corner_marks(image, top_left, bottom_right, color, max(2, round(2 * ui_scale)))
        draw_label(image, f"{name} {track.confidence:.2f}", top_left, color, ui_scale)

    @staticmethod
    def _draw_target(
        image: np.ndarray,
        track: Track,
        top_left: tuple[int, int],
        bottom_right: tuple[int, int],
        is_locked: bool,
        ui_scale: float,
    ) -> None:
        """The followed car: a red box with corner marks, a reticle and a TARGET/LOCKED label (dashed if coasting)."""
        status = "LOCKED" if is_locked else "TARGET"
        name = f"{status} {VEHICLE_CLASSES[track.class_index].display_name} #{track.track_id}"

        if track.is_coasting:
            draw_dashed_rect(image, top_left, bottom_right, TARGET_COLOR, max(2, round(2 * ui_scale)))
            draw_label(image, f"{name} searching", top_left, TARGET_COLOR, ui_scale)
            return

        cv2.rectangle(image, top_left, bottom_right, darken(TARGET_COLOR, 0.7), 1, cv2.LINE_AA)
        draw_corner_marks(image, top_left, bottom_right, TARGET_COLOR, max(2, round(3 * ui_scale)))
        Overlay._draw_reticle(image, top_left, bottom_right, ui_scale)
        draw_label(image, f"{name} {track.confidence:.2f}", top_left, TARGET_COLOR, ui_scale)

    @staticmethod
    def _draw_reticle(
        image: np.ndarray, top_left: tuple[int, int], bottom_right: tuple[int, int], ui_scale: float
    ) -> None:
        """A short tick at the middle of each side of a box, pointing in towards the center."""
        (left, top), (right, bottom) = top_left, bottom_right
        center_x, center_y = (left + right) // 2, (top + bottom) // 2
        tick = round(10 * ui_scale)
        thickness = max(1, round(2 * ui_scale))

        ticks = (
            ((center_x, top - tick), (center_x, top + tick // 2)),
            ((center_x, bottom + tick), (center_x, bottom - tick // 2)),
            ((left - tick, center_y), (left + tick // 2, center_y)),
            ((right + tick, center_y), (right - tick // 2, center_y)),
        )
        for start, end in ticks:
            cv2.line(image, start, end, TARGET_COLOR, thickness, cv2.LINE_AA)

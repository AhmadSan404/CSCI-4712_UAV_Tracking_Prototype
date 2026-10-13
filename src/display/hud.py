"""The heads-up display: FPS counter (top right), info panel (top left), banner (top center), help bar (bottom)."""

from dataclasses import dataclass

import cv2
import numpy as np

from src.display.drawing import (
    FONT,
    GREY,
    WHITE,
    calculate_ui_scale,
    darken_region,
    draw_text_panel,
    format_clock,
    put_text,
)

FPS_GOOD_COLOR = (90, 230, 90)
FPS_WARNING_COLOR = (0, 210, 255)
FPS_BAD_COLOR = (60, 60, 255)
FPS_GOOD_RATIO = 0.95  # processing at least 95% of the video's frame rate counts as keeping up
FPS_WARNING_RATIO = 0.7
HELP_TEXT_COLOR = (220, 220, 220)


@dataclass
class HudState:
    """Everything the HUD shows for one frame."""

    fps: float
    video_fps: float
    is_paused: bool
    has_ended: bool
    models_text: str
    hardware_text: str
    detect_ms: float
    track_ms: float
    num_tracks: int
    target_text: str
    video_position_seconds: float
    video_duration_seconds: float
    is_real_time: bool
    help_text: str
    is_live: bool = False


class Hud:
    """Draws the heads-up display onto a display-size image.

    Attributes:
        show_info_panel: Whether the info panel is drawn (the FPS counter and banner always are).
        show_help_bar: Whether the help bar is drawn.
    """

    def __init__(self) -> None:
        self.show_info_panel = True
        self.show_help_bar = True

    def draw(self, image: np.ndarray, state: HudState) -> None:
        """Draws the FPS counter, the PAUSED/END banner, and the info panel and help bar if they are shown."""
        ui_scale = calculate_ui_scale(image)
        padding = round(10 * ui_scale)

        self._draw_fps_counter(image, state, ui_scale, padding)
        if state.is_paused or state.has_ended:
            self._draw_banner(image, state, ui_scale, padding)
        if self.show_info_panel:
            self._draw_info_panel(image, state, ui_scale, padding)
        if self.show_help_bar:
            self._draw_help_bar(image, state, ui_scale, padding)

    @staticmethod
    def _draw_fps_counter(image: np.ndarray, state: HudState, ui_scale: float, padding: int) -> None:
        """The FPS readout in the top-right corner: green, yellow or red against the video's frame rate."""
        value_text = f"{state.fps:5.1f}" if state.fps > 0 else "  -- "
        value_scale, unit_scale = 0.95 * ui_scale, 0.5 * ui_scale
        (value_width, value_height), _ = cv2.getTextSize(value_text, FONT, value_scale, 2)
        (unit_width, _), _ = cv2.getTextSize("FPS", FONT, unit_scale, 1)
        gap = round(6 * ui_scale)

        right, top = image.shape[1] - padding, padding
        left = right - (value_width + gap + unit_width + 2 * padding)
        bottom = top + value_height + 2 * padding
        darken_region(image, left, top, right, bottom)

        put_text(image, value_text, (left + padding, bottom - padding), value_scale, Hud._pick_fps_color(state), 2)
        put_text(image, "FPS", (right - padding - unit_width, bottom - padding), unit_scale, GREY, 1)

    @staticmethod
    def _pick_fps_color(state: HudState) -> tuple[int, int, int]:
        """Green when keeping up with the video's frame rate, yellow or red when falling behind."""
        ratio = state.fps / state.video_fps if state.video_fps else 1
        if ratio >= FPS_GOOD_RATIO or state.is_paused:  # a paused video is not "falling behind"
            return FPS_GOOD_COLOR
        return FPS_WARNING_COLOR if ratio >= FPS_WARNING_RATIO else FPS_BAD_COLOR

    @staticmethod
    def _draw_banner(image: np.ndarray, state: HudState, ui_scale: float, padding: int) -> None:
        """A PAUSED or END banner at the top center."""
        if state.has_ended:
            text = "STREAM ENDED - press Q for the menu" if state.is_live else "END - press R to replay"
        else:
            text = "PAUSED"
        font_scale = 0.7 * ui_scale
        (text_width, text_height), _ = cv2.getTextSize(text, FONT, font_scale, 2)
        left = (image.shape[1] - text_width) // 2
        darken_region(image, left - padding, padding, left + text_width + padding, padding + text_height + 2 * padding)
        put_text(image, text, (left, padding + text_height + padding), font_scale, WHITE, 2)

    @staticmethod
    def _draw_info_panel(image: np.ndarray, state: HudState, ui_scale: float, padding: int) -> None:
        """Models, hardware, timings, track count, target and playback position in the top-left corner."""
        position = format_clock(state.video_position_seconds)
        if state.video_duration_seconds > 0:
            position += f" / {format_clock(state.video_duration_seconds)}"

        lines = [
            ("DRONE CAR TRACKER", WHITE),
            (state.models_text, GREY),
            (state.hardware_text, GREY),
            (f"detect {state.detect_ms:4.1f} ms  |  motion+track {state.track_ms:4.1f} ms", GREY),
            (f"cars {state.num_tracks}  |  target {state.target_text}", GREY),
            (f"{position}  |  {Hud._describe_playback(state)}", GREY),
        ]
        draw_text_panel(image, lines, padding, padding, 0.48 * ui_scale, ui_scale)

    @staticmethod
    def _describe_playback(state: HudState) -> str:
        """'live', 'real-time' or 'max speed'."""
        if state.is_live:
            return "live"
        return "real-time" if state.is_real_time else "max speed"

    @staticmethod
    def _draw_help_bar(image: np.ndarray, state: HudState, ui_scale: float, padding: int) -> None:
        """The key bindings along the bottom edge, shrinking the text to fit narrow windows."""
        font_scale = 0.42 * ui_scale
        (text_width, text_height), _ = cv2.getTextSize(state.help_text, FONT, font_scale, 1)

        if text_width + 2 * padding > image.shape[1]:  # too wide: shrink the text until it fits
            font_scale *= (image.shape[1] - 2 * padding) / (text_width + 2 * padding)
            (text_width, text_height), _ = cv2.getTextSize(state.help_text, FONT, font_scale, 1)

        bottom = image.shape[0] - padding
        darken_region(image, padding, bottom - text_height - padding, 2 * padding + text_width, bottom + round(4 * ui_scale))
        put_text(image, state.help_text, (padding + padding // 2, bottom - padding // 2), font_scale, HELP_TEXT_COLOR, 1)

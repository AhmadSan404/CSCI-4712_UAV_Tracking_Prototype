"""Low-level OpenCV drawing helpers: text, translucent panels, box decorations and trails.

Colors are BGR tuples. Sizes that should look the same at any resolution are multiplied by a "UI scale".
"""

import math
from collections.abc import Sequence

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX
LINE_AA = cv2.LINE_AA

WHITE = (255, 255, 255)
GREY = (200, 200, 200)
BLACK = (0, 0, 0)

REFERENCE_HEIGHT_PX = 720  # at this image height the UI scale is exactly 1.0
MIN_UI_SCALE = 0.7
MAX_UI_SCALE = 2.0
BRIGHT_COLOR_THRESHOLD = 380  # if a color's B+G+R exceeds this, black text is more readable on it than white
PANEL_BASE_FONT_SCALE = 0.48  # the font scale that the info panel's spacing numbers were designed around

Color = Sequence[int]


def calculate_ui_scale(image: np.ndarray) -> float:
    """Image height divided by 720, limited to between 0.7 and 2.0."""
    return min(MAX_UI_SCALE, max(MIN_UI_SCALE, image.shape[0] / REFERENCE_HEIGHT_PX))


def to_pixel(point: Sequence[float]) -> tuple[int, int]:
    """Rounds a point to the integer pixel coordinates OpenCV drawing functions require."""
    return int(round(float(point[0]))), int(round(float(point[1])))


def darken(color: Color, factor: float = 0.55) -> tuple[int, ...]:
    """Multiplies every channel of a color by `factor`."""
    return tuple(int(channel * factor) for channel in color)


def format_clock(seconds: float) -> str:
    """Formats seconds as 'MM:SS.s', e.g. '01:05.5'."""
    minutes, remaining_seconds = divmod(max(0.0, seconds), 60)
    return f"{int(minutes):02d}:{remaining_seconds:04.1f}"


def put_text(
    image: np.ndarray, text: str, origin: tuple[int, int], font_scale: float, color: Color, thickness: int = 1
) -> None:
    """Draws anti-aliased text with its bottom-left corner at `origin`."""
    # Always measure text with the same thickness it is drawn with: thicker text is bolder and wider.
    cv2.putText(image, text, origin, FONT, font_scale, color, thickness, LINE_AA)


def darken_region(image: np.ndarray, left: int, top: int, right: int, bottom: int, strength: float = 0.65) -> None:
    """Darkens a rectangle (clipped to the image) so text on top of it stays readable."""
    image_height, image_width = image.shape[:2]
    left, top = max(0, left), max(0, top)
    right, bottom = min(image_width, right), min(image_height, bottom)
    if right > left and bottom > top:
        image[top:bottom, left:right] = cv2.convertScaleAbs(image[top:bottom, left:right], alpha=1 - strength)


def draw_text_panel(
    image: np.ndarray, lines: list[tuple[str, Color]], left: int, top: int, font_scale: float, ui_scale: float
) -> None:
    """Draws lines of (text, color) on one dark panel. The first line is a title, drawn bold."""
    spacing = font_scale / PANEL_BASE_FONT_SCALE
    line_height = round(22 * spacing)
    first_baseline = round(14 * spacing)

    thicknesses = [2] + [1] * (len(lines) - 1)
    text_widths = [
        cv2.getTextSize(text, FONT, font_scale, thickness)[0][0] for (text, _), thickness in zip(lines, thicknesses)
    ]
    padding = round(8 * ui_scale)

    darken_region(image, left, top, left + max(text_widths) + 2 * padding, top + line_height * len(lines) + padding)
    for row, ((text, color), thickness) in enumerate(zip(lines, thicknesses)):
        baseline_y = top + padding + line_height * row + first_baseline
        put_text(image, text, (left + padding, baseline_y), font_scale, color, thickness)


def draw_label(image: np.ndarray, text: str, box_top_left: tuple[int, int], color: Color, ui_scale: float) -> None:
    """Draws a filled label above a box's top-left corner (inside the box if there is no room above)."""
    font_scale = 0.5 * ui_scale
    (text_width, text_height), baseline = cv2.getTextSize(text, FONT, font_scale, 1)
    padding = max(2, round(3 * ui_scale))
    label_width = text_width + 2 * padding
    label_height = text_height + baseline + 2 * padding

    x, y = box_top_left
    top = y - label_height if y - label_height >= 0 else y
    x = min(max(0, x), image.shape[1] - label_width)

    cv2.rectangle(image, (x, top), (x + label_width, top + label_height), color, -1)
    text_color = BLACK if sum(color) > BRIGHT_COLOR_THRESHOLD else WHITE
    put_text(image, text, (x + padding, top + padding + text_height), font_scale, text_color, 1)


def draw_corner_marks(
    image: np.ndarray, top_left: tuple[int, int], bottom_right: tuple[int, int], color: Color, thickness: int
) -> None:
    """Draws L-shaped marks on the four corners of a rectangle."""
    (left, top), (right, bottom) = top_left, bottom_right
    arm_length = max(5, round(0.25 * min(right - left, bottom - top)))

    # (corner x, corner y, direction of the horizontal arm, direction of the vertical arm)
    corners = ((left, top, +1, +1), (right, top, -1, +1), (left, bottom, +1, -1), (right, bottom, -1, -1))
    for corner_x, corner_y, x_direction, y_direction in corners:
        horizontal_end = (corner_x + x_direction * arm_length, corner_y)
        vertical_end = (corner_x, corner_y + y_direction * arm_length)
        cv2.line(image, (corner_x, corner_y), horizontal_end, color, thickness, LINE_AA)
        cv2.line(image, (corner_x, corner_y), vertical_end, color, thickness, LINE_AA)


def draw_dashed_rect(
    image: np.ndarray,
    top_left: tuple[int, int],
    bottom_right: tuple[int, int],
    color: Color,
    thickness: int,
    dash_length_px: int = 8,
) -> None:
    """Draws a rectangle outline made of dashes."""
    (left, top), (right, bottom) = top_left, bottom_right
    edges = (
        ((left, top), (right, top)),
        ((right, top), (right, bottom)),
        ((right, bottom), (left, bottom)),
        ((left, bottom), (left, top)),
    )
    for start, end in edges:
        edge_length = math.hypot(end[0] - start[0], end[1] - start[1])
        if edge_length < 1:
            continue
        for dash_start in np.arange(0, edge_length, 2 * dash_length_px):
            fraction_from = dash_start / edge_length
            fraction_to = min(1.0, (dash_start + dash_length_px) / edge_length)
            cv2.line(
                image,
                _point_along(start, end, fraction_from),
                _point_along(start, end, fraction_to),
                color,
                thickness,
                LINE_AA,
            )


def _point_along(start: tuple[int, int], end: tuple[int, int], fraction: float) -> tuple[int, int]:
    """The point a given fraction of the way from `start` to `end`, rounded to pixels."""
    return round(start[0] + (end[0] - start[0]) * fraction), round(start[1] + (end[1] - start[1]) * fraction)


def draw_trail(image: np.ndarray, points: np.ndarray, color: Color, ui_scale: float) -> None:
    """Draws a fading trail (older segments thinner and darker) with a dot on the newest point."""
    pixels = np.round(points).astype(np.int32)
    last = len(pixels) - 1
    for i in range(1, len(pixels)):
        recency = i / last  # 0 for the oldest segment, 1 for the newest
        segment_color = darken(color, 0.3 + 0.7 * recency)
        segment_thickness = max(1, round((1 + 2.5 * recency) * ui_scale))
        cv2.line(image, tuple(pixels[i - 1]), tuple(pixels[i]), segment_color, segment_thickness, LINE_AA)
    cv2.circle(image, tuple(pixels[-1]), max(2, round(3 * ui_scale)), color, -1, LINE_AA)

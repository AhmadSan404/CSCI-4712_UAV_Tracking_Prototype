"""The OpenCV window: a fixed-size window that fits the screen, plus mouse clicks."""

from typing import NamedTuple

import cv2
import numpy as np

WINDOW_NAME = "Drone Car Tracker"
MAX_SCREEN_WIDTH_FRACTION = 0.85
MAX_SCREEN_HEIGHT_FRACTION = 0.80


class Click(NamedTuple):
    """A mouse click in source-frame pixels."""

    button: str  # 'left' or 'right'
    x: float
    y: float


class DisplayWindow:
    """Shows frames at one fixed size and reports mouse clicks in source-frame coordinates.

    Attributes:
        scale: Ratio of display pixels to source-frame pixels (never above 1.0).
    """

    def __init__(
        self, frame_width: int, frame_height: int, screen_width: int, screen_height: int, title: str
    ) -> None:
        self.scale = min(
            MAX_SCREEN_WIDTH_FRACTION * screen_width / frame_width,
            MAX_SCREEN_HEIGHT_FRACTION * screen_height / frame_height,
            1.0,
        )
        self._display_size = (round(frame_width * self.scale), round(frame_height * self.scale))
        self._pending_clicks: list[Click] = []

        # AUTOSIZE fixes the window to the image size, so one scale factor maps clicks back to the frame.
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        cv2.setWindowTitle(WINDOW_NAME, title)
        cv2.setMouseCallback(WINDOW_NAME, self._on_mouse_event)

    def resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Shrinks a source frame to the display size (a copy is returned even when no resize is needed)."""
        if self.scale >= 1.0:
            return frame.copy()
        # INTER_LINEAR is much cheaper than INTER_AREA and looks fine for the mild shrink (to ~80%) used here.
        return cv2.resize(frame, self._display_size, interpolation=cv2.INTER_LINEAR)

    def show(self, image: np.ndarray) -> int:
        """Displays an image and returns the pressed key code, or -1 if none."""
        cv2.imshow(WINDOW_NAME, image)
        # waitKey(1) would block for ~15 ms on Windows; pollKey returns immediately.
        key = cv2.pollKey()
        return key & 0xFF if key != -1 else -1

    def is_open(self) -> bool:
        """False once the user has closed the window."""
        try:
            return cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) >= 1
        except cv2.error:
            return False

    def take_clicks(self) -> list[Click]:
        """Returns the clicks received since the last call, oldest first."""
        clicks, self._pending_clicks = self._pending_clicks, []
        return clicks

    def close(self) -> None:
        """Closes the window."""
        cv2.destroyAllWindows()
        cv2.pollKey()  # lets the window system process the close

    def _on_mouse_event(self, event: int, x: int, y: int, flags: int, param: object) -> None:
        """OpenCV mouse callback: queues left and right clicks, converted to source-frame pixels."""
        if event not in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
            return
        button = "left" if event == cv2.EVENT_LBUTTONDOWN else "right"
        self._pending_clicks.append(Click(button, x / self.scale, y / self.scale))

"""Real-time pacing and frame-interval measurement for playback."""

import time

LATE_FRAME_TOLERANCE_SECONDS = 0.1  # if processing falls this far behind real time, stop trying to catch up


class PlaybackPacer:
    """Sleeps so frames appear at the video's frame rate, and measures the time between shown frames."""

    def __init__(self, video_fps: float) -> None:
        self._frame_seconds = 1.0 / video_fps
        self._next_frame_due: float | None = None
        self._previous_shown_time: float | None = None

    def sleep_until_next_frame_is_due(self) -> None:
        """Sleeps until the next frame is due; restarts the schedule if processing fell behind."""
        now = time.perf_counter()
        if self._next_frame_due is None or now - self._next_frame_due > LATE_FRAME_TOLERANCE_SECONDS:
            self._next_frame_due = now
        elif self._next_frame_due > now:
            time.sleep(self._next_frame_due - now)
        self._next_frame_due += self._frame_seconds

    def mark_frame_shown(self) -> float | None:
        """Notes that a frame was just shown; returns the time since the previous one (None for the first)."""
        now = time.perf_counter()
        interval = None if self._previous_shown_time is None else now - self._previous_shown_time
        self._previous_shown_time = now
        return interval

    def reset(self) -> None:
        """Forgets the schedule and the previous frame, e.g. after pausing, restarting or changing playback mode."""
        self._next_frame_due = None
        self._previous_shown_time = None

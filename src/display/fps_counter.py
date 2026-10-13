"""A frames-per-second counter that is smooth and readable."""

import time
from collections import deque

WINDOW_SECONDS = 1.0
REFRESH_SECONDS = 0.25  # the shown value only changes this often, so the on-screen number does not flicker


class FpsCounter:
    """Events per second over a sliding window; call `tick` once per frame (or once per detection pass)."""

    def __init__(self) -> None:
        self.value = 0.0
        self._tick_times: deque[float] = deque()
        self._last_refresh_time = 0.0

    def tick(self) -> None:
        """Records that one frame has just been shown."""
        now = time.perf_counter()
        self._tick_times.append(now)
        while now - self._tick_times[0] > WINDOW_SECONDS:
            self._tick_times.popleft()

        is_refresh_due = now - self._last_refresh_time >= REFRESH_SECONDS
        if is_refresh_due and len(self._tick_times) >= 2:
            elapsed = self._tick_times[-1] - self._tick_times[0]
            self.value = (len(self._tick_times) - 1) / elapsed
            self._last_refresh_time = now

    def reset(self) -> None:
        """Forgets recorded ticks, e.g. after pausing, so stale timings do not skew the reading."""
        self._tick_times.clear()

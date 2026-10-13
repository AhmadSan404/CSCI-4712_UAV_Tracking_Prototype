"""Running statistics for one run. Nothing here stores every detection, only counts, sums and extremes."""

import math

SLOWEST_FRAME_FRACTION = 0.01  # the "1% low" looks at the slowest 1% of frame intervals
MILLISECONDS_PER_SECOND = 1000.0

TARGET_STATE_TRACKING = "tracking"
TARGET_STATE_COASTING = "coasting"


class RunningStatistics:
    """Count, mean, minimum and maximum of a stream of numbers."""

    def __init__(self) -> None:
        self.count = 0
        self._total = 0.0
        self.minimum = math.inf
        self.maximum = -math.inf

    def add(self, value: float) -> None:
        """Includes one more value."""
        self.count += 1
        self._total += value
        self.minimum = min(self.minimum, value)
        self.maximum = max(self.maximum, value)

    @property
    def mean(self) -> float:
        """Average of all values, or 0.0 if there are none."""
        return self._total / self.count if self.count else 0.0

    def to_dict(self, digits: int = 3) -> dict:
        """Count, mean, min and max as a JSON-friendly dict."""
        if not self.count:
            return {"count": 0, "mean": None, "min": None, "max": None}
        return {
            "count": self.count,
            "mean": round(self.mean, digits),
            "min": round(self.minimum, digits),
            "max": round(self.maximum, digits),
        }


class FrameTimingStatistics:
    """Frame intervals (wall-clock time between displayed frames) and per-frame work time."""

    def __init__(self) -> None:
        self.frames_displayed = 0
        self._intervals_seconds: list[float] = []
        self._work_seconds = RunningStatistics()

    def add_frame(self, interval_seconds: float | None, work_seconds: float) -> None:
        """Records a displayed frame. The interval is None for the first frame after a pause or restart."""
        self.frames_displayed += 1
        self._work_seconds.add(work_seconds)
        if interval_seconds is not None:
            self._intervals_seconds.append(interval_seconds)

    @property
    def average_fps(self) -> float:
        """Displayed frames per second: 1 / mean interval."""
        if not self._intervals_seconds:
            return 0.0
        return len(self._intervals_seconds) / sum(self._intervals_seconds)

    @property
    def one_percent_low_fps(self) -> float:
        """1 / (mean of the slowest 1% of frame intervals)."""
        if not self._intervals_seconds:
            return 0.0
        slowest_count = max(1, math.ceil(SLOWEST_FRAME_FRACTION * len(self._intervals_seconds)))
        slowest = sorted(self._intervals_seconds, reverse=True)[:slowest_count]
        return slowest_count / sum(slowest)

    @property
    def capacity_fps(self) -> float:
        """How fast frames could be processed without the real-time pacing sleep: 1 / mean work time."""
        return 1.0 / self._work_seconds.mean if self._work_seconds.mean else 0.0

    def to_dict(self) -> dict:
        """The statistics as a JSON-friendly dict."""
        return {
            "frames_displayed": self.frames_displayed,
            "average_fps": round(self.average_fps, 2),
            "one_percent_low_fps": round(self.one_percent_low_fps, 2),
            "average_work_ms_per_frame": round(self._work_seconds.mean * MILLISECONDS_PER_SECOND, 2),
            "processing_capacity_fps": round(self.capacity_fps, 2),
        }


class DetectionStatistics:
    """Detection pass timing and per-class confidence."""

    def __init__(self) -> None:
        self._detect_ms = RunningStatistics()
        self._confidence_by_class: dict[str, RunningStatistics] = {}

    @property
    def passes(self) -> int:
        """Number of detection passes run."""
        return self._detect_ms.count

    def add_pass(self, detect_ms: float, detections: list[tuple[str, float]]) -> None:
        """Records one detection pass with its (class name, confidence) results."""
        self._detect_ms.add(detect_ms)
        for class_name, confidence in detections:
            self._confidence_by_class.setdefault(class_name, RunningStatistics()).add(confidence)

    def to_dict(self) -> dict:
        """The statistics as a JSON-friendly dict."""
        return {
            "detection_passes": self.passes,
            "average_detect_ms": round(self._detect_ms.mean, 2),
            "max_detect_ms": round(self._detect_ms.maximum, 2) if self.passes else 0.0,
            "confidence_by_class": {name: stats.to_dict() for name, stats in self._confidence_by_class.items()},
        }


class TrackingStatistics:
    """How the tracker and the target behaved: tracks created, target matched or coasting, target changes, cuts."""

    def __init__(self, frame_seconds: float) -> None:
        self._frame_seconds = frame_seconds
        self._seen_track_ids: set[int] = set()
        self.event_counts: dict[str, int] = {}
        self.frames_target_matched = 0
        self.frames_target_coasting = 0
        self.frames_target_locked = 0
        self.frames_target_automatic = 0
        self.longest_coasting_seconds = 0.0
        self._current_coasting_frames = 0

    def add_frame(self, track_ids: list[int], target_state: str | None, is_target_locked: bool) -> None:
        """Records one displayed frame: the track IDs present and the target's state (None if no target)."""
        self._seen_track_ids.update(track_ids)
        if target_state == TARGET_STATE_TRACKING:
            self.frames_target_matched += 1
        elif target_state == TARGET_STATE_COASTING:
            self.frames_target_coasting += 1
        if target_state is not None:
            if is_target_locked:
                self.frames_target_locked += 1
            else:
                self.frames_target_automatic += 1

        if target_state == TARGET_STATE_COASTING:
            self._current_coasting_frames += 1
            self.longest_coasting_seconds = max(
                self.longest_coasting_seconds, self._current_coasting_frames * self._frame_seconds
            )
        else:
            self._current_coasting_frames = 0

    def add_event(self, event: str) -> None:
        """Counts one event such as 'target_changed' or 'scene_cut'."""
        self.event_counts[event] = self.event_counts.get(event, 0) + 1

    def to_dict(self) -> dict:
        """The statistics as a JSON-friendly dict."""
        return {
            "tracks_created": len(self._seen_track_ids),
            "target_changes": self.event_counts.get("target_changed", 0),
            "frames_target_matched": self.frames_target_matched,
            "frames_target_coasting": self.frames_target_coasting,
            "frames_target_locked": self.frames_target_locked,
            "frames_target_automatic": self.frames_target_automatic,
            "longest_coasting_seconds": round(self.longest_coasting_seconds, 2),
            "scene_cuts": self.event_counts.get("scene_cut", 0),
        }

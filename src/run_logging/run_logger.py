"""Writes one log per run: a detailed JSON file, a row in the run-history CSV, and a console summary."""

import json
import time
from datetime import datetime

from src.project_paths import LOGS_DIR
from src.run_logging.environment_info import EnvironmentInfo
from src.run_logging.run_history import RunHistory
from src.run_logging.run_statistics import (
    DetectionStatistics,
    FrameTimingStatistics,
    RunningStatistics,
    TrackingStatistics,
)


class RunLogger:
    """Collects everything about one run and writes it out in `finish_run`.

    Call `start_run` once, the `record_*` methods while running, and `finish_run` last (also after a crash).
    """

    def __init__(self) -> None:
        self._started_at = datetime.now()
        self._started_clock = time.perf_counter()
        self._active_seconds = 0.0
        self._active_since: float | None = None
        self._start_data: dict = {}
        self._environment: dict = {}
        self._live_changes: list[dict] = []
        self._events: list[dict] = []
        self._frame_timing = FrameTimingStatistics()
        self._detection = DetectionStatistics()
        self._tracking: TrackingStatistics | None = None
        self._peak_vram_mb = RunningStatistics()

    def start_run(self, video: dict, models: dict, settings: dict, hardware: dict) -> None:
        """Stores the facts known at the start: video, models, chosen settings and hardware (plain dicts)."""
        self._start_data = {"video": video, "models": models, "settings_used": settings}
        self._environment = EnvironmentInfo.collect(
            hardware["device_name"], hardware["precision"], hardware["total_vram_mb"]
        )
        self._tracking = TrackingStatistics(1.0 / video["fps"])
        self._active_since = time.perf_counter()

    def record_frame(
        self,
        interval_seconds: float | None,
        work_seconds: float,
        track_ids: list[int],
        target_state: str | None,
        is_target_locked: bool,
    ) -> None:
        """Records a displayed frame: time since the previous one (None if excluded), work time, tracks, target."""
        self._frame_timing.add_frame(interval_seconds, work_seconds)
        self._tracking.add_frame(track_ids, target_state, is_target_locked)

    def record_detection_pass(
        self, detect_ms: float, detections: list[tuple[str, float]], vram_used_mb: float | None
    ) -> None:
        """Records one detection pass: its time, the (class name, confidence) results and the VRAM in use."""
        self._detection.add_pass(detect_ms, detections)
        if vram_used_mb is not None:
            self._peak_vram_mb.add(vram_used_mb)

    def record_target_event(self, event: str, video_seconds: float, detail: str = "") -> None:
        """Records something that happened: target_changed (detail says to which track) or scene_cut."""
        self._tracking.add_event(event)
        self._events.append(
            {"run_seconds": self._run_seconds(), "video_seconds": round(video_seconds, 2), "event": event, "detail": detail}
        )

    def record_setting_change(self, name: str, value: object) -> None:
        """Records a setting changed while running, such as detect_every_n_frames."""
        self._live_changes.append({"run_seconds": self._run_seconds(), "setting": name, "value": value})

    def record_playback_active(self, is_active: bool) -> None:
        """Tells the logger whether playback is running (False while paused or ended), for the active duration."""
        now = time.perf_counter()
        if is_active and self._active_since is None:
            self._active_since = now
        elif not is_active and self._active_since is not None:
            self._active_seconds += now - self._active_since
            self._active_since = None

    def finish_run(self, end_reason: str, error_message: str | None = None) -> None:
        """Writes the JSON log, appends the history row and prints the summary. Never raises."""
        try:
            self.record_playback_active(False)
            report = self._build_report(end_reason, error_message)
            LOGS_DIR.mkdir(exist_ok=True)
            json_path = LOGS_DIR / f"run_{self._started_at:%Y%m%d_%H%M%S}.json"
            json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            RunHistory.append_row(report)
            self._print_summary(report, json_path.name)
        except Exception as error:  # a logging problem must never hide the real error or crash the app
            print(f"Warning: could not write the run log: {error}")

    def _run_seconds(self) -> float:
        """Wall-clock seconds since the run started."""
        return round(time.perf_counter() - self._started_clock, 2)

    def _build_report(self, end_reason: str, error_message: str | None) -> dict:
        """Assembles everything into the JSON structure."""
        ended_at = datetime.now()
        settings_used = self._start_data.get("settings_used", {})
        environment = dict(self._environment)
        environment["peak_vram_used_mb"] = round(self._peak_vram_mb.maximum) if self._peak_vram_mb.count else None
        return {
            "run": {
                "started_at": self._started_at.isoformat(timespec="seconds"),
                "ended_at": ended_at.isoformat(timespec="seconds"),
                "active_seconds": round(self._active_seconds, 2),
                "end_reason": end_reason,
                "error_message": error_message,
            },
            "video": self._start_data.get("video"),
            "models": self._start_data.get("models"),
            "classes": {
                "chosen": settings_used.get("selected_class_names"),
                "missing_from_a_model": self._start_data.get("models", {}).get("missing_classes", []),
            },
            "settings": {"used": settings_used, "live_changes": self._live_changes},
            "fps": self._frame_timing.to_dict(),
            "detection": {"frames_displayed": self._frame_timing.frames_displayed, **self._detection.to_dict()},
            "tracking": {**self._tracking.to_dict(), "events": self._events},
            "environment": environment,
        }

    @staticmethod
    def _print_summary(report: dict, json_name: str) -> None:
        """Prints a short summary of the run to the console."""
        fps, detection, tracking = report["fps"], report["detection"], report["tracking"]
        print(f"\nRun finished ({report['run']['end_reason']}), {report['run']['active_seconds']:.1f} s active.")
        print(
            f"  FPS: {fps['average_fps']} average, {fps['one_percent_low_fps']} 1% low, "
            f"capacity {fps['processing_capacity_fps']}"
        )
        print(f"  Detection: {detection['detection_passes']} passes, {detection['average_detect_ms']} ms average")
        print(
            f"  Tracking: {tracking['tracks_created']} tracks, {tracking['target_changes']} target changes, "
            f"{tracking['scene_cuts']} scene cuts"
        )
        print(f"  Log: logs/{json_name}")

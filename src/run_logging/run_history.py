"""The run-history CSV: one summary row per run, for comparing runs."""

import csv

from src.project_paths import LOGS_DIR

HISTORY_FILE_NAME = "run_history.csv"
HISTORY_COLUMNS = [
    "started_at", "end_reason", "active_seconds", "video_name", "models", "device_name", "precision",
    "classes", "min_confidence", "show_trails", "playback_mode",
    "average_fps", "one_percent_low_fps", "average_work_ms_per_frame", "processing_capacity_fps",
    "detection_passes", "average_detect_ms", "tracks_created", "target_changes", "scene_cuts", "peak_vram_mb",
]  # fmt: skip


class RunHistory:
    """Appends one row per run to `logs/run_history.csv`."""

    @staticmethod
    def append_row(report: dict) -> None:
        """Adds a summary of a finished run's report, writing the header first if the file is new."""
        history_path = LOGS_DIR / HISTORY_FILE_NAME
        is_new_file = not history_path.exists()
        with history_path.open("a", newline="", encoding="utf-8") as history_file:
            writer = csv.DictWriter(history_file, fieldnames=HISTORY_COLUMNS)
            if is_new_file:
                writer.writeheader()
            writer.writerow(RunHistory._build_row(report))

    @staticmethod
    def _build_row(report: dict) -> dict:
        """Picks the key columns out of a run report."""
        settings = report["settings"]["used"]
        run, fps = report["run"], report["fps"]
        detection, tracking = report["detection"], report["tracking"]
        environment = report["environment"]
        return {
            "started_at": run["started_at"],
            "end_reason": run["end_reason"],
            "active_seconds": run["active_seconds"],
            "video_name": report["video"]["name"],
            "models": report["models"]["description"],
            "device_name": environment["device_name"],
            "precision": environment["precision"],
            "classes": "+".join(settings["selected_class_names"]),
            "min_confidence": settings["min_confidence"],
            "show_trails": settings["show_trails"],
            "playback_mode": "max speed" if settings["should_start_at_max_speed"] else "real-time",
            "average_fps": fps["average_fps"],
            "one_percent_low_fps": fps["one_percent_low_fps"],
            "average_work_ms_per_frame": fps["average_work_ms_per_frame"],
            "processing_capacity_fps": fps["processing_capacity_fps"],
            "detection_passes": detection["detection_passes"],
            "average_detect_ms": detection["average_detect_ms"],
            "tracks_created": tracking["tracks_created"],
            "target_changes": tracking["target_changes"],
            "scene_cuts": tracking["scene_cuts"],
            "peak_vram_mb": environment["peak_vram_used_mb"],
        }

"""Loading and saving the last-used settings as JSON."""

import dataclasses
import json
from pathlib import Path

from src.detection.vehicle_types import find_vehicle_class
from src.menu.tracker_settings import (
    DEVICE_CHOICES,
    MAX_CONFIDENCE,
    MAX_IMAGE_SIZE_PX,
    MIN_CONFIDENCE,
    MIN_IMAGE_SIZE_PX,
    SOURCE_CHOICES,
    TrackerSettings,
)
from src.project_paths import SETTINGS_FILE


def load_settings(settings_file: Path = SETTINGS_FILE) -> TrackerSettings:
    """Loads saved settings; anything missing, wrong or corrupt falls back to the defaults."""
    settings = TrackerSettings()
    try:
        saved = json.loads(settings_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return settings
    if not isinstance(saved, dict):
        return settings

    for field in dataclasses.fields(settings):
        if field.name in saved:
            setattr(settings, field.name, _coerce(saved[field.name], getattr(settings, field.name)))
    _repair_invalid_values(settings)
    return settings


def save_settings(settings: TrackerSettings, settings_file: Path = SETTINGS_FILE) -> None:
    """Writes the settings to disk. A failure only prints a warning, since this is a convenience."""
    try:
        settings_file.write_text(json.dumps(dataclasses.asdict(settings), indent=2), encoding="utf-8")
    except OSError as error:
        print(f"Warning: could not save settings: {error}")


def _coerce(saved_value: object, default_value: object) -> object:
    """Returns the saved value if it has the same type as the default, otherwise the default."""
    is_number = isinstance(saved_value, (int, float)) and not isinstance(saved_value, bool)
    if isinstance(default_value, bool):
        return saved_value if isinstance(saved_value, bool) else default_value
    if isinstance(default_value, int):
        return int(saved_value) if is_number else default_value
    if isinstance(default_value, float):
        return float(saved_value) if is_number else default_value
    if isinstance(default_value, str):
        return saved_value if isinstance(saved_value, str) else default_value
    if isinstance(default_value, list):
        is_string_list = isinstance(saved_value, list) and all(isinstance(item, str) for item in saved_value)
        return saved_value if is_string_list else default_value
    return default_value


def _repair_invalid_values(settings: TrackerSettings) -> None:
    """Clamps numbers into their allowed range and resets values that make no sense."""
    defaults = TrackerSettings()
    settings.min_confidence = min(MAX_CONFIDENCE, max(MIN_CONFIDENCE, settings.min_confidence))
    settings.main_image_size_px = min(MAX_IMAGE_SIZE_PX, max(MIN_IMAGE_SIZE_PX, settings.main_image_size_px))
    settings.aerial_image_size_px = min(MAX_IMAGE_SIZE_PX, max(MIN_IMAGE_SIZE_PX, settings.aerial_image_size_px))
    if settings.device_choice not in DEVICE_CHOICES:
        settings.device_choice = defaults.device_choice
    if settings.source_type not in SOURCE_CHOICES:
        settings.source_type = defaults.source_type

    settings.selected_class_names = [name for name in settings.selected_class_names if find_vehicle_class(name)]
    if not settings.selected_class_names:
        settings.selected_class_names = defaults.selected_class_names
    if not Path(settings.video_path).is_file():
        settings.video_path = ""  # the menu then picks the first video in videos/

"""Entry point: menu -> load models -> run session -> save settings -> menu again."""

import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

from src.detection.model_catalog import resolve_weights_path
from src.detection.vehicle_types import VehicleClass, find_vehicle_class
from src.menu.menu_window import MenuResult, show_error_dialog, show_menu
from src.menu.settings_storage import load_settings, save_settings
from src.menu.tracker_settings import NO_AERIAL_MODEL, TrackerSettings


def enable_windows_dpi_awareness() -> None:
    """Makes the process DPI-aware, so windows are sharp and screen sizes are real pixels. Must run first."""
    if sys.platform == "win32":
        import ctypes

        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)  # 2 = per-monitor aware
        except (AttributeError, OSError):
            pass  # already set (for example by a manifest), or not supported


@dataclass
class DetectorCache:
    """The most recently loaded detector, reused while the models, sizes, classes and device stay the same."""

    key: tuple | None = None
    detector: object | None = None


def load_detector(settings: TrackerSettings, cache: DetectorCache) -> tuple[object, dict]:
    """Loads (or reuses) the detector for these settings.

    Returns:
        The detector and a plain dict describing the models for the run log.

    Raises:
        ValueError: If a model has none of the chosen classes.
        RuntimeError: If the GPU is requested but unavailable.
    """
    # Importing the detector imports torch and ultralytics, which is slow, so it waits until after Start.
    from src.detection.vehicle_detector import VehicleDetector

    main_path = resolve_weights_path(settings.main_model_name)
    uses_aerial = settings.aerial_model_name != NO_AERIAL_MODEL
    aerial_path = resolve_weights_path(settings.aerial_model_name) if uses_aerial else None
    classes = [find_vehicle_class(name) for name in settings.selected_class_names]
    if settings.track_people:
        classes.append(VehicleClass.PERSON)
    key = (main_path, settings.main_image_size_px, aerial_path, settings.aerial_image_size_px,
           settings.device_choice, tuple(c.value for c in classes))

    model_info = {"main_weights": main_path, "aerial_weights": aerial_path, "was_cached": cache.key == key}
    if cache.detector is None or cache.key != key:
        model_info["was_downloaded"] = not Path(main_path).exists() or (uses_aerial and not Path(aerial_path).exists())
        print("Loading models" + (" (downloading first)..." if model_info["was_downloaded"] else "..."))
        started = time.perf_counter()
        cache.detector = None  # drop the old models first so both are never in memory at once
        cache.detector = VehicleDetector(
            main_path, settings.main_image_size_px, aerial_path, settings.aerial_image_size_px,
            settings.min_confidence, settings.device_choice, classes,
        )
        cache.key = key
        model_info["load_seconds"] = round(time.perf_counter() - started, 2)
        print(f"Loaded {cache.detector.description} in {model_info['load_seconds']} s.")
    else:
        model_info.update(was_downloaded=False, load_seconds=0.0)

    cache.detector.min_confidence = settings.min_confidence  # cheap to change without reloading
    missing_names = [vehicle_class.display_name for vehicle_class in cache.detector.missing_classes]
    model_info["missing_classes"] = missing_names
    model_info["tasks"] = [model.task for model in cache.detector.models]
    if missing_names:
        print(f"Note: at least one model has no class {missing_names}; those are ignored for that model.")
    return cache.detector, model_info


def run_one_session(menu_result: MenuResult, cache: DetectorCache) -> None:
    """Loads the models and plays one session. Problems are shown in a dialog so the menu can reopen."""
    from src.tracking_session import TrackingSession

    settings = menu_result.settings
    try:
        detector, model_info = load_detector(settings, cache)
    except (ValueError, RuntimeError, OSError) as error:
        show_error_dialog("Could not load the models", str(error))
        return

    try:
        screen_size = (menu_result.screen_width, menu_result.screen_height)
        TrackingSession(settings, detector, screen_size, model_info).run()
    except Exception as error:
        traceback.print_exc()
        show_error_dialog("Tracking stopped", f"{type(error).__name__}: {error}")


def main() -> None:
    """Shows the menu again and again until the user quits it."""
    enable_windows_dpi_awareness()
    settings = load_settings()
    cache = DetectorCache()
    while True:
        menu_result = show_menu(settings)
        if menu_result is None:
            break
        settings = menu_result.settings
        save_settings(settings)
        run_one_session(menu_result, cache)
        save_settings(settings)  # live keys changed some options during the run


if __name__ == "__main__":
    main()

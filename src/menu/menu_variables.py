"""The tkinter variables behind the menu fields, and their conversion to and from `TrackerSettings`."""

import tkinter as tk
from pathlib import Path

from src.detection.model_catalog import ModelChoice
from src.detection.vehicle_types import VEHICLE_ONLY_CLASSES
from src.menu.tracker_settings import NO_AERIAL_MODEL, TrackerSettings
from src.project_paths import VIDEO_EXTENSIONS, VIDEOS_DIR

NO_AERIAL_LABEL = "None (use only the main model)"
PREFERRED_CAMERA_TEXT = "OBS"  # with no saved camera, the first one whose name contains this is selected
OTHER_VIDEO_LABEL = "Other..."  # the last video dropdown entry: the video is then chosen with the text box / Browse


def list_video_names() -> list[str]:
    """File names of the videos in the videos folder, sorted."""
    if not VIDEOS_DIR.exists():
        return []
    return sorted(path.name for path in VIDEOS_DIR.iterdir() if path.suffix.lower() in VIDEO_EXTENSIONS)


class MenuVariables:
    """One tkinter variable per menu option. A tk root window must already exist."""

    def __init__(self, model_choices: list[ModelChoice]) -> None:
        self.main_choices = [choice for choice in model_choices if not choice.is_obb]
        self.aerial_choices = [choice for choice in model_choices if choice.is_obb]
        self.video_names = list_video_names()
        self.camera_names: list[str] = []  # filled by `set_camera_names` after a scan
        self.source_type = tk.StringVar()  # SOURCE_VIDEO or SOURCE_STREAM
        self.camera_name = tk.StringVar()
        self.video_choice = tk.StringVar()  # a file name from the videos folder, or OTHER_VIDEO_LABEL
        self.video_path = tk.StringVar()  # the text box: only used while the choice is OTHER_VIDEO_LABEL
        self.main_model_label = tk.StringVar()
        self.main_image_size_px = tk.IntVar()
        self.aerial_model_label = tk.StringVar()
        self.aerial_image_size_px = tk.IntVar()
        self.min_confidence = tk.DoubleVar()
        self.class_flags = {vehicle_class.display_name: tk.BooleanVar() for vehicle_class in VEHICLE_ONLY_CLASSES}
        self.track_people = tk.BooleanVar()
        self.show_trails = tk.BooleanVar()
        self.should_start_at_max_speed = tk.BooleanVar()
        self.device_label = tk.StringVar()

    def apply(self, settings: TrackerSettings) -> None:
        """Fills every variable from a settings object (used at startup and for 'Reset to defaults')."""
        defaults = TrackerSettings()
        self.source_type.set(settings.source_type)
        self._apply_video(settings.video_path)
        self.camera_name.set(self._pick_camera(settings.camera_name))
        self.main_model_label.set(self._find_label(settings.main_model_name, defaults.main_model_name, self.main_choices))
        self.main_image_size_px.set(settings.main_image_size_px)
        self.aerial_model_label.set(
            self._find_label(settings.aerial_model_name, defaults.aerial_model_name, self.aerial_choices)
        )
        self.aerial_image_size_px.set(settings.aerial_image_size_px)
        self.min_confidence.set(settings.min_confidence)
        for name, flag in self.class_flags.items():
            flag.set(name in settings.selected_class_names)
        self.track_people.set(settings.track_people)
        self.show_trails.set(settings.show_trails)
        self.should_start_at_max_speed.set(settings.should_start_at_max_speed)
        self.device_label.set(settings.device_choice.upper())

    def collect(self) -> TrackerSettings:
        """Builds a settings object from the current variable values."""
        return TrackerSettings(
            source_type=self.source_type.get(),
            video_path=self._chosen_video_path(),
            camera_name=self.camera_name.get(),
            main_model_name=self._find_file_name(self.main_model_label.get(), self.main_choices),
            main_image_size_px=self.main_image_size_px.get(),
            aerial_model_name=self._find_file_name(self.aerial_model_label.get(), self.aerial_choices),
            aerial_image_size_px=self.aerial_image_size_px.get(),
            min_confidence=round(self.min_confidence.get(), 2),
            selected_class_names=[name for name, flag in self.class_flags.items() if flag.get()],
            track_people=self.track_people.get(),
            show_trails=self.show_trails.get(),
            should_start_at_max_speed=self.should_start_at_max_speed.get(),
            device_choice=self.device_label.get().lower(),
        )

    def set_camera_names(self, names: list[str]) -> None:
        """Replaces the camera list after a scan, keeping the selected camera if it is still there."""
        self.camera_names = names
        self.camera_name.set(self._pick_camera(self.camera_name.get()))

    def _pick_camera(self, wanted_name: str) -> str:
        """The wanted camera if it is connected, else the first OBS camera, else the first camera, else ''."""
        if wanted_name in self.camera_names:
            return wanted_name
        preferred = [name for name in self.camera_names if PREFERRED_CAMERA_TEXT.lower() in name.lower()]
        return (preferred or self.camera_names or [""])[0]

    def _apply_video(self, video_path: str) -> None:
        """Selects the video: its dropdown entry if it is in the videos folder, else 'Other...' with the path typed in.

        With no saved video, the first one in the videos folder is selected (or 'Other...' if the folder is empty).
        """
        if not video_path:
            self.video_choice.set(self.video_names[0] if self.video_names else OTHER_VIDEO_LABEL)
            return
        path = Path(video_path)
        is_listed = path.name in self.video_names and path.parent.resolve() == VIDEOS_DIR.resolve()
        self.video_choice.set(path.name if is_listed else OTHER_VIDEO_LABEL)
        self.video_path.set("" if is_listed else video_path)

    def _chosen_video_path(self) -> str:
        """The full path of the chosen video."""
        choice = self.video_choice.get()
        if choice == OTHER_VIDEO_LABEL:
            return self.video_path.get().strip()
        return str(VIDEOS_DIR / choice)

    @staticmethod
    def _find_label(model_name: str, fallback_name: str, choices: list[ModelChoice]) -> str:
        """The dropdown label for a model file name (or for 'none'), else the fallback model's, else the first."""
        if model_name == NO_AERIAL_MODEL:
            return NO_AERIAL_LABEL
        for wanted_name in (model_name, fallback_name):
            for choice in choices:
                if choice.file_name == wanted_name:
                    return choice.label
        return choices[0].label

    @staticmethod
    def _find_file_name(label: str, choices: list[ModelChoice]) -> str:
        """The model file name for a dropdown label ('none' for the no-aerial entry)."""
        if label == NO_AERIAL_LABEL:
            return NO_AERIAL_MODEL
        return next(choice.file_name for choice in choices if choice.label == label)

"""Every option chosen in the menu, with its default and limits. The defaults are the ones that were tuned on the
sample footage, so only change them if you know why."""

from dataclasses import dataclass, field

from src.detection.vehicle_types import VEHICLE_ONLY_CLASSES

NO_AERIAL_MODEL = "none"  # the value of `aerial_model_name` when only the main model should run
DEVICE_CHOICES = ("auto", "cpu", "gpu")
SOURCE_VIDEO = "video"  # play a video file
SOURCE_STREAM = "stream"  # read a live camera, such as OBS Virtual Camera
SOURCE_CHOICES = (SOURCE_VIDEO, SOURCE_STREAM)

MIN_CONFIDENCE = 0.05
MAX_CONFIDENCE = 0.90
CONFIDENCE_STEP = 0.05
MIN_IMAGE_SIZE_PX = 320
MAX_IMAGE_SIZE_PX = 1920
IMAGE_SIZE_STEP_PX = 32  # YOLO input sizes must be a multiple of 32


@dataclass
class TrackerSettings:
    """The options for one tracking run. Live keys change some of these in place."""

    source_type: str = SOURCE_VIDEO
    video_path: str = ""
    camera_name: str = ""  # cameras are remembered by name, since their numbers change when devices come and go
    main_model_name: str = "yolo26m.pt"  # COCO model: good on low and oblique views
    main_image_size_px: int = 960
    aerial_model_name: str = "yolo26m-obb.pt"  # DOTA aerial model: good on cars seen from straight above
    aerial_image_size_px: int = 1280
    min_confidence: float = 0.10  # a low floor on purpose: the tracker applies its own stricter thresholds
    selected_class_names: list[str] = field(
        default_factory=lambda: [vehicle_class.display_name for vehicle_class in VEHICLE_ONLY_CLASSES]
    )
    track_people: bool = False  # also detect and track pedestrians (only the main model has a person class)
    show_trails: bool = True
    should_start_at_max_speed: bool = False
    device_choice: str = "auto"

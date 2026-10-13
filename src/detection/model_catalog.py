"""Which YOLO models are installed, which can be downloaded, and where their weights live."""

from dataclasses import dataclass
from pathlib import Path

from src.project_paths import MODELS_DIR

# Checked against ultralytics.utils.downloads.GITHUB_ASSETS_NAMES (standard detection and aerial OBB only).
MODEL_SIZES = ("n", "s", "m", "l", "x")
SIZE_DESCRIPTIONS = {
    "n": "fastest, least accurate",
    "s": "fast, decent accuracy",
    "m": "balanced speed and accuracy",
    "l": "slower, more accurate",
    "x": "slowest, most accurate",
}


@dataclass(frozen=True)
class ModelChoice:
    """One entry of the model dropdown."""

    file_name: str
    label: str
    is_installed: bool
    is_obb: bool = False  # aerial oriented-box model (the file name ends in -obb.pt); otherwise a detection model


def list_installed_model_names() -> list[str]:
    """Lists the weights files found in the models folder."""
    if not MODELS_DIR.exists():
        return []
    return sorted(path.name for path in MODELS_DIR.glob("*.pt"))


def list_downloadable_model_names() -> list[str]:
    """Lists every YOLO26 detection and OBB model that can be downloaded, smallest first."""
    standard = [f"yolo26{size}.pt" for size in MODEL_SIZES]
    aerial = [f"yolo26{size}-obb.pt" for size in MODEL_SIZES]
    return standard + aerial


def describe_model(name: str) -> str:
    """A short description such as 'balanced speed and accuracy', or '' for a model that is not a standard size."""
    size = name.removesuffix(".pt").removesuffix("-obb")[-1]
    if name not in list_downloadable_model_names():
        return ""
    description = SIZE_DESCRIPTIONS[size]
    return f"aerial view, {description}" if name.endswith("-obb.pt") else description


def list_model_choices() -> list[ModelChoice]:
    """Builds the dropdown entries: installed models first, then the ones that would be downloaded."""
    installed_names = list_installed_model_names()
    choices = []
    for name in installed_names:
        description = describe_model(name)
        label = f"{name}  (installed)" + (f" - {description}" if description else "")
        choices.append(ModelChoice(name, label, True, name.endswith("-obb.pt")))

    for name in list_downloadable_model_names():
        if name not in installed_names:
            label = f"{name}  (not installed, downloads on first use) - {describe_model(name)}"
            choices.append(ModelChoice(name, label, False, name.endswith("-obb.pt")))
    return choices


def resolve_weights_path(name: str) -> str:
    """Returns where a model's weights live (or should be downloaded to).

    A bare name such as 'yolo26m.pt' goes in the models folder, so Ultralytics downloads it there on first use.
    Existing files, and names that already include a folder, are used as given.
    """
    path = Path(name)
    if path.exists() or len(path.parts) > 1:
        return str(path)
    MODELS_DIR.mkdir(exist_ok=True)
    return str(MODELS_DIR / path.name)

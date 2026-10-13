"""Folder and file locations used across the project."""

from pathlib import Path

# This file lives in <project>/src/, so the project folder is one level up.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

VIDEOS_DIR = PROJECT_ROOT / "videos"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
SETTINGS_FILE = PROJECT_ROOT / "last_settings.json"

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}

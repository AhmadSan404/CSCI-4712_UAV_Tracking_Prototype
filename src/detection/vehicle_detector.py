"""Runs the main (COCO) model and the optional aerial (OBB) model on every frame and merges their detections."""

import time

import numpy as np
import torch

from src.detection.duplicate_merging import merge_duplicates
from src.detection.vehicle_types import VehicleClass
from src.detection.vram_monitor import VramMonitor
from src.detection.yolo_model import YoloModel

WARMUP_PASSES = 2


class VehicleDetector:
    """Detects vehicles with one or two YOLO models.

    Two models are better than one on drone footage: the COCO model handles low and oblique views, while the aerial
    OBB model (trained on DOTA) recognises cars seen from straight above.

    Attributes:
        models: The main model, followed by the aerial model if one is used.
        min_confidence: Confidence floor for both models (the tracker applies its own stricter thresholds).
        last_detect_ms: Time taken by the most recent detection, in milliseconds.
        gpu_name: Short GPU name, or 'CPU'.
        precision: 'fp16' on CUDA, otherwise 'fp32'.
    """

    def __init__(
        self,
        main_weights: str | None,
        main_image_size: int,
        aerial_weights: str | None,
        aerial_image_size: int,
        min_confidence: float,
        device_choice: str,
        vehicle_classes: list[VehicleClass],
    ) -> None:
        """Loads the models.

        Args:
            main_weights: Weights for the COCO model, or None to use only the aerial model.
            aerial_weights: Weights for the top-down model, or None to use only the main model.
            device_choice: 'auto', 'cpu' or 'gpu'.

        Raises:
            RuntimeError: If the GPU is requested but CUDA is not available.
            ValueError: If a model has none of the chosen classes.
        """
        has_cuda = torch.cuda.is_available()
        if device_choice == "gpu" and not has_cuda:
            raise RuntimeError("GPU was selected, but CUDA is not available on this machine.")
        is_cuda = has_cuda and device_choice != "cpu"

        self.device = "cuda:0" if is_cuda else "cpu"
        self.precision = "fp16" if is_cuda else "fp32"
        self.gpu_name = torch.cuda.get_device_name(0).replace("NVIDIA GeForce ", "") if is_cuda else "CPU"
        if is_cuda:
            torch.backends.cudnn.benchmark = True  # the input size is fixed for the whole run, so pick kernels once

        self.min_confidence = min_confidence
        self.models = [
            YoloModel(weights, image_size, self.device, vehicle_classes)
            for weights, image_size in ((main_weights, main_image_size), (aerial_weights, aerial_image_size))
            if weights
        ]
        self.last_detect_ms = 0.0
        self._vram = VramMonitor(self.device)

    @property
    def description(self) -> str:
        """Model names with their input sizes, e.g. 'yolo26m@960 + yolo26m-obb@1280'."""
        return " + ".join(f"{model.name}@{model.image_size}" for model in self.models)

    @property
    def missing_classes(self) -> list[VehicleClass]:
        """Chosen classes that at least one model does not have."""
        missing = []
        for model in self.models:
            missing += [vehicle_class for vehicle_class in model.missing_classes if vehicle_class not in missing]
        return missing

    @property
    def hardware_label(self) -> str:
        """GPU name, precision and VRAM use, e.g. 'RTX 3050 Laptop GPU fp16  VRAM 1.9/4.0 GB', or 'CPU fp32'."""
        if self._vram.used_mb is None or self._vram.total_mb is None:
            return f"{self.gpu_name} {self.precision}"
        used_gb, total_gb = self._vram.used_mb / 1024, self._vram.total_mb / 1024
        return f"{self.gpu_name} {self.precision}  VRAM {used_gb:.1f}/{total_gb:.1f} GB"

    @property
    def vram_used_mb(self) -> float | None:
        """GPU memory in use in MB, or None on the CPU."""
        return self._vram.used_mb

    @property
    def vram_total_mb(self) -> float | None:
        """Total GPU memory in MB, or None on the CPU."""
        return self._vram.total_mb

    def warm_up(self, frame_width: int, frame_height: int) -> None:
        """Runs blank frames through the models, so the first real frames are not slowed by one-off setup."""
        blank_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        for _ in range(WARMUP_PASSES):
            self.detect(blank_frame)

    def detect(self, frame: np.ndarray) -> np.ndarray:
        """Detects vehicles with every model and merges the results into one (N, 6) array."""
        started = time.perf_counter()
        per_model = [model.detect(frame, self.min_confidence) for model in self.models]
        merged = merge_duplicates(np.concatenate(per_model))
        self.last_detect_ms = (time.perf_counter() - started) * 1000
        return merged

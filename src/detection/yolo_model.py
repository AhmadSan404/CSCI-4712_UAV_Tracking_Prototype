"""One YOLO model (standard or OBB) that reports only vehicles."""

from pathlib import Path

import numpy as np
from ultralytics import YOLO

from src.detection.model_catalog import resolve_weights_path
from src.detection.vehicle_types import (
    MODEL_CLASS_NAME_TO_VEHICLE_CLASS,
    VEHICLE_CLASSES,
    VehicleClass,
    empty_detections,
)


class YoloModel:
    """Loads one weights file and turns its predictions into (N, 6) detection arrays.

    Attributes:
        name: Weights file name without its extension.
        task: 'detect' for standard models, 'obb' for aerial oriented-box models.
        image_size: Input size in pixels used for inference.
        missing_classes: Requested vehicle classes this model does not have (they are ignored).
    """

    def __init__(
        self, weights: str, image_size: int, device: str, vehicle_classes: list[VehicleClass]
    ) -> None:
        """Loads the weights and works out which of the model's classes to report.

        Raises:
            ValueError: If the model has none of the requested vehicle classes.
        """
        self.name = Path(weights).stem
        self.image_size = image_size
        self._device = device
        self._quantize = 16 if device.startswith("cuda") else None  # half precision on the GPU

        self._model = YOLO(resolve_weights_path(weights))
        self.task: str = self._model.task

        model_class_ids = {}  # the model's own class id -> index into VEHICLE_CLASSES
        for class_id, class_name in self._model.names.items():
            vehicle_class = MODEL_CLASS_NAME_TO_VEHICLE_CLASS.get(class_name.lower())
            if vehicle_class in vehicle_classes:
                model_class_ids[class_id] = VEHICLE_CLASSES.index(vehicle_class)
        if not model_class_ids:
            raise ValueError(f"{self.name} has none of the chosen classes.")

        available = {VEHICLE_CLASSES[index] for index in model_class_ids.values()}
        self.missing_classes = [vehicle_class for vehicle_class in vehicle_classes if vehicle_class not in available]
        self._class_ids = sorted(model_class_ids)
        self._class_index_lookup = np.full(max(self._model.names) + 1, -1, dtype=np.float32)
        for class_id, class_index in model_class_ids.items():
            self._class_index_lookup[class_id] = class_index

    def detect(self, frame: np.ndarray, min_confidence: float) -> np.ndarray:
        """Detects vehicles in a BGR frame. Returns an (N, 6) array, see `vehicle_types`."""
        result = self._model.predict(
            frame,
            imgsz=self.image_size,
            conf=min_confidence,
            classes=self._class_ids,
            device=self._device,
            quantize=self._quantize,
            verbose=False,
        )[0]

        # OBB models report rotated boxes. Their `xyxy` is the axis-aligned box that encloses the rotated one.
        boxes = (result.obb if result.obb is not None else result.boxes).cpu().numpy()
        if len(boxes) == 0:
            return empty_detections()

        frame_height, frame_width = frame.shape[:2]
        corners = np.clip(boxes.xyxy, 0, [frame_width, frame_height, frame_width, frame_height])
        class_indices = self._class_index_lookup[boxes.cls.astype(int)]
        return np.column_stack([corners, boxes.conf, class_indices]).astype(np.float32)

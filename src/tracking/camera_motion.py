"""Camera motion estimation and hard-cut detection for moving-camera (drone) footage.

A drone that follows a car moves constantly, so the whole image shifts, rotates and zooms from frame to frame. This
module measures that motion from the background: corner features (vehicles masked out) are tracked with
Lucas-Kanade optical flow on a downscaled frame, and a similarity transform (shift, rotation, zoom) is fitted with
RANSAC. The tracker uses the resulting warp to predict where cars should be, and trails use it to stay on the ground.

The same measurements spot hard cuts: the color histogram jumps, or the motion-compensated previous frame no longer
looks like the current one.
"""
from dataclasses import dataclass

import cv2
import numpy as np

WORK_WIDTH_PX = 640  # frames are downscaled to this width before analysis
MAX_FEATURES = 400
FEATURE_QUALITY_LEVEL = 0.01
FEATURE_MIN_DISTANCE_PX = 8
FEATURE_BLOCK_SIZE = 7
FLOW_WINDOW_PX = (21, 21)
FLOW_PYRAMID_LEVELS = 3
FLOW_CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03)
MIN_POINTS_FOR_ESTIMATE = 8  # fewer tracked points than this and we do not trust a fit
RANSAC_PIXEL_THRESHOLD = 1.5  # a point is an inlier if it lands within this many pixels of the fitted motion
RESIDUAL_BORDER_PX = 8  # ignore this border when comparing frames: the warp pulls in reflected pixels there
VEHICLE_MASK_PADDING = 0.10  # grow each vehicle box by 10% when masking, because detected boxes are not exact

HISTOGRAM_BINS = [12, 8, 8]
HISTOGRAM_RANGES = [0, 180, 0, 256, 0, 256]
CUT_MIN_HISTOGRAM_SIMILARITY = 0.5  # below this histogram correlation, the colors changed abruptly
CUT_MIN_RESIDUAL = 20.0  # above this, together with few inliers, the motion model failed
CUT_MAX_INLIER_RATIO = 0.2


def identity_warp() -> np.ndarray:
    """The 2x3 transform that moves nothing."""
    return np.eye(2, 3, dtype=np.float64)

@dataclass
class MotionResult:
    """Camera motion for one frame.

    Attributes:
        warp: 2x3 transform mapping previous-frame pixels to current-frame pixels (full resolution). It is the
            identity on the first frame and on cuts.
        is_cut: True when this frame starts a new shot.
    """
    warp:np.ndarray
    is_cut:bool

class CameraMotion:
    """Estimates how the camera moved between consecutive frames."""
    def __init__(self) -> None:
        self._feature_params = dict(
            maxCorners=MAX_FEATURES,
            qualityLevel=FEATURE_QUALITY_LEVEL,
            minDistance=FEATURE_MIN_DISTANCE_PX,
            blockSize=FEATURE_BLOCK_SIZE,
        )
        self.reset()

    def reset(self) -> None:
        """Forgets the previous frame, so the next `update` returns the identity warp (e.g. after a restart)."""
        self._previous_gray: np.ndarray | None = None
        self._previous_points: np.ndarray | None = None
        self._previous_histogram: np.ndarray | None = None

    def update(self, frame:np.ndarray, vehicle_boxes:np.ndarray | None=None) -> MotionResult:
        """Estimates the motion since the previous frame.

        Args:
            frame: The current BGR frame at full resolution.
            vehicle_boxes: Detections (N, 4+) whose areas are left out of the background features.
        """
        frame_height, frame_width = frame.shape[:2]
        scale = min(1.0, WORK_WIDTH_PX / frame_width)
        small_frame = cv2.resize(frame, (round(frame_width * scale), round(frame_height * scale)), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        histogram = self._compute_histogram(small_frame)

        result = MotionResult(identity_warp(), is_cut=False)
        if self._previous_gray is not None:
            result = self._compare_with_previous_frame(gray, histogram, scale)

        # Remember this frame, and pick the background features that the next call will track.
        self._previous_gray = gray
        self._previous_histogram = histogram
        mask = self._build_background_mask(gray.shape, vehicle_boxes, scale)
        self._previous_points = cv2.goodFeaturesToTrack(gray, mask=mask, **self._feature_params)
        return result

    def _compare_with_previous_frame(self, gray:np.ndarray, histogram:np.ndarray, scale:float) -> MotionResult:
        """Measures the motion and decides whether this frame is a cut."""
        histogram_similarity = float(cv2.compareHist(self._previous_histogram, histogram, cv2.HISTCMP_CORREL))
        small_warp, inlier_ratio = self._fit_warp(gray)

        if small_warp is not None:
            residual = self._measure_residual(small_warp, gray)
            warp = small_warp.copy()
            warp[:, 2] /= scale  # rotation and zoom do not depend on resolution, but the shift does
        else:
            residual = float(cv2.absdiff(self._previous_gray, gray).mean())
            warp = identity_warp()

        colors_changed = histogram_similarity < CUT_MIN_HISTOGRAM_SIMILARITY
        motion_model_failed = residual > CUT_MIN_RESIDUAL and inlier_ratio < CUT_MAX_INLIER_RATIO
        is_cut = colors_changed or motion_model_failed
        return MotionResult(identity_warp() if is_cut else warp, is_cut)

    def _fit_warp(self, gray:np.ndarray) -> tuple[np.ndarray | None, float]:
        """Tracks the features into this frame and fits a similarity transform with RANSAC.

        Returns:
            The 2x3 transform in working-resolution pixels (None if no fit was possible), and the share of the
            features that RANSAC kept as inliers.
        """
        points = self._previous_points
        if points is None or len(points) < MIN_POINTS_FOR_ESTIMATE:
            return None, 0.0

        moved_points, status, _ = cv2.calcOpticalFlowPyrLK(
            self._previous_gray,
            gray,
            points,
            None,
            winSize=FLOW_WINDOW_PX,
            maxLevel=FLOW_PYRAMID_LEVELS,
            criteria=FLOW_CRITERIA,
        )
        was_tracked = status.ravel() == 1
        if was_tracked.sum() < MIN_POINTS_FOR_ESTIMATE:
            return None, 0.0

        warp, inlier_mask = cv2.estimateAffinePartial2D(
            points[was_tracked],
            moved_points[was_tracked],
            method=cv2.RANSAC,
            ransacReprojThreshold=RANSAC_PIXEL_THRESHOLD,
            maxIters=2000,
            confidence=0.995,
        )
        if warp is None:
            return None, 0.0
        # The ratio is relative to all features we tried to track, not just the ones optical flow followed.
        return warp, float(inlier_mask.sum()) / len(points)

    def _measure_residual(self, small_warp:np.ndarray, gray:np.ndarray) -> float:
        """Mean pixel difference (0-255) between the motion-compensated previous frame and this one."""
        height, width = gray.shape
        compensated = cv2.warpAffine(self._previous_gray, small_warp, (width, height), borderMode=cv2.BORDER_REFLECT)
        margin = RESIDUAL_BORDER_PX
        difference = cv2.absdiff(compensated, gray)
        return float(difference[margin:-margin, margin:-margin].mean())

    @staticmethod
    def _build_background_mask(shape:tuple[int, int], vehicle_boxes:np.ndarray | None, scale:float) -> np.ndarray:
        """A mask that is 255 on background and 0 on (padded) vehicle boxes."""
        mask = np.full(shape, 255, dtype=np.uint8)
        if vehicle_boxes is not None:
            for x1, y1, x2, y2 in np.asarray(vehicle_boxes)[:, :4] * scale:
                pad_x = VEHICLE_MASK_PADDING * (x2 - x1)
                pad_y = VEHICLE_MASK_PADDING * (y2 - y1)
                mask[max(0, int(y1 - pad_y)):int(y2 + pad_y) + 1, max(0, int(x1 - pad_x)):int(x2 + pad_x) + 1] = 0
        return mask

    @staticmethod
    def _compute_histogram(small_frame:np.ndarray) -> np.ndarray:
        """A normalized HSV color histogram, used to spot hard cuts."""
        hsv = cv2.cvtColor(small_frame, cv2.COLOR_BGR2HSV)
        histogram = cv2.calcHist([hsv], [0, 1, 2], None, HISTOGRAM_BINS, HISTOGRAM_RANGES)
        return cv2.normalize(histogram, histogram).flatten()

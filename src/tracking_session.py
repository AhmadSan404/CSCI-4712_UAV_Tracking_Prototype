"""The main per-frame loop.

For every video frame:
1. detect vehicles          (detection/vehicle_detector.py)
2. measure camera motion    (tracking/camera_motion.py), which also spots hard cuts
3. track the vehicles       (tracking/vehicle_tracker.py), and update trails
4. choose the target        (tracking/target_selector.py)
5. draw the result and handle keys and clicks (display/)
"""
import dataclasses
import time
from pathlib import Path

import numpy as np

from src.detection.vehicle_detector import VehicleDetector
from src.detection.vehicle_types import CLASS_INDEX, CONFIDENCE, PERSON_CLASS_INDEX, VEHICLE_CLASSES
from src.display.display_window import DisplayWindow
from src.display.fps_counter import FpsCounter
from src.display.hud import Hud, HudState
from src.display.overlay import Overlay
from src.menu.tracker_settings import SOURCE_STREAM, TrackerSettings
from src.playback_pacer import PlaybackPacer
from src.run_logging.run_logger import RunLogger
from src.run_logging.run_statistics import TARGET_STATE_COASTING, TARGET_STATE_TRACKING
from src.tracking.camera_motion import CameraMotion
from src.tracking.target_selector import TargetSelector
from src.tracking.track import Track
from src.tracking.track_picking import find_next_track, find_track_at_point
from src.tracking.trails import Trails
from src.tracking.vehicle_tracker import VehicleTracker
from src.video_reader import VideoReader

IDLE_SLEEP_SECONDS = 0.01  # while paused or ended, sleep this long per loop so we do not spin a CPU core
KEY_TAB = 9
KEY_SPACE = 32
HELP_TEXT = (
    "SPACE pause   F speed   R restart   T trails   I info   H help   Q menu   |   "
    "click or TAB: lock a car   C / right-click: automatic target"
)
LIVE_HELP_TEXT = (
    "SPACE pause   T trails   I info   H help   Q menu   |   click or TAB: lock a car   C / right-click: automatic target"
)

END_REASON_VIDEO_ENDED = "video ended"  # also used when a live stream stops
END_REASON_USER_QUIT = "user quit"
END_REASON_ERROR = "error"


class TrackingSession:
    """One run over one video or live stream. Create it, then call `run`; live key changes are written into `settings`."""

    def __init__(self, settings:TrackerSettings, detector:VehicleDetector, screen_size:tuple[int, int], model_info:dict) -> None:
        self.settings = settings
        self.detector = detector
        self.model_info = model_info
        if settings.source_type == SOURCE_STREAM:
            self.reader = VideoReader(settings.camera_name, is_live=True)
            self.source_name = settings.camera_name
        else:
            self.reader = VideoReader(settings.video_path)
            self.source_name = Path(settings.video_path).name
        try:
            print(f"Warming up {detector.description} on {detector.gpu_name} ({detector.precision})...")
            detector.warm_up(self.reader.width, self.reader.height)
            title = f"Drone Car Tracker - {self.source_name}"
            self.window = DisplayWindow(self.reader.width, self.reader.height, *screen_size, title)
        except Exception:
            self.reader.close()
            raise

        self.camera_motion = CameraMotion()
        self.tracker = VehicleTracker(self.reader.fps)
        self.trails = Trails(self.reader.fps)
        self.selector = TargetSelector((PERSON_CLASS_INDEX,))  # people can be locked by click, never auto-picked
        self.overlay = Overlay(settings.show_trails)
        self.hud = Hud()
        self.fps_counter = FpsCounter()
        self.pacer = PlaybackPacer(self.reader.fps)
        self.logger = RunLogger()

        self.is_real_time = not settings.should_start_at_max_speed
        self.is_paused = False
        self.has_ended = False
        self.frame_index = 0
        self.frame_shape: tuple[int, ...] | None = None
        self.display_base: np.ndarray | None = None  # the current frame resized to display size
        self.tracks: list[Track] = []
        self.target_id: int | None = None
        self.detect_ms = 0.0
        self.track_ms = 0.0

    def run(self) -> None:
        """Plays the video until the user quits or the window closes. The run log is always written."""
        self.logger.start_run(*self._describe_run())
        error_message = None
        try:
            self._main_loop()
        except Exception as error:
            error_message = f"{type(error).__name__}: {error}"
            raise
        finally:
            self.reader.close()
            self.window.close()
            if error_message:
                end_reason = END_REASON_ERROR
            else:
                end_reason = END_REASON_VIDEO_ENDED if self.has_ended else END_REASON_USER_QUIT
            self.logger.finish_run(end_reason, error_message)

    def _describe_run(self) -> tuple[dict, dict, dict, dict]:
        """The plain-data descriptions of the video, models, settings and hardware for the logger."""
        video = {
            "path": "" if self.reader.is_live else self.settings.video_path,
            "name": self.source_name,
            "is_live": self.reader.is_live,
            "width": self.reader.width,
            "height": self.reader.height,
            "fps": self.reader.fps,
            "frame_count": self.reader.frame_count,
        }
        models = {**self.model_info, "description": self.detector.description}
        hardware = {
            "device_name": self.detector.gpu_name,
            "precision": self.detector.precision,
            "total_vram_mb": self.detector.vram_total_mb,
        }
        return video, models, dataclasses.asdict(self.settings), hardware

    def _main_loop(self) -> None:
        """Per iteration: get a frame (unless paused), draw it, pace to the video's speed, show it, read keys."""
        while True:
            iteration_start = time.perf_counter()
            has_new_frame = False
            if not self.is_paused and not self.has_ended:
                has_new_frame = self._read_and_process_frame()

            self._handle_clicks()
            self._update_target()
            image = self._render()
            work_seconds = time.perf_counter() - iteration_start

            if not has_new_frame:
                time.sleep(IDLE_SLEEP_SECONDS)
            elif self.is_real_time and not self.reader.is_live:  # a live source already arrives in real time
                self.pacer.sleep_until_next_frame_is_due()
            key = self.window.show(image)
            if has_new_frame:
                self._record_shown_frame(work_seconds)

            if not self.window.is_open() or not self._handle_key(key):
                return

    # ------------------------------------------------------------------ processing one frame

    def _read_and_process_frame(self) -> bool:
        """Reads the next frame and processes it. Returns False when the video has just ended."""
        item = self.reader.read()
        if item is None:
            if self.display_base is None:
                raise RuntimeError(f"Could not read any frames from {self.source_name}.")
            self.has_ended = True
            self.logger.record_playback_active(False)
            return False
        self._process_frame(*item)
        return True

    def _process_frame(self, frame_index:int, frame:np.ndarray) -> None:
        """Detects, measures camera motion, tracks, and updates trails for one new frame."""
        self.frame_index = frame_index
        self.frame_shape = frame.shape

        detect_started = time.perf_counter()
        detections = self.detector.detect(frame)
        detect_finished = time.perf_counter()

        motion = self.camera_motion.update(frame, detections)
        if motion.is_cut:
            self._handle_scene_cut()
        self.tracks = self.tracker.update(detections, frame.shape, motion.warp)
        self.trails.update(self.tracks, motion.warp)

        self.detect_ms = (detect_finished - detect_started) * 1000
        self.track_ms = (time.perf_counter() - detect_finished) * 1000
        self.display_base = self.window.resize_frame(frame)
        found = [(VEHICLE_CLASSES[int(row[CLASS_INDEX])].display_name, float(row[CONFIDENCE])) for row in detections]
        self.logger.record_detection_pass(self.detect_ms, found, self.detector.vram_used_mb)

    def _handle_scene_cut(self) -> None:
        """A new shot began: positions from the old shot mean nothing now."""
        self.logger.record_target_event("scene_cut", self._video_seconds())
        self.tracker.clear()
        self.trails.clear()

    def _update_target(self) -> None:
        """Lets the selector confirm or choose the target, and logs when the target changes."""
        if self.frame_shape is None:
            return
        previous_id = self.target_id
        self.target_id = self.selector.update(self.tracks, self.frame_shape)
        if self.target_id != previous_id:
            mode = "locked" if self.selector.is_locked else "automatic"
            self.logger.record_target_event("target_changed", self._video_seconds(), f"#{self.target_id} ({mode})")

    # ------------------------------------------------------------------ pacing and logging

    def _record_shown_frame(self, work_seconds:float) -> None:
        """Updates the FPS counter and logs the frame; the first frame after a reset has no interval."""
        interval = self.pacer.mark_frame_shown()
        self.fps_counter.tick()

        target = next((track for track in self.tracks if track.track_id == self.target_id), None)
        target_state = None
        if target is not None:
            target_state = TARGET_STATE_COASTING if target.is_coasting else TARGET_STATE_TRACKING
        track_ids = [track.track_id for track in self.tracks]
        self.logger.record_frame(interval, work_seconds, track_ids, target_state, self.selector.is_locked)

    def _reset_timing(self) -> None:
        """Forgets pacing and FPS history, e.g. after pausing, restarting or switching playback mode."""
        self.pacer.reset()
        self.fps_counter.reset()

    def _video_seconds(self) -> float:
        """Current position in the video, in seconds."""
        return self.frame_index / self.reader.fps

    # ------------------------------------------------------------------ drawing

    def _render(self) -> np.ndarray:
        """Draws the tracks and the HUD on a copy of the current display-size frame."""
        image = self.display_base.copy()
        self.overlay.draw_tracks(
            image, self.tracks, self.trails, self.target_id, self.selector.is_locked, self.window.scale
        )
        self.hud.draw(image, self._build_hud_state())
        return image

    def _build_hud_state(self) -> HudState:
        """Collects the values the HUD shows."""
        if self.target_id is None:
            target_text = "none"
        else:
            mode = "locked" if self.selector.is_locked else "auto"
            target = next((track for track in self.tracks if track.track_id == self.target_id), None)
            searching = " (searching)" if target is not None and target.is_coasting else ""
            target_text = f"#{self.target_id} {mode}{searching}"

        return HudState(
            fps=self.fps_counter.value,
            video_fps=self.reader.fps,
            is_paused=self.is_paused and not self.has_ended,
            has_ended=self.has_ended,
            models_text=self.detector.description,
            hardware_text=self.detector.hardware_label,
            detect_ms=self.detect_ms,
            track_ms=self.track_ms,
            num_tracks=sum(not track.is_coasting for track in self.tracks),
            target_text=target_text,
            video_position_seconds=self._video_seconds(),
            video_duration_seconds=self.reader.duration_seconds,
            is_real_time=self.is_real_time,
            help_text=LIVE_HELP_TEXT if self.reader.is_live else HELP_TEXT,
            is_live=self.reader.is_live,
        )

    # ------------------------------------------------------------------ user input

    def _handle_clicks(self) -> None:
        """Left-click locks the car under the cursor; right-click returns to the automatic target."""
        for click in self.window.take_clicks():
            if click.button == "right":
                self.selector.release()
                continue
            track = find_track_at_point(self.tracks, click.x, click.y)
            if track is not None:
                self.selector.lock(track.track_id)

    def _handle_key(self, key:int) -> bool:
        """Performs the action bound to a key. Returns False if the user asked to go back to the menu."""
        if key == ord("q"):
            return False
        if key == KEY_SPACE and not self.has_ended:
            self.is_paused = not self.is_paused
            self.logger.record_playback_active(not self.is_paused)
            self._reset_timing()
        elif key == ord("f") and not self.reader.is_live:
            self.is_real_time = not self.is_real_time
            self.settings.should_start_at_max_speed = not self.is_real_time
            self.logger.record_setting_change("playback_mode", "real-time" if self.is_real_time else "max speed")
            self._reset_timing()
        elif key == ord("r") and not self.reader.is_live:
            self._restart_video()
        elif key == ord("t"):
            self.overlay.show_trails = not self.overlay.show_trails
            self.settings.show_trails = self.overlay.show_trails
            self.logger.record_setting_change("show_trails", self.overlay.show_trails)
        elif key == ord("i"):
            self.hud.show_info_panel = not self.hud.show_info_panel
        elif key == ord("h"):
            self.hud.show_help_bar = not self.hud.show_help_bar
        elif key == ord("c"):
            self.selector.release()
        elif key == KEY_TAB:
            next_track = find_next_track(self.tracks, self.target_id)
            if next_track is not None:
                self.selector.lock(next_track.track_id)
        return True

    def _restart_video(self) -> None:
        """Jumps back to the start and clears everything that belongs to the old position."""
        self.reader.restart()
        self.camera_motion.reset()
        self.tracker.clear()
        self.trails.clear()
        self.selector.release()
        self.tracks = []
        self.frame_index = 0
        self.is_paused = False
        self.has_ended = False
        self.logger.record_playback_active(True)
        self._reset_timing()


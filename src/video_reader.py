"""Background-thread video decoding, so reading frames overlaps with running the model."""

import queue
import threading
import time

import cv2
import numpy as np

from src.camera_scanner import CAMERA_BACKEND, find_camera_index

BUFFER_SIZE_FRAMES = 8
LIVE_BUFFER_SIZE_FRAMES = 1  # a live source keeps only the newest frame, so the picture never falls behind
REQUESTED_CAMERA_SIZE_PX = (1920, 1080)  # DirectShow often defaults to 640x480 unless a size is asked for
STREAM_LOST_SECONDS = 2.0  # a camera that gives no frames for this long counts as ended
DEFAULT_FPS = 30.0  # used when the file reports a missing or implausible frame rate
PLAUSIBLE_FPS_RANGE = (1, 240)
QUEUE_TIMEOUT_SECONDS = 0.1


class VideoReader:
    """Decodes a video file (or reads a live camera) on a worker thread and hands out frames through a small queue.

    Attributes:
        is_live: True for a camera. Live frames arrive in real time, the length is unknown and there is no restart.
        fps: Frame rate of the video.
        width: Frame width in pixels.
        height: Frame height in pixels.
        frame_count: Total number of frames (0 if unknown).
    """

    def __init__(self, source: str, is_live: bool = False) -> None:
        """Opens the source and starts decoding.

        Args:
            source: A video file path, or a camera name when `is_live` is True.

        Raises:
            RuntimeError: If OpenCV cannot open the file or camera.
        """
        self.is_live = is_live
        if is_live:
            self._capture = self._open_camera(source)
        else:
            self._capture = cv2.VideoCapture(source)
            if not self._capture.isOpened():
                raise RuntimeError(f"Could not open video: {source}")

        reported_fps = self._capture.get(cv2.CAP_PROP_FPS)
        min_fps, max_fps = PLAUSIBLE_FPS_RANGE
        self.fps = reported_fps if min_fps <= reported_fps <= max_fps else DEFAULT_FPS
        self.width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_count = max(0, int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT)))

        if is_live:
            print(f"Camera '{source}' opened at {self.width}x{self.height}, {self.fps:.0f} fps.")

        self._frame_queue: queue.Queue = queue.Queue(maxsize=LIVE_BUFFER_SIZE_FRAMES if is_live else BUFFER_SIZE_FRAMES)
        self._start_decoder()

    @staticmethod
    def _open_camera(camera_name: str) -> cv2.VideoCapture:
        """Opens a camera by name, asking for a full-HD picture.

        Raises:
            RuntimeError: If the camera is not connected or cannot be opened.
        """
        index = find_camera_index(camera_name)
        if index is None:
            raise RuntimeError(f"Camera not found: {camera_name}. Is it connected (or is OBS Virtual Camera started)?")
        capture = cv2.VideoCapture(index, CAMERA_BACKEND)
        if not capture.isOpened():
            raise RuntimeError(
                f"Could not open camera: {camera_name}. For OBS, press 'Start Virtual Camera' in OBS first."
            )
        width, height = REQUESTED_CAMERA_SIZE_PX
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return capture

    @property
    def duration_seconds(self) -> float:
        """Length of the video in seconds, or 0.0 if the frame count is unknown."""
        return self.frame_count / self.fps if self.frame_count else 0.0

    def read(self) -> tuple[int, np.ndarray] | None:
        """Returns the next (frame index, BGR frame), or None when the video has ended."""
        while True:
            try:
                return self._frame_queue.get(timeout=QUEUE_TIMEOUT_SECONDS)
            except queue.Empty:
                if not self._decoder_thread.is_alive():
                    return None

    def restart(self) -> None:
        """Jumps back to the first frame and discards any queued frames. Does nothing for a live source."""
        if self.is_live:
            return
        self._stop_decoder()
        self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self._start_decoder()

    def close(self) -> None:
        """Stops decoding and releases the file or camera."""
        self._stop_decoder()
        self._capture.release()

    def _start_decoder(self) -> None:
        """Launches a daemon thread that decodes frames into the queue."""
        self._stop_event = threading.Event()
        self._decoder_thread = threading.Thread(
            target=self._live_loop if self.is_live else self._decode_loop, args=(self._stop_event,),
            name="video-decoder", daemon=True,
        )
        self._decoder_thread.start()

    def _stop_decoder(self) -> None:
        """Asks the decoder thread to stop, waits for it, and empties the queue."""
        self._stop_event.set()
        while self._decoder_thread.is_alive():
            self._empty_queue()  # frees space so a decoder blocked on a full queue notices the stop request
            self._decoder_thread.join(timeout=0.05)
        self._empty_queue()

    def _empty_queue(self) -> None:
        """Discards every frame currently waiting in the queue."""
        try:
            while True:
                self._frame_queue.get_nowait()
        except queue.Empty:
            pass

    def _decode_loop(self, stop_event: threading.Event) -> None:
        """Worker loop: reads frames and queues them until the video ends or a stop is requested."""
        frame_index = int(self._capture.get(cv2.CAP_PROP_POS_FRAMES))
        while not stop_event.is_set():
            is_success, frame = self._capture.read()
            item = (frame_index, frame) if is_success else None  # None marks the end of the video
            self._put_unless_stopped(item, stop_event)
            if not is_success:
                return
            frame_index += 1

    def _put_unless_stopped(self, item: tuple[int, np.ndarray] | None, stop_event: threading.Event) -> None:
        """Queues an item, waiting while the queue is full, but giving up if a stop is requested."""
        while not stop_event.is_set():
            try:
                self._frame_queue.put(item, timeout=QUEUE_TIMEOUT_SECONDS)
                return
            except queue.Full:
                pass

    def _live_loop(self, stop_event: threading.Event) -> None:
        """Worker loop for a camera: keeps only the newest frame, and ends when no frame arrives for a while."""
        frame_index = 0
        last_frame_time = time.perf_counter()
        while not stop_event.is_set():
            is_success, frame = self._capture.read()
            if is_success:
                self._replace_queued_frame((frame_index, frame))
                frame_index += 1
                last_frame_time = time.perf_counter()
            elif time.perf_counter() - last_frame_time > STREAM_LOST_SECONDS:
                self._replace_queued_frame(None)  # None marks the end of the stream
                return
            else:
                time.sleep(QUEUE_TIMEOUT_SECONDS)  # a failed read can return at once, so do not spin

    def _replace_queued_frame(self, item: tuple[int, np.ndarray] | None) -> None:
        """Drops the frame still waiting in the live queue (if any) and queues this one instead."""
        self._empty_queue()
        self._frame_queue.put_nowait(item)

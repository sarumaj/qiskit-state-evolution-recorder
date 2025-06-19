from typing import Generator, Union, Optional
from numpy import ndarray, uint8
from tqdm import tqdm
import logging
import time
import os
from tempfile import gettempdir
from multiprocessing import Lock

from .backend import AnimationBackend
from ..frame_renderer import FrameRenderer

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

logger = logging.getLogger("qiskit_state_evolution_recorder.backend.opencv")


class OpenCVBackend(AnimationBackend):
    """
    OpenCV-based animation backend.
    """

    def __init__(self, frame_renderer: FrameRenderer):
        """
        Initialize the OpenCV backend.

        Parameters:
        -----------
        frame_renderer: FrameRenderer
            The frame renderer instance
        """
        self._frame_renderer = frame_renderer
        self._lock = Lock()
        self._writer = None

    def is_available(self) -> bool:
        """Check if OpenCV is available."""
        return CV2_AVAILABLE

    @classmethod
    def get_install_instructions(cls) -> str:
        """Get OpenCV installation instructions."""
        return "pip install opencv-python"

    def record(
        self,
        filename: str,
        frames: Generator[Union[str, ndarray[uint8]], None, None],
        total_frames: int,
        *,
        fps: int = 60,
        interval: int = 200,
        disk: bool = False,
        pbar: Optional[tqdm] = None
    ):
        """Record using OpenCV backend."""
        if not CV2_AVAILABLE:
            logger.error(
                "OpenCV is not installed. Please install opencv-python: %s",
                self.get_install_instructions()
            )
            return

        if fps <= 0:
            raise ValueError("fps must be positive")
        if interval <= 0:
            raise ValueError("interval must be positive")

        try:
            start_time = time.time()

            # Get first frame to determine dimensions
            first_frame = next(frames)
            if isinstance(first_frame, str):
                # Load image from file
                frame_array = cv2.imread(first_frame)
                if frame_array is None:
                    raise RuntimeError(f"Could not load frame from {first_frame}")
            else:
                # Convert from RGB to BGR (OpenCV format)
                frame_array = cv2.cvtColor(first_frame, cv2.COLOR_RGB2BGR)

            height, width = frame_array.shape[:2]

            # Initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            with self._lock:
                self._writer = cv2.VideoWriter(filename, fourcc, fps, (width, height))

            if not self._writer.isOpened():
                raise RuntimeError("Could not open video writer")

            # Write first frame
            self._writer.write(frame_array)
            if pbar:
                pbar.n = 1
                pbar.refresh()

            # Write remaining frames
            for frame_idx, frame in enumerate(frames, 1):
                if isinstance(frame, str):
                    # Load image from file
                    frame_array = cv2.imread(frame)
                    if frame_array is None:
                        logger.warning(f"Could not load frame from {frame}, skipping")
                        continue
                else:
                    # Convert from RGB to BGR (OpenCV format)
                    frame_array = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                self._writer.write(frame_array)

                if pbar:
                    pbar.n = min(frame_idx + 1, total_frames)
                    pbar.refresh()

        except KeyboardInterrupt:
            # Clean up if interrupted
            try:
                with self._lock:
                    if os.path.exists(filename):
                        os.remove(filename)
            except Exception as e:
                logger.warning(f"Error cleaning up recording: {str(e)}")
            raise
        else:
            logger.info(f"\nRecording finished. Elapsed time: {time.time() - start_time:.2f} seconds")
        finally:
            # Clean up resources
            try:
                with self._lock:
                    if self._writer:
                        self._writer.release()
                        self._writer = None
                    if disk:
                        self._cleanup_temp_files(total_frames)
            except Exception as e:
                logger.error(f"Error cleaning up resources: {str(e)}")

    def _cleanup_temp_files(self, total_frames: int):
        """Clean up temporary frame files."""
        for index in range(total_frames):
            filename = os.path.join(gettempdir(), f"{index}.png")
            try:
                if os.path.exists(filename):
                    os.remove(filename)
            except Exception as e:
                logger.error(f"Error cleaning up file {filename}: {str(e)}")

    def cleanup(self):
        """Clean up resources used by the OpenCV backend."""
        try:
            with self._lock:
                if self._writer:
                    self._writer.release()
                    self._writer = None
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

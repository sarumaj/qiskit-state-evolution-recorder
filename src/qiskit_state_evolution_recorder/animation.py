from typing import Generator, Union, Optional
from numpy import ndarray, uint8
import os
from multiprocessing import Lock
from tqdm import tqdm
import logging

from .frame_renderer import FrameRenderer
from .backend import get_best_backend

logger = logging.getLogger("qiskit_state_evolution_recorder.animation")


class AnimationRecorder:
    """
    A class to handle animation recording and video creation.

    This class is responsible for:
    1. Creating animations from frames
    2. Saving animations to video files
    3. Managing progress tracking
    4. Cleaning up temporary files
    """

    def __init__(
        self,
        frame_renderer: FrameRenderer,
    ):
        """
        Initialize the AnimationRecorder.

        Parameters:
        -----------
        frame_renderer: FrameRenderer
            The frame renderer instance that handles frame generation
        """
        self._frame_renderer = frame_renderer
        self._lock = Lock()
        self._backend = None

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
        """
        Record the frames into a video file.

        This method:
        1. Creates an animation from the provided frames
        2. Saves it to a video file
        3. Handles cleanup of temporary files
        4. Tracks progress during encoding

        Parameters:
        -----------
        filename: str
            The name of the video file to create
        frames: Generator[Union[str, ndarray[uint8]], None, None]
            Generator yielding frames
        total_frames: int
            Total number of frames to record
        fps: int
            The number of frames per second
        interval: int
            The interval between each frame in milliseconds
        disk: bool
            Whether to save the frames to disk
        pbar: Optional[tqdm]
            Progress bar to update during encoding

        Raises:
        -------
        RuntimeError
            If no animation backends are available
        ValueError
            If fps or interval is invalid
        """
        if fps <= 0:
            raise ValueError("fps must be positive")
        if interval <= 0:
            raise ValueError("interval must be positive")

        try:
            # Get the best available backend
            self._backend = get_best_backend(self._frame_renderer)

            # Log which backend is being used
            backend_name = self._backend.__class__.__name__
            logger.info(f"Using {backend_name} for video recording")

            # Record using the selected backend
            self._backend.record(
                filename=filename,
                frames=frames,
                total_frames=total_frames,
                fps=fps,
                interval=interval,
                disk=disk,
                pbar=pbar
            )

        except KeyboardInterrupt:
            # Clean up if interrupted
            try:
                if os.path.exists(filename):
                    os.remove(filename)
            except Exception as e:
                logger.warning(f"Error cleaning up recording: {str(e)}")
            raise
        finally:
            # Clean up resources
            try:
                if self._backend:
                    self._backend.cleanup()
                self._frame_renderer.close()
            except Exception as e:
                logger.error(f"Error cleaning up resources: {str(e)}")

    def close(self):
        """
        Clean up resources used by the animation recorder.

        This method:
        1. Stops any running animation
        2. Cleans up temporary files
        3. Closes the frame renderer
        """
        try:
            if self._backend:
                self._backend.cleanup()
                self._backend = None
            self._frame_renderer.close()
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

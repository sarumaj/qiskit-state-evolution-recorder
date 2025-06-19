from abc import ABC, abstractmethod
from typing import Generator, Optional, Union

from numpy import ndarray, uint8
from tqdm import tqdm


class AnimationBackend(ABC):
    """
    Abstract base class for animation backends.

    This class defines the interface that all animation backends must implement.
    """

    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if this backend is available on the system.

        Returns:
        --------
        bool
            True if the backend is available, False otherwise
        """
        pass

    @classmethod
    @abstractmethod
    def get_install_instructions(cls) -> str:
        """
        Get installation instructions for this backend.

        Returns:
        --------
        str
            Installation instructions for the current operating system
        """
        pass

    @abstractmethod
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
        Record frames into a video file.

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
        """
        pass

    @abstractmethod
    def cleanup(self):
        """
        Clean up any resources used by this backend.
        """
        pass

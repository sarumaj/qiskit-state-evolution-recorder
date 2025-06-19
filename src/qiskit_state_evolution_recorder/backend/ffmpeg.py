from typing import Generator, Union, Optional
from numpy import ndarray, uint8
from tqdm import tqdm
import logging
import time
import os
from tempfile import gettempdir
from multiprocessing import Lock
import subprocess
import platform
import shutil
from .backend import AnimationBackend

from ..frame_renderer import FrameRenderer

logger = logging.getLogger("qiskit_state_evolution_recorder.backend.ffmpeg")


class FFmpegBackend(AnimationBackend):
    """
    FFmpeg-based animation backend using matplotlib's FuncAnimation.
    """

    def __init__(self, frame_renderer: FrameRenderer):
        """
        Initialize the FFmpeg backend.

        Parameters:
        -----------
        frame_renderer: FrameRenderer
            The frame renderer instance
        """
        self._frame_renderer = frame_renderer
        self._lock = Lock()
        self._anim = None

    def is_available(self) -> bool:
        """
        Check if ffmpeg is installed and available in the system.

        This function works across different operating systems by:
        1. Using shutil.which() to find ffmpeg in PATH
        2. Handling platform-specific command execution
        3. Providing appropriate error messages for each OS

        Returns:
        --------
        bool
            True if ffmpeg is installed, False otherwise
        """
        # Check if ffmpeg is in PATH
        if not shutil.which('ffmpeg'):
            return False

        # Execute ffmpeg to verify it works
        try:
            subprocess.run(
                ['ffmpeg', '-version'],
                capture_output=True,
                check=True,
                shell=platform.system() == 'Windows'  # On Windows, use shell=True to handle PATH properly
            )
            return True

        except (subprocess.SubprocessError, FileNotFoundError):
            return False

    @classmethod
    def get_install_instructions(cls) -> str:
        """
        Get platform-specific instructions for installing ffmpeg.

        Returns:
        --------
        str
            Installation instructions for the current operating system
        """
        match platform.system().lower():
            case 'linux':
                # Try to detect the distribution
                try:
                    with open('/etc/os-release', 'r') as f:
                        content = f.read().lower()
                        if 'ubuntu' in content or 'debian' in content:
                            return "sudo apt-get install ffmpeg"
                        elif 'fedora' in content:
                            return "sudo dnf install ffmpeg"
                        elif 'arch' in content:
                            return "sudo pacman -S ffmpeg"
                        else:
                            return "Use your distribution's package manager to install ffmpeg"
                except FileNotFoundError:
                    return "Use your distribution's package manager to install ffmpeg"

            case 'darwin':  # macOS
                return "brew install ffmpeg"

            case 'windows':
                return "Download from https://ffmpeg.org/download.html or use Chocolatey: choco install ffmpeg"

            case _:
                return "Please install ffmpeg using your system's package manager"

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
        """Record using FFmpeg backend."""
        try:
            from matplotlib.animation import FuncAnimation
        except ImportError:
            raise RuntimeError("matplotlib is required for FFmpeg backend")

        if fps <= 0:
            raise ValueError("fps must be positive")
        if interval <= 0:
            raise ValueError("interval must be positive")

        if not self.is_available():
            raise RuntimeError(
                f"ffmpeg is not installed. Please install ffmpeg to record videos.\n"
                f"Installation command: {self.get_install_instructions()}"
            )

        try:
            start_time = time.time()

            # Create animation
            with self._lock:
                self._anim = FuncAnimation(
                    self._frame_renderer._fig,
                    self._frame_renderer.update_frame,
                    frames=frames,
                    fargs=(disk,),
                    save_count=total_frames,
                    cache_frame_data=False,
                    interval=interval
                )

            # Save animation to video file
            with self._lock:
                def progress_callback(frame: int, total: int):
                    if pbar:
                        pbar.n = min(frame + 1, total)
                        pbar.refresh()

                self._anim.save(
                    filename,
                    writer="ffmpeg",
                    fps=fps,
                    progress_callback=progress_callback
                )

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
                    if self._anim:
                        self._anim.event_source.stop()
                        self._anim = None
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
        """Clean up resources used by the FFmpeg backend."""
        try:
            with self._lock:
                if self._anim:
                    self._anim.event_source.stop()
                    self._anim = None
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

from .backend import AnimationBackend
from .ffmpeg import FFmpegBackend
from .opencv import OpenCVBackend
from ..frame_renderer import FrameRenderer


def get_available_backends(frame_renderer: FrameRenderer) -> list[AnimationBackend]:
    """
    Get a list of available animation backends.

    Parameters:
    -----------
    frame_renderer: FrameRenderer
        The frame renderer instance

    Returns:
    --------
    list[AnimationBackend]
        List of available backends, ordered by preference
    """
    return [
        backend for backend in (
            backend_factory(frame_renderer)
            for backend_factory in (FFmpegBackend, OpenCVBackend)
        )
        if backend.is_available()
    ]


def get_best_backend(frame_renderer: FrameRenderer) -> AnimationBackend:
    """
    Get the best available animation backend.

    Parameters:
    -----------
    frame_renderer: FrameRenderer
        The frame renderer instance

    Returns:
    --------
    AnimationBackend
        The best available backend

    Raises:
    -------
    RuntimeError
        If no backends are available
    """

    if not (available_backends := get_available_backends(frame_renderer)):
        raise RuntimeError((
            "No animation backends are available.\n\n"
            f"FFmpeg: {FFmpegBackend.get_install_instructions()}\n"
            f"OpenCV: {OpenCVBackend.get_install_instructions()}\n\n"
            "Please install at least one of these backends."
        ))

    return available_backends[0]

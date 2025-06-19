import pytest
from unittest.mock import Mock, patch
from qiskit import QuantumCircuit
from tqdm import tqdm
import numpy as np

from qiskit_state_evolution_recorder.backend import (
    AnimationBackend, FFmpegBackend, OpenCVBackend,
    get_available_backends, get_best_backend
)
from qiskit_state_evolution_recorder.frame_renderer import FrameRenderer


@pytest.fixture
def simple_circuit():
    """Create a simple quantum circuit for testing."""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    return qc


@pytest.fixture
def frame_renderer(simple_circuit):
    """Create a FrameRenderer instance for testing."""
    renderer = FrameRenderer(simple_circuit)
    yield renderer
    renderer.close()


class TestAnimationBackend:
    """Test the abstract AnimationBackend class."""

    def test_abstract_methods(self):
        """Test that AnimationBackend is abstract and cannot be instantiated."""
        with pytest.raises(TypeError):
            AnimationBackend()


class TestFFmpegBackend:
    """Test the FFmpegBackend class."""

    def test_initialization(self, frame_renderer):
        """Test FFmpegBackend initialization."""
        backend = FFmpegBackend(frame_renderer)
        assert backend._frame_renderer == frame_renderer
        assert backend._lock is not None
        assert backend._anim is None

    @patch('qiskit_state_evolution_recorder.backend.ffmpeg.FFmpegBackend.is_available')
    def test_is_available_true(self, mock_is_available, frame_renderer):
        """Test is_available when ffmpeg is available."""
        mock_is_available.return_value = True
        backend = FFmpegBackend(frame_renderer)
        assert backend.is_available() is True

    @patch('qiskit_state_evolution_recorder.backend.ffmpeg.FFmpegBackend.is_available')
    def test_is_available_false(self, mock_is_available, frame_renderer):
        """Test is_available when ffmpeg is not available."""
        mock_is_available.return_value = False
        backend = FFmpegBackend(frame_renderer)
        assert backend.is_available() is False

    def test_get_install_instructions(self, frame_renderer):
        """Test get_install_instructions returns a string."""
        backend = FFmpegBackend(frame_renderer)
        instructions = backend.get_install_instructions()
        assert isinstance(instructions, str)
        assert len(instructions) > 0

    def test_cleanup(self, frame_renderer):
        """Test cleanup method."""
        backend = FFmpegBackend(frame_renderer)
        # Should not raise any exceptions
        backend.cleanup()


class TestOpenCVBackend:
    """Test the OpenCVBackend class."""

    def test_initialization(self, frame_renderer):
        """Test OpenCVBackend initialization."""
        backend = OpenCVBackend(frame_renderer)
        assert backend._frame_renderer == frame_renderer
        assert backend._lock is not None
        assert backend._writer is None

    @patch('qiskit_state_evolution_recorder.backend.opencv.CV2_AVAILABLE', True)
    def test_is_available_true(self, frame_renderer):
        """Test is_available when OpenCV is available."""
        backend = OpenCVBackend(frame_renderer)
        assert backend.is_available() is True

    @patch('qiskit_state_evolution_recorder.backend.opencv.CV2_AVAILABLE', False)
    def test_is_available_false(self, frame_renderer):
        """Test is_available when OpenCV is not available."""
        backend = OpenCVBackend(frame_renderer)
        assert backend.is_available() is False

    def test_get_install_instructions(self, frame_renderer):
        """Test get_install_instructions returns a string."""
        backend = OpenCVBackend(frame_renderer)
        instructions = backend.get_install_instructions()
        assert isinstance(instructions, str)
        assert "opencv-python" in instructions

    def test_cleanup(self, frame_renderer):
        """Test cleanup method."""
        backend = OpenCVBackend(frame_renderer)
        # Should not raise any exceptions
        backend.cleanup()


class TestBackendSelection:
    """Test backend selection functions."""

    def test_get_available_backends_none(self, frame_renderer):
        """Test get_available_backends when no backends are available."""
        with patch.object(FFmpegBackend, 'is_available', return_value=False), \
             patch.object(OpenCVBackend, 'is_available', return_value=False):
            backends = get_available_backends(frame_renderer)
            assert len(backends) == 0

    def test_get_available_backends_ffmpeg_only(self, frame_renderer):
        """Test get_available_backends when only FFmpeg is available."""
        with patch.object(FFmpegBackend, 'is_available', return_value=True), \
             patch.object(OpenCVBackend, 'is_available', return_value=False):
            backends = get_available_backends(frame_renderer)
            assert len(backends) == 1
            assert isinstance(backends[0], FFmpegBackend)

    def test_get_available_backends_opencv_only(self, frame_renderer):
        """Test get_available_backends when only OpenCV is available."""
        with patch.object(FFmpegBackend, 'is_available', return_value=False), \
             patch.object(OpenCVBackend, 'is_available', return_value=True):
            backends = get_available_backends(frame_renderer)
            assert len(backends) == 1
            assert isinstance(backends[0], OpenCVBackend)

    def test_get_available_backends_both(self, frame_renderer):
        """Test get_available_backends when both backends are available."""
        with patch.object(FFmpegBackend, 'is_available', return_value=True), \
             patch.object(OpenCVBackend, 'is_available', return_value=True):
            backends = get_available_backends(frame_renderer)
            assert len(backends) == 2
            assert isinstance(backends[0], FFmpegBackend)  # FFmpeg should be first
            assert isinstance(backends[1], OpenCVBackend)

    def test_get_best_backend_success(self, frame_renderer):
        """Test get_best_backend when backends are available."""
        with patch.object(FFmpegBackend, 'is_available', return_value=True):
            backend = get_best_backend(frame_renderer)
            assert isinstance(backend, FFmpegBackend)

    def test_get_best_backend_failure(self, frame_renderer):
        """Test get_best_backend when no backends are available."""
        with patch.object(FFmpegBackend, 'is_available', return_value=False), \
             patch.object(OpenCVBackend, 'is_available', return_value=False):
            with pytest.raises(RuntimeError) as exc_info:
                get_best_backend(frame_renderer)
            error_msg = str(exc_info.value)
            assert "No animation backends are available" in error_msg
            assert "FFmpeg:" in error_msg
            assert "OpenCV:" in error_msg


class TestBackendIntegration:
    """Test integration of backends with actual recording."""

    @patch('matplotlib.animation.FuncAnimation')
    @patch('qiskit_state_evolution_recorder.backend.ffmpeg.FFmpegBackend.is_available')
    def test_ffmpeg_backend_record(self, mock_is_available, mock_func_animation, frame_renderer, tmp_path):
        """Test FFmpeg backend recording."""
        mock_is_available.return_value = True
        mock_anim = Mock()
        mock_func_animation.return_value = mock_anim

        backend = FFmpegBackend(frame_renderer)
        output_file = tmp_path / "test_output.mp4"

        # Create a simple frame generator
        def frame_generator():
            yield b"fake_frame_data"

        # Mock the save method
        mock_anim.save = Mock()

        # Record the animation
        with tqdm(total=1) as pbar:
            backend.record(
                filename=str(output_file),
                frames=frame_generator(),
                total_frames=1,
                fps=30,
                interval=100,
                disk=False,
                pbar=pbar
            )

        # Verify that save was called
        mock_anim.save.assert_called_once()

    @patch('qiskit_state_evolution_recorder.backend.opencv.CV2_AVAILABLE', True)
    @patch('qiskit_state_evolution_recorder.backend.opencv.cv2')
    def test_opencv_backend_record(self, mock_cv2, frame_renderer, tmp_path):
        """Test OpenCV backend recording."""
        # Mock OpenCV
        mock_writer = Mock()
        mock_writer.isOpened.return_value = True
        mock_cv2.VideoWriter.return_value = mock_writer
        mock_cv2.VideoWriter_fourcc.return_value = "mp4v"

        # Create a proper mock numpy array for the frame
        mock_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_cv2.cvtColor.return_value = mock_frame

        backend = OpenCVBackend(frame_renderer)
        output_file = tmp_path / "test_output.mp4"

        # Create a simple frame generator with numpy array
        def frame_generator():
            yield np.zeros((100, 100, 3), dtype=np.uint8)

        # Record the animation
        with tqdm(total=1) as pbar:
            backend.record(
                filename=str(output_file),
                frames=frame_generator(),
                total_frames=1,
                fps=30,
                interval=100,
                disk=False,
                pbar=pbar
            )

        # Verify that VideoWriter was created and used
        mock_cv2.VideoWriter.assert_called_once()
        mock_writer.write.assert_called_once()
        mock_writer.release.assert_called_once()

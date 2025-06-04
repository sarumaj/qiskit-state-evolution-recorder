import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit_state_evolution_recorder.animation import AnimationRecorder
from qiskit_state_evolution_recorder.frame_renderer import FrameRenderer
import os
from tempfile import gettempdir


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


@pytest.fixture
def animation_recorder(frame_renderer):
    """Create an AnimationRecorder instance for testing."""
    return AnimationRecorder(frame_renderer)


def test_initialization(frame_renderer):
    """Test the initialization of AnimationRecorder."""
    recorder = AnimationRecorder(frame_renderer)
    assert recorder._frame_renderer == frame_renderer
    assert recorder._lock is not None


def test_record_to_disk(frame_renderer, tmp_path):
    """Test recording animation to disk."""
    recorder = AnimationRecorder(frame_renderer)
    output_file = tmp_path / "test_output.mp4"

    # Create a simple frame generator
    def frame_generator():
        state = Statevector.from_label('00')
        operations = []
        frame_data = (state, operations)
        yield frame_renderer.render_frame(0, frame_data, disk=True)

    # Record the animation
    from tqdm import tqdm
    with tqdm(total=1) as pbar:
        recorder.record(
            filename=str(output_file),
            frames=frame_generator(),
            total_frames=1,
            fps=30,
            interval=100,
            disk=True,
            pbar=pbar
        )

    assert output_file.exists()
    assert output_file.stat().st_size > 0


def test_record_to_memory(frame_renderer, tmp_path):
    """Test recording animation to memory."""
    recorder = AnimationRecorder(frame_renderer)
    output_file = tmp_path / "test_output.mp4"

    # Create a simple frame generator
    def frame_generator():
        state = Statevector.from_label('00')
        operations = []
        frame_data = (state, operations)
        yield frame_renderer.render_frame(0, frame_data, disk=False)

    # Record the animation
    from tqdm import tqdm
    with tqdm(total=1) as pbar:
        recorder.record(
            filename=str(output_file),
            frames=frame_generator(),
            total_frames=1,
            fps=30,
            interval=100,
            disk=False,
            pbar=pbar
        )

    assert output_file.exists()
    assert output_file.stat().st_size > 0


def test_cleanup_temp_files(frame_renderer, tmp_path):
    """Test cleanup of temporary files."""
    recorder = AnimationRecorder(frame_renderer)
    # Create some temporary files
    temp_dir = gettempdir()
    test_files = []
    for i in range(3):
        filename = os.path.join(temp_dir, f"{i}.png")
        with open(filename, 'w') as f:
            f.write('test')
        test_files.append(filename)

    # Clean up the files
    recorder._cleanup_temp_files(3)

    # Check that files are removed
    for filename in test_files:
        assert not os.path.exists(filename)


def test_record_with_progress_bar(frame_renderer, tmp_path):
    """Test recording with progress bar."""
    recorder = AnimationRecorder(frame_renderer)
    output_file = tmp_path / "test_output.mp4"

    # Create a simple frame generator
    def frame_generator():
        state = Statevector.from_label('00')
        operations = []
        frame_data = (state, operations)
        yield frame_renderer.render_frame(0, frame_data, disk=False)

    # Record the animation with progress bar
    from tqdm import tqdm
    with tqdm(total=1) as pbar:
        recorder.record(
            filename=str(output_file),
            frames=frame_generator(),
            total_frames=1,
            fps=30,
            interval=100,
            disk=False,
            pbar=pbar
        )

    assert output_file.exists()
    assert output_file.stat().st_size > 0


def test_record_with_multiple_frames(frame_renderer, tmp_path):
    """Test recording with multiple frames."""
    recorder = AnimationRecorder(frame_renderer)
    output_file = tmp_path / "test_output.mp4"

    # Set up the frame renderer
    # frame_renderer._selected_qubits = [0, 1]  # Set selected qubits
    frame_renderer._setup_layout(num_cols=2)  # Set up layout with 2 columns

    # Create a frame generator with multiple frames
    def frame_generator():
        state = Statevector.from_label('00')
        operations = []
        frame_data = (state, operations)
        for i in range(3):
            yield frame_renderer.render_frame(i, frame_data, disk=False)

    # Record the animation
    from tqdm import tqdm
    with tqdm(total=3) as pbar:
        recorder.record(
            filename=str(output_file),
            frames=frame_generator(),
            total_frames=3,
            fps=30,
            interval=100,
            disk=False,
            pbar=pbar
        )

    assert output_file.exists()
    assert output_file.stat().st_size > 0

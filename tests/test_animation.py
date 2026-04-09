from pathlib import Path
from typing import Iterable, List, Tuple

from qiskit.circuit import InstructionSet  # type: ignore[reportMissingTypeStubs]
from qiskit.quantum_info import Statevector  # type: ignore[reportMissingTypeStubs]
from tqdm import tqdm

from qiskit_state_evolution_recorder.animation import AnimationRecorder
from qiskit_state_evolution_recorder.frame_renderer import FrameRenderer


def test_initialization(frame_renderer: FrameRenderer):
    """Test the initialization of AnimationRecorder."""
    recorder = AnimationRecorder(frame_renderer)
    assert recorder._frame_renderer == frame_renderer  # type: ignore[reportPrivateUsage]
    assert recorder._lock is not None  # type: ignore[reportPrivateUsage]


def test_record_to_disk(frame_renderer: FrameRenderer, tmp_path: Path):
    """Test recording animation to disk."""
    recorder = AnimationRecorder(frame_renderer)
    output_file = tmp_path / "test_output.mp4"

    # Create a simple frame generator
    def frame_generator():
        state = Statevector.from_label("00")
        operations: Iterable[InstructionSet] = []
        frame_data: Tuple[Statevector, List[InstructionSet]] = (state, operations)
        yield frame_renderer.render_frame(0, frame_data, disk=True)

    # Record the animation
    with tqdm(total=1) as pbar:
        recorder.record(  # type: ignore[reportUnknownMemberType]
            filename=str(output_file),
            frames=frame_generator(),
            total_frames=1,
            fps=30,
            interval=100,
            disk=True,
            pbar=pbar,
        )

    assert output_file.exists()
    assert output_file.stat().st_size > 0


def test_record_to_memory(frame_renderer: FrameRenderer, tmp_path: Path):
    """Test recording animation to memory."""
    recorder = AnimationRecorder(frame_renderer)
    output_file = tmp_path / "test_output.mp4"

    # Create a simple frame generator
    def frame_generator():
        state = Statevector.from_label("00")
        operations: Iterable[InstructionSet] = []
        frame_data: Tuple[Statevector, List[InstructionSet]] = (state, operations)
        yield frame_renderer.render_frame(0, frame_data, disk=False)

    # Record the animation
    with tqdm(total=1) as pbar:
        recorder.record(  # type: ignore[reportUnknownMemberType]
            filename=str(output_file),
            frames=frame_generator(),
            total_frames=1,
            fps=30,
            interval=100,
            disk=False,
            pbar=pbar,
        )

    assert output_file.exists()
    assert output_file.stat().st_size > 0


def test_record_with_progress_bar(frame_renderer: FrameRenderer, tmp_path: Path):
    """Test recording with progress bar."""
    recorder = AnimationRecorder(frame_renderer)
    output_file = tmp_path / "test_output.mp4"

    # Create a simple frame generator
    def frame_generator():
        state = Statevector.from_label("00")
        operations: Iterable[InstructionSet] = []
        frame_data: Tuple[Statevector, List[InstructionSet]] = (state, operations)
        yield frame_renderer.render_frame(0, frame_data, disk=False)

    # Record the animation with progress bar
    with tqdm(total=1) as pbar:
        recorder.record(  # type: ignore[reportUnknownMemberType]
            filename=str(output_file),
            frames=frame_generator(),
            total_frames=1,
            fps=30,
            interval=100,
            disk=False,
            pbar=pbar,
        )

    assert output_file.exists()
    assert output_file.stat().st_size > 0


def test_record_with_multiple_frames(frame_renderer: FrameRenderer, tmp_path: Path):
    """Test recording with multiple frames."""
    recorder = AnimationRecorder(frame_renderer)
    output_file = tmp_path / "test_output.mp4"

    # Set up layout with 2 columns
    frame_renderer._setup_layout(num_cols=2)  # type: ignore[reportPrivateUsage]

    # Create a frame generator with multiple frames
    def frame_generator():
        state = Statevector.from_label("00")
        operations: Iterable[InstructionSet] = []
        frame_data: Tuple[Statevector, List[InstructionSet]] = (state, operations)
        for i in range(3):
            yield frame_renderer.render_frame(i, frame_data, disk=False)

    # Record the animation
    with tqdm(total=3) as pbar:
        recorder.record(  # type: ignore[reportUnknownMemberType]
            filename=str(output_file),
            frames=frame_generator(),
            total_frames=3,
            fps=30,
            interval=100,
            disk=False,
            pbar=pbar,
        )

    assert output_file.exists()
    assert output_file.stat().st_size > 0

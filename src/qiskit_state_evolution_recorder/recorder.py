from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from multiprocessing import Process, Queue, cpu_count, Lock
from typing import Generator, Union, Optional, List
from numpy import ndarray, uint8
from tqdm import tqdm
import queue
from pathlib import Path

from .state_evolution import generate_states, interpolate_states, group_instructions
from .frame_renderer import FrameRenderer
from .animation import AnimationRecorder


class FrameRendererProcess:
    """Handles the multiprocessing logic for frame rendering."""

    def __init__(self, frame_renderer: FrameRenderer):
        self._frame_renderer = frame_renderer
        self._task_queue: Queue = Queue()
        self._result_queue: Queue = Queue()
        self._lock = Lock()
        self._processes: List[Process] = []

    def start(self) -> None:
        """Start the rendering processes."""
        self._processes = [
            Process(target=self._render_task)
            for _ in range(cpu_count())
        ]
        for process in self._processes:
            process.start()

    def stop(self) -> None:
        """Stop the rendering processes."""
        for _ in self._processes:
            self._task_queue.put(None)
        for process in self._processes:
            process.join()

    def _render_task(self) -> None:
        """Process rendering tasks from the queue."""
        while True:
            if (args := self._task_queue.get()) is None:
                break

            index, frame_data, disk = args
            with self._lock:
                result = self._frame_renderer.render_frame(index, frame_data, disk)
                self._result_queue.put((index, result))

    def render_frames(
        self,
        states: Generator,
        total_frames: int,
        disk: bool = False,
        pbar: Optional[tqdm] = None
    ) -> Generator[Union[str, ndarray[uint8]], None, None]:
        """Render frames using multiple processes."""
        frames_processed = 0
        frame_buffer = {}

        # Send tasks to the processes
        for index, frame_data in enumerate(states):
            self._task_queue.put((index, frame_data, disk))

        # Collect frames from the processes
        while frames_processed < total_frames:
            try:
                index, result = self._result_queue.get(timeout=30)
                frame_buffer[index] = result

                # Yield frames in order
                while frames_processed in frame_buffer:
                    yield frame_buffer.pop(frames_processed)
                    frames_processed += 1
                    if pbar:
                        pbar.update(1)

            except queue.Empty:
                raise TimeoutError("Frame collection timed out")


class StateEvolutionRecorder:
    """
    A class to record the evolution of a quantum state through a quantum circuit.

    This class handles the recording of quantum state evolution through a circuit,
    including state interpolation and frame rendering for animation.

    Attributes:
    -----------
    size: int
        The number of frames to record
    """

    def __init__(
        self,
        qc: QuantumCircuit,
        initial_state: Optional[Statevector] = None,
        figsize: Optional[tuple[float, float]] = None,
        dpi: Optional[float] = None,
        num_cols: int = 5,
        select: Optional[List[int]] = None,
        style: Optional[dict] = None,
    ) -> None:
        """
        Initialize the StateEvolutionRecorder.

        Parameters:
        -----------
        qc: QuantumCircuit
            The quantum circuit to record
        initial_state: Optional[Statevector]
            The initial state of the quantum circuit, default is |0>^n
        figsize: Optional[tuple[float, float]]
            The size of the figure, default is (6, 6)
        dpi: Optional[float]
            The resolution of the figure, default is 100
        num_cols: int
            Number of columns in the visualization grid
        select: Optional[List[int]]
            The qubits to select for the bloch vectors to display, default is all qubits
        style: Optional[dict]
            The style of the quantum circuit diagram, default is {'name':'textbook'}

        Raises:
        -------
        ValueError
            If the circuit is empty or invalid
            If the initial state dimension doesn't match the circuit
            If selected qubits are invalid
        """
        if not qc or not qc.data:
            raise ValueError("Quantum circuit cannot be empty")

        self._qc = qc
        self._initial_state = initial_state or Statevector.from_label('0'*qc.num_qubits)

        # Validate initial state
        if self._initial_state.dim != 2**qc.num_qubits:
            raise ValueError(
                f"Initial state dimension {self._initial_state.dim} "
                f"doesn't match circuit size {2**qc.num_qubits}"
            )

        # Validate selected qubits
        if select is not None:
            if not all(0 <= q < qc.num_qubits for q in select):
                raise ValueError("Selected qubits must be within circuit range")
            self._selected_qubits = sorted(select)
        else:
            self._selected_qubits = list(range(qc.num_qubits))

        self._size = 0
        self._states = None
        self._lock = Lock()

        # Cache grouped instructions
        self._grouped_instructions = group_instructions(self._qc)

        # Initialize components
        self._frame_renderer = FrameRenderer(
            qc=self._qc,
            figsize=figsize,
            dpi=dpi,
            num_cols=num_cols,
            select=self._selected_qubits,
            style=style
        )
        self._animation_recorder = AnimationRecorder(self._frame_renderer)
        self._frame_renderer_process = FrameRendererProcess(self._frame_renderer)

        # Evolve the circuit to cover ground states
        self.evolve()

    def evolve(self, intermediate_steps: int = 0) -> None:
        """
        Evolve the initial state through the circuit.

        Parameters:
        -----------
        intermediate_steps: int
            The number of intermediate steps to interpolate between two adjacent states

        Raises:
        -------
        ValueError
            If intermediate_steps is negative
        """
        if intermediate_steps < 0:
            raise ValueError("intermediate_steps cannot be negative")

        # Get the number of instruction groups from cache
        num_groups = len(self._grouped_instructions)

        if intermediate_steps <= 1:
            self._states = generate_states(self._qc, self._initial_state, self._grouped_instructions)
            self._size = num_groups + 1
            return

        self._states = interpolate_states(self._qc, self._initial_state, intermediate_steps, self._grouped_instructions)
        # Update size to include intermediate steps
        self._size = num_groups * intermediate_steps + 1

    def record(
        self,
        filename: str,
        *,
        fps: int = 60,
        interval: int = 200,
        disk: bool = False
    ) -> None:
        """
        Record the frames into a video file.

        Parameters:
        -----------
        filename: str
            The name of the video file to create
        fps: int
            The number of frames per second
        interval: int
            The interval between each frame in milliseconds
        disk: bool
            Whether to save the frames to disk

        Raises:
        -------
        ValueError
            If fps or interval is invalid
            If the output directory doesn't exist
        """
        if fps <= 0:
            raise ValueError("fps must be positive")
        if interval <= 0:
            raise ValueError("interval must be positive")

        # Ensure output directory exists
        output_path = Path(filename)
        if not output_path.parent.exists():
            raise ValueError(f"Output directory {output_path.parent} does not exist")

        with self._lock:
            try:
                with (
                    tqdm(total=self._size, desc="Rendering frames") as pbar_rendering,
                    tqdm(total=self._size, desc="Encoding video") as pbar_encoding
                ):
                    self._frame_renderer_process.start()
                    self._animation_recorder.record(
                        filename=filename,
                        frames=self._frame_renderer_process.render_frames(
                            self._states,
                            self._size,
                            disk,
                            pbar_rendering
                        ),
                        total_frames=self._size,
                        fps=fps,
                        interval=interval,
                        disk=disk,
                        pbar=pbar_encoding
                    )
            finally:
                self._frame_renderer_process.stop()

    @property
    def size(self) -> int:
        """The number of frames to record."""
        return self._size

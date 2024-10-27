from qiskit.circuit import InstructionSet, QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_bloch_vector
from qiskit.visualization.state_visualization import _bloch_multivector_data

from numpy import linspace, frombuffer, uint8, ndarray, ceil
from numpy.linalg import norm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation

from typing import Iterable, Union, Generator
from multiprocessing import Process, Queue, cpu_count
from tempfile import gettempdir
from os import remove, path as os_path
from PIL.Image import open as open_image
import time


class StateEvolutionRecorder:
    """
    A class to record the evolution of a quantum state through a quantum circuit

    Methods:
    --------
    evolve(intermediate_steps: int = 0)
        Evolve the initial state through the circuit

    record(filename: str, *, fps: int = 60, interval: int = 200, disk: bool = False)
        Record the frames into a video file    
    """

    def __init__(
        self,
        qc: QuantumCircuit,
        initial_state: Statevector = None,
        figsize: tuple[float, float] = None,
        dpi: float = None,
        num_cols: int = 5,
        select: list[int] = None,
        style: dict = None,
    ) -> None:
        """
        Initialize the StateEvolutionRecorder

        Parameters:
        -----------
        qc: QuantumCircuit
            The quantum circuit to record

        initial_state: Statevector
            The initial state of the quantum circuit, default is |0>^n

        figsize: tuple[float, float]
            The size of the figure, default is (6, 6)

        dpi: float
            The resolution of the figure, default is 100

        select: list[int]
            The qubits to select for the bloch vectors to display, default is all qubits

        style: dict
            The style of the quantum circuit diagram, default is {'name':'textbook'}
        """

        self._qc = qc
        self._states = None
        self._size = 1
        self._initial_state = initial_state or Statevector.from_label(
            '0'*self._qc.num_qubits)
        self._fig = plt.figure(figsize=figsize, dpi=dpi)
        self._selected_qubits = list(
            range(self._qc.num_qubits)) if not select else sorted(select)
        num_cols = num_cols if 0 < num_cols <= len(
            self._selected_qubits) else 3
        num_rows = 1 + int(ceil(len(self._selected_qubits) / num_cols))
        self._gs = gridspec.GridSpec(num_rows, num_cols)
        self._gs.set_height_ratios([3] + [2] * (num_rows - 1))
        self._ax = [[self._fig.add_subplot(self._gs[0, :])]]
        self._qubit_partitions = [
            self._selected_qubits[i:i + num_cols]
            for i in range(
                0,
                min(
                    len(self._selected_qubits),
                    num_cols * num_rows
                ),
                num_cols
            )
        ]
        self._ax.extend([
            [
                self._fig.add_subplot(self._gs[i + 1, j], projection="3d")
                for j in range(len(partition))
            ]
            for i, partition in enumerate(self._qubit_partitions)
        ])
        for axes in self._ax[1:]:
            for ax in axes:
                ax.axis('off')
        self._text = None
        qc.draw(output='mpl',
                fold=-1,
                style=style if style else {'name': 'textbook'},
                ax=self._ax[0][0])

    def evolve(self, intermediate_steps: int = 0):
        """
        Evolve the initial state through the circuit

        Parameters:
        -----------
        intermediate_steps: int
            The number of intermediate steps to interpolate between two adjacent states, default is 0
        """

        def group_instructions() -> list[list[InstructionSet]]:
            """
            Group the instructions taking place in parallel based on the qubits involved or barriers

            Returns:
            --------
            list[list[InstructionSet]]: A list of grouped instructions taking place in parallel
            """

            current_group = []
            active_qubits = set()
            instructions = []

            for instruction in self._qc.data:
                if instruction.operation.name == 'barrier':
                    if current_group:
                        instructions.append(current_group)
                        current_group = []
                    active_qubits.clear()
                    continue

                if instruction.operation.name == 'measure':
                    continue

                # Get the qubits involved in the instruction
                qubits_involved = set(
                    f"{q._register.prefix}{q._index}" for q in instruction.qubits)

                # If the qubits involved in the current instruction are not the same as the active qubits
                if active_qubits & qubits_involved:  # If there is an intersection
                    instructions.append(current_group)
                    current_group = []
                    active_qubits.clear()

                current_group.append(instruction)
                active_qubits.update(qubits_involved)

            if current_group:
                instructions.append(current_group)

            return instructions

        # Group the instructions taking place in parallel
        grouped_instructions = group_instructions()

        # The number of frames to render
        self._size = len(grouped_instructions)+1

        def generate_states() -> Generator[tuple[Statevector, Iterable[InstructionSet]], None, None]:
            """
            Generate the states of the quantum circuit at each step

            Yields:
            -------
            tuple[Statevector, Iterable[InstructionSet]]: 
                A tuple containing the state of the quantum circuit and 
                the instructions responsible for the state transition
            """

            state = self._initial_state
            for instructionSet in grouped_instructions:
                yield (state, instructionSet)

                sub_circuit = QuantumCircuit(*self._qc.qregs, *self._qc.cregs)
                for instruction in instructionSet:
                    sub_circuit.append(instruction.operation,
                                       instruction.qubits, instruction.clbits)

                state = state.evolve(sub_circuit)

            yield (state, [])

        # If there are no intermediate steps
        if intermediate_steps <= 1:
            self._states = generate_states()
            return

        def interpolate_states() -> Generator[tuple[Statevector, Iterable[InstructionSet]], None, None]:
            """
            Interpolate between two adjacent states to smooth the transition

            Yields:
            -------
            tuple[Statevector, Iterable[InstructionSet]]: 
                A tuple containing the state of the quantum circuit and 
                the instructions responsible for the state transition
            """

            states = list(generate_states())
            for lhs, rhs in zip(states[:-1], states[1:]):
                for i in linspace(0, 1, intermediate_steps, endpoint=False):
                    intermediate_state = (
                        1 - i) * lhs[0].data + i * rhs[0].data
                    state_norm = norm(intermediate_state)
                    if state_norm:
                        intermediate_state = intermediate_state / state_norm
                    yield (Statevector(intermediate_state), lhs[1])

            yield states[-1]

        # Interpolate between each pair of two adjacent states to smooth the transition
        self._states = interpolate_states()
        # Update the size to reflect the number of intermediate steps
        self._size += len(grouped_instructions) * (intermediate_steps-1)

    def _render_frame(
        self,
        index: int,
        frame_data: tuple[Statevector, Iterable[InstructionSet]],
        disk: bool
    ) -> tuple[int, Union[str, ndarray[uint8]]]:
        """
        Render a frame. If disk is True, save the frames to disk at the OS-specific temporary directory.

        Parameters:
        -----------
        index: int
            The index of the frame

        frame_data: tuple[Statevector, Iterable[InstructionSet]]
            A tuple containing the state of the quantum circuit and
            the instructions responsible for the state transition

        disk: bool
            Whether to save the frame to hard disk or not

        Returns:
        --------
        tuple[int, Union[str, ndarray[uint8]]]:
            A tuple containing the index of the frame and the frame data
            either as a filename or as an image array
        """

        # Remove the text describing the operations
        if self._text:
            self._text.remove()

        # Clear the axes for the bloch vectors
        for axes in self._ax[1:]:
            for ax in axes:
                ax.clear()

        state, operations = frame_data

        # Allocate the axes for the bloch vectors
        if len(self._ax) < 2:
            self._ax.extend([
                [
                    self._fig.add_subplot(self._gs[i + 1, j], projection="3d")
                    for j in range(len(partition))
                ]
                for i, partition in enumerate(self._qubit_partitions)
            ])

        # Plot the bloch vectors
        bloch_data = _bloch_multivector_data(state)
        for i, partition in enumerate(self._qubit_partitions):
            for j in range(len(partition)):
                plot_bloch_vector(
                    bloch_data[partition[j]], f"q{partition[j]}", ax=self._ax[i + 1][j])

        # Describe the operations taking place
        if operations:
            fragments = []
            for gate in operations:
                fragments.append("{0} -> {1}".format(gate.operation.name,
                                 [f"{q._register._name}{q._index}" for q in gate.qubits]))

            self._text = self._fig.text(
                0.5, 0.95,
                " | ".join(fragments),
                ha='center', va='bottom',
                fontsize=15,
                transform=self._fig.transFigure
            )

        # Update the figure
        self._fig.canvas.draw()

        # Save the frame to disk
        if disk:
            filename = os_path.join(gettempdir(), f"{index}.png")
            self._fig.savefig(filename, format="png")
            return index, filename

        # Return the frame as an image array
        buf = self._fig.canvas.tostring_rgb()
        dim = self._fig.canvas.get_width_height()[::-1] + (3,)
        img = frombuffer(buf, dtype=uint8).reshape(dim)
        return index, img

    def _render_frames(self, disk: bool = False) -> Generator[Union[str, ndarray[uint8]], None, None]:
        """
        Render the frames

        Parameters:
        -----------
        disk: bool
            Whether to save the frames to disk or not

        Yields:
        -------
        Union[str, ndarray[uint8]]: 
            The frame data either as a filename or as an image array
        """

        def generate_frames() -> Generator[tuple[int, Union[str, ndarray[uint8]]], None, None]:
            """
            Generate the frames using multiple processes

            Yields:
            -------
            tuple[int, Union[str, ndarray[uint8]]]: 
                A tuple containing the index of the frame and the frame data
                either as a filename or as an image array
            """

            task_queue = Queue()
            result_queue = Queue()

            def task():
                """
                The task to be executed by each process
                """

                while True:
                    args = task_queue.get()

                    # If there are no arguments left, terminate the process
                    if args is None:
                        break

                    index, frame_data, disk = args
                    result_queue.put(self._render_frame(
                        index, frame_data, disk))

            # Create a process for each CPU core
            processes = [Process(target=task) for _ in range(cpu_count())]
            for process in processes:
                process.start()

            # Distribute the tasks among the processes to render a frame for each state
            for index, frame_data in enumerate(self._states):
                task_queue.put((index, frame_data, disk))

            # Terminate the processes
            for _ in processes:
                task_queue.put(None)

            # Yield the frames in the order they were rendered
            for i in range(self._size):
                yield result_queue.get()
                self._progress_callback(i, self._size, msg="Rendering frames")

            # Wait for all the processes to terminate
            for _ in processes:
                process.join()

        # Sort the frames in the order they were scheduled
        return (frame for _, frame in sorted(generate_frames(), key=lambda x: x[0]))

    def _update(self, image: Union[str, ndarray[uint8]], disk: bool):
        """
        Update the figure with the new frame

        Parameters:
        -----------
        image: Union[str, ndarray[uint8]]
            The frame data either as a filename or as an image array

        disk: bool
            Whether the frame data is a filename or an image array
        """

        # Remove the text describing the operations
        if self._text:
            self._text.remove()

        # Remove the axes from the figure
        if len(self._ax) > 1:
            for axes in self._ax:
                for ax in axes:
                    ax.clear()
                    ax.remove()
            # Create a new axes for the image
            self._ax = [[self._fig.add_axes([0, 0, 1, 1])]]

        # If the frame data is a filename, open the image
        if disk:
            image = open_image(image)

        # Display the image
        self._ax[0][0].clear()
        self._ax[0][0].imshow(image, aspect='auto')
        self._ax[0][0].axis('off')

        # Update the figure
        self._fig.canvas.draw()

    def _progress_callback(self, frame: int, total_frames: int, bar_length: int = 40, msg: str = "Progress:"):
        """
        Display a progress bar

        Parameters:
        -----------
        frame: int
            The current frame number

        total_frames: int
            The total number of frames to render

        bar_length: int
            The length of the progress bar

        msg: str
            The message to display along with the progress bar
        """

        progress = (frame + 1) / total_frames
        percent = int(progress * 100)
        filled_length = int(bar_length * progress)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)
        template = f"\r{msg}: |{bar}| {percent:>3}% ({{frame:>{len(str(total_frames))}}}/{total_frames})"
        print(template.format(frame=frame + 1), end=" "*10)

    def record(self, filename: str, *, fps: int = 60, interval: int = 200, disk: bool = False):
        """
        Record the frames into a video file

        Parameters:
        -----------
        filename: str
            The name of the video file to create

        fps: int
            The number of frames per second

        interval: int
            The interval between each frame in milliseconds

        disk: bool
            Whether to save the frames to disk or not
        """

        try:
            start_time = time.time()
            # Create an animation object
            anim = FuncAnimation(self._fig,
                                 self._update,
                                 frames=self._render_frames(disk),
                                 fargs=(disk,),
                                 save_count=self._size,
                                 cache_frame_data=False,
                                 interval=interval)

            def callback(frame, total_frames):
                return self._progress_callback(frame, total_frames, msg="Encoding video")

            # Save the animation to a video file
            anim.save(filename,
                      writer="ffmpeg",
                      fps=fps,
                      progress_callback=callback)

        except KeyboardInterrupt:
            # If the recording was interrupted, remove the video file
            remove(filename)

        else:
            elapsed_time = time.time() - start_time
            print(
                f"\nRecording finished. Elapsed time: {elapsed_time:.2f} seconds")

        finally:
            # Close the figure
            plt.close(self._fig)

            if not disk:
                return

            # Remove the frames from disk if any
            for index in range(self._size):
                filename = os_path.join(gettempdir(), f"{index}.png")
                try:
                    remove(filename)
                except FileNotFoundError:
                    pass

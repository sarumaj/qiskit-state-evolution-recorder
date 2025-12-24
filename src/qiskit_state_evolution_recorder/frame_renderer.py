# pyright: basic

import logging
import os
from tempfile import gettempdir
from typing import Any, Iterable, List, Optional, Tuple, TypeAlias, Union

FileDescriptorOrPath: TypeAlias = int | str | os.PathLike[str] | os.PathLike[bytes] | bytes


import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib import get_backend
from matplotlib import use as use_backend
from numpy import asarray, ceil, ndarray, uint8
from PIL.Image import open as open_image
from qiskit.circuit import InstructionSet, QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_bloch_vector
from qiskit.visualization.state_visualization import _bloch_multivector_data

from .compatibility import proxy_obj

# Use Agg backend by default to avoid GUI-related issues in headless environments
# This can be overridden before importing if needed
if get_backend() in ("TkAgg", "Qt5Agg"):
    try:
        use_backend("Agg", force=True)
    except Exception:
        pass

logger = logging.getLogger("qiskit_state_evolution_recorder.frame_renderer")


class FrameRenderer:
    """
    A class to handle frame rendering for quantum state visualization.

    This class is responsible for:
    1. Setting up the visualization layout
    2. Rendering quantum states as Bloch spheres
    3. Displaying operation information
    4. Managing frame generation and updates
    """

    def __init__(
        self,
        qc: QuantumCircuit,
        figsize: Optional[Tuple[float, float]] = None,
        dpi: Optional[float] = None,
        num_cols: int = 5,
        select: Optional[List[int]] = None,
        style: Optional[dict] = None,
    ):
        """
        Initialize the FrameRenderer.

        Parameters:
        -----------
        qc: QuantumCircuit
            The quantum circuit to visualize
        figsize: Optional[Tuple[float, float]]
            The size of the figure, default is (6, 6)
        dpi: Optional[float]
            The resolution of the figure, default is 100
        num_cols: int
            Number of columns for Bloch sphere visualization
        select: Optional[List[int]]
            The qubits to select for the bloch vectors to display
        style: Optional[dict]
            The style of the quantum circuit diagram

        Raises:
        -------
        ValueError
            If the circuit is empty or invalid
            If num_cols is invalid
            If selected qubits are invalid
        """
        if not qc or not qc.data:
            raise ValueError("Quantum circuit cannot be empty")

        if num_cols <= 0:
            raise ValueError("num_cols must be positive")

        if select is not None and not all(0 <= q < qc.num_qubits for q in select):
            raise ValueError("Selected qubits must be within circuit range")

        self._qc = qc
        self._fig = plt.figure(figsize=figsize, dpi=dpi)
        self._selected_qubits = list(range(qc.num_qubits)) if not select else sorted(select)
        self._style = style or {"name": "textbook"}
        self._text = None
        self._ax = None
        self._gs = None
        self._qubit_partitions = None

        # Setup the layout for visualization
        self._setup_layout(num_cols)

        # Draw the quantum circuit diagram
        self._qc.draw(output="mpl", fold=-1, style=self._style, ax=self._ax[0][0] if self._ax else None)

    def _setup_layout(self, num_cols: int):
        """
        Setup the layout for visualization.

        This method creates the grid layout for the visualization, including:
        1. Circuit diagram at the top
        2. Bloch spheres for each qubit below
        3. Proper spacing and sizing

        Parameters:
        -----------
        num_cols: int
            Number of columns for Bloch sphere visualization
        """
        num_cols = min(num_cols, len(self._selected_qubits))
        num_rows = 1 + int(ceil(len(self._selected_qubits) / num_cols))

        # Create grid layout
        self._gs = gridspec.GridSpec(num_rows, num_cols)
        self._gs.set_height_ratios([3] + [2] * (num_rows - 1))

        # Add circuit diagram
        if not self._fig:
            raise RuntimeError("Figure not initialized")

        self._ax = [[self._fig.add_subplot(self._gs[0, :])]]

        # Create qubit partitions for Bloch spheres
        self._qubit_partitions = [
            self._selected_qubits[i : i + num_cols] for i in range(0, len(self._selected_qubits), num_cols)
        ]

        # Add Bloch sphere axes
        self._allocate_bloch_axes()

    def _allocate_bloch_axes(self):
        """
        Allocate the axes for the bloch vectors.
        """
        if not self._fig:
            raise RuntimeError("Figure not initialized")

        if not self._gs:
            raise RuntimeError("GridSpec not initialized")

        if not self._ax:
            self._ax = []

        if len(self._ax) < 2:
            self._ax.extend(
                [
                    [self._fig.add_subplot(self._gs[i + 1, j], projection="3d") for j in range(len(partition))]
                    for i, partition in enumerate(self._qubit_partitions or [])
                ]
            )

            # Hide axes for Bloch spheres
            for axes in self._ax[1:]:
                for ax in axes:
                    ax.axis("off")

    def render_frame(
        self, index: int, frame_data: Tuple[Statevector, Iterable[InstructionSet]], disk: bool
    ) -> Union[str, ndarray[Any, Any]]:
        """
        Render a frame with the current quantum state and operations.

        Parameters:
        -----------
        index: int
            The index of the frame
        frame_data: Tuple[Statevector, Iterable[InstructionSet]]
            A tuple containing the state and operations
        disk: bool
            Whether to save the frame to disk

        Returns:
        --------
        Union[str, ndarray[uint8]]
            The frame data (either filename or image array)

        Raises:
        -------
        ValueError
            If the frame data is invalid
        RuntimeError
            If there's an error saving the frame
        """
        if not isinstance(frame_data, tuple) or len(frame_data) != 2:
            raise ValueError("frame_data must be a tuple of (Statevector, Iterable[InstructionSet])")

        state, operations = frame_data
        if not isinstance(state, Statevector):
            raise ValueError("frame_data[0] must be a Statevector")

        if self._text:
            self._text.remove()

        for axes in self._ax[1:] if self._ax else []:
            for ax in axes:
                ax.clear()

        # Allocate the axes for the bloch vectors
        self._allocate_bloch_axes()

        # Plot the bloch vectors
        self._plot_bloch_vectors(state)

        # Update the operation text
        self._update_operation_text(operations)

        if not self._fig:
            raise RuntimeError("Figure not initialized")

        # Draw the figure
        self._fig.canvas.draw()

        # Save the figure to disk
        if disk:
            filename = os.path.join(gettempdir(), f"{index}.png")
            try:
                self._fig.savefig(filename, format="png")
                return filename
            except Exception as e:
                raise RuntimeError(f"Error saving frame to disk: {str(e)}")

        # Convert figure to image array
        try:
            img = asarray(self._fig.canvas.renderer.buffer_rgba())[  # pyright: ignore[reportAttributeAccessIssue]
                :, :, :3
            ]
            return img
        except Exception as e:
            raise RuntimeError(f"Error converting frame to image array: {str(e)}")

    def _plot_bloch_vectors(self, state: Statevector):
        """
        Plot Bloch vectors for the current state.

        Parameters:
        -----------
        state: Statevector
            The quantum state to visualize

        Raises:
        -------
        ValueError
            If the state dimension doesn't match the circuit
        """
        if state.dim != 2**self._qc.num_qubits:
            raise ValueError(f"State dimension {state.dim} doesn't match circuit size {2**self._qc.num_qubits}")

        bloch_data = _bloch_multivector_data(state)
        for i, partition in enumerate(self._qubit_partitions or []):
            for j, qubit in enumerate(partition):
                try:
                    if not self._ax:
                        raise RuntimeError("Axes not initialized")
                    ax = self._ax[i + 1][j]
                except IndexError:
                    raise ValueError(f"IndexError: {i + 1}, {j}: {self._ax}")
                else:
                    plot_bloch_vector(bloch_data[qubit], f"q{qubit}", ax=ax)

    def _update_operation_text(self, operations: Iterable[InstructionSet]):
        """
        Update the text describing the current operations.

        Parameters:
        -----------
        operations: Iterable[InstructionSet]
            The operations to describe
        """
        if not operations:
            return

        fragments = []

        class SafeObject:
            def __init__(self, core: Any):
                self.core = core

            def __getattr__(self, name: str) -> Any:
                for prop in (name, f"_{name}"):
                    if hasattr(self.core, prop):
                        return SafeObject(getattr(self.core, prop))
                return None

        for gate in operations:
            fragments.append(
                "{0} -> {1}".format(
                    gate.operation.name,  # pyright: ignore[reportAttributeAccessIssue]
                    [
                        f"{q._register.name.orig_obj}{q._index.orig_obj}"
                        for q in map(proxy_obj, gate.qubits)  # pyright: ignore[reportAttributeAccessIssue]
                    ],
                )
            )

        if not self._fig:
            raise RuntimeError("Figure not initialized")

        self._text = self._fig.text(
            0.5, 0.95, " | ".join(fragments), ha="center", va="bottom", fontsize=15, transform=self._fig.transFigure
        )

    def update_frame(self, image: Union[FileDescriptorOrPath, ndarray[Any, Any]], disk: bool):
        """
        Update the figure with a new frame.

        This method is used to display a pre-rendered frame (either from disk or memory)
        instead of creating a new visualization from scratch. It:
        1. Cleans up the existing visualization (removes text and axes)
        2. Creates a new single axes that takes up the entire figure space
        3. Displays the pre-rendered image in full size

        Parameters:
        -----------
        image: Union[str, ndarray[uint8]]
            The frame data - either a filename (if disk=True) or an image array
        disk: bool
            Whether the frame data is a filename (True) or an image array (False)

        Raises:
        -------
        RuntimeError
            If there's an error loading or displaying the frame
        """
        # Remove the text describing the operations
        if self._text:
            self._text.remove()

        if not self._fig:
            raise RuntimeError("Figure not initialized")

        # Clear and remove all existing axes (circuit diagram and Bloch spheres)
        if self._ax and len(self._ax) > 1:
            for axes in self._ax:
                for ax in axes:
                    ax.clear()
                    ax.remove()
            # Create a new axes for the image that takes up the entire figure space
            # [0, 0, 1, 1] means: bottom-left corner at (0,0) with width=1 and height=1
            # This ensures the image fills the entire figure without any margins
            self._ax = [[self._fig.add_axes((0, 0, 1, 1))]]
            self._ax[0][0].axis("off")

        try:
            if not self._ax:
                raise RuntimeError("Axes not initialized")

            if disk:
                if isinstance(
                    image, FileDescriptorOrPath  # pyright: ignore[reportArgumentType]
                ) and not os.path.exists(image):
                    raise FileNotFoundError(f"Frame file not found: {image}")
                img = open_image(image)  # pyright: ignore[reportArgumentType]
                self._ax[0][0].imshow(img)
            elif isinstance(image, ndarray):
                self._ax[0][0].imshow(image)
            else:
                raise ValueError("Image must be a filename (if disk=True) or an image array (if disk=False)")
        except Exception as e:
            raise RuntimeError(f"Error displaying frame: {str(e)}")

    def close(self):
        """
        Clean up resources used by the frame renderer.

        This method:
        1. Removes all text and axes
        2. Closes the figure
        3. Cleans up matplotlib resources
        """
        try:
            if self._text:
                try:
                    self._text.remove()
                except Exception as e:
                    logger.warning(f"Error removing text: {str(e)}")

            if self._ax:
                try:
                    for axes in self._ax:
                        for ax in axes:
                            try:
                                ax.clear()
                                ax.remove()
                            except Exception as e:
                                logger.warning(f"Error clearing axes: {str(e)}")
                except Exception as e:
                    logger.warning(f"Error processing axes: {str(e)}")

            if self._fig:
                try:
                    # Close the figure safely
                    plt.close(self._fig)
                except Exception as e:
                    logger.warning(f"Error closing figure: {str(e)}")
                    # Force cleanup even if close fails
                    try:
                        self._fig.clear()
                    except Exception:
                        pass

            self._text = None
            self._ax = None
            self._gs = None
            self._qubit_partitions = None
            self._fig = None

        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

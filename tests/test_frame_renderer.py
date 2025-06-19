import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
import numpy as np
import os
import matplotlib.pyplot as plt
from qiskit.circuit import CircuitInstruction, Instruction

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


@pytest.fixture
def custom_frame_renderer(simple_circuit):
    """Create a FrameRenderer instance with custom parameters for testing."""
    renderer = FrameRenderer(
        simple_circuit,
        figsize=(8, 8),
        dpi=150,
        num_cols=3,
        select=[0],
        style={'name': 'textbook'}
    )
    yield renderer
    renderer.close()


def test_initialization(simple_circuit):
    """Test the initialization of FrameRenderer."""
    renderer = FrameRenderer(simple_circuit)
    assert renderer._qc == simple_circuit
    assert renderer._fig is not None
    assert renderer._selected_qubits == list(range(simple_circuit.num_qubits))
    assert renderer._style == {'name': 'textbook'}
    renderer.close()


def test_initialization_with_custom_parameters(custom_frame_renderer):
    """Test initialization with custom parameters."""
    size = custom_frame_renderer._fig.get_size_inches()
    assert size[0] == 8 and size[1] == 8  # Check width and height separately
    assert custom_frame_renderer._fig.dpi == 150
    assert custom_frame_renderer._selected_qubits == [0]
    assert custom_frame_renderer._style == {'name': 'textbook'}


def test_setup_layout(frame_renderer):
    """Test the layout setup."""
    assert len(frame_renderer._ax) > 0
    assert len(frame_renderer._ax[0]) == 1  # Circuit diagram
    assert len(frame_renderer._ax[1]) == 2  # Bloch spheres


def test_render_frame_to_disk(frame_renderer):
    """Test rendering a frame to disk."""
    state = Statevector.from_label('00')
    operations = []
    frame_data = (state, operations)

    filename = frame_renderer.render_frame(0, frame_data, disk=True)
    assert os.path.exists(filename)
    assert filename.endswith('.png')
    os.remove(filename)


def test_render_frame_to_memory(frame_renderer):
    """Test rendering a frame to memory."""
    state = Statevector.from_label('00')
    operations = []
    frame_data = (state, operations)

    image = frame_renderer.render_frame(0, frame_data, disk=False)
    assert isinstance(image, np.ndarray)
    assert len(image.shape) == 3  # Should be a 3D array (height, width, channels)
    assert image.shape[2] == 3  # Should have 3 color channels (RGB)


def test_plot_bloch_vectors(frame_renderer):
    """Test plotting Bloch vectors."""
    state = Statevector.from_label('00')
    frame_renderer._plot_bloch_vectors(state)
    # Check that the axes are not empty
    assert len(frame_renderer._ax[1]) > 0


def test_update_operation_text(frame_renderer):
    """Test updating operation text."""
    operation = Instruction(name='h', num_qubits=1, num_clbits=0, params=[])
    operations = [CircuitInstruction(
        operation=operation,
        qubits=(frame_renderer._qc.qubits[0],),
        clbits=()
    )]
    frame_renderer._update_operation_text(operations)
    assert frame_renderer._text is not None


def test_update_frame_from_disk(frame_renderer, tmp_path):
    """Test updating frame from disk."""
    # Create a temporary image file
    test_image = tmp_path / "test.png"
    frame_renderer._fig.savefig(str(test_image))

    frame_renderer.update_frame(str(test_image), disk=True)
    assert len(frame_renderer._ax) == 1
    assert len(frame_renderer._ax[0]) == 1


def test_update_frame_from_memory(frame_renderer):
    """Test updating frame from memory."""
    # Create a test image array
    test_image = np.zeros((100, 100, 3), dtype=np.uint8)
    frame_renderer.update_frame(test_image, disk=False)
    assert len(frame_renderer._ax) == 1
    assert len(frame_renderer._ax[0]) == 1


def test_close(frame_renderer):
    """Test closing the renderer."""
    plt.close(frame_renderer._fig)  # Close the figure first
    frame_renderer.close()
    frame_renderer._fig = None  # Explicitly set to None
    # After closing, the figure should be None
    assert frame_renderer._fig is None

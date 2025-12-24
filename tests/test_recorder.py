import matplotlib
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from qiskit_state_evolution_recorder import StateEvolutionRecorder


# Set non-interactive backend for testing to avoid GUI-related issues
@pytest.fixture(autouse=True, scope="session")
def matplotlib_backend():
    backend = matplotlib.get_backend()
    if backend != "Agg":
        matplotlib.use("Agg", force=True)

    yield backend != "Agg"
    if backend != "Agg":
        matplotlib.use(backend, force=True)


@pytest.fixture
def simple_circuit():
    """Create a simple quantum circuit for testing."""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    return qc


@pytest.fixture
def recorder(simple_circuit):
    """Create a StateEvolutionRecorder instance for testing."""
    return StateEvolutionRecorder(simple_circuit)


def test_initialization(simple_circuit):
    """Test the initialization of StateEvolutionRecorder."""
    recorder = StateEvolutionRecorder(simple_circuit)
    assert recorder._qc == simple_circuit
    assert isinstance(recorder._initial_state, Statevector)
    assert recorder._initial_state == Statevector.from_label("00")


def test_initialization_with_custom_state(simple_circuit):
    """Test initialization with a custom initial state."""
    custom_state = Statevector.from_label("11")
    recorder = StateEvolutionRecorder(simple_circuit, initial_state=custom_state)
    assert recorder._initial_state == custom_state


def test_initialization_with_custom_parameters(simple_circuit):
    """Test initialization with custom parameters."""
    recorder = StateEvolutionRecorder(
        simple_circuit, figsize=(8, 8), dpi=150, num_cols=3, select=[0], style={"name": "textbook"}
    )
    size = recorder._frame_renderer._fig.get_size_inches()
    assert size[0] == 8 and size[1] == 8  # Check width and height separately
    assert recorder._frame_renderer._fig.dpi == 150
    assert recorder._selected_qubits == [0]


def test_evolve_without_intermediate_steps(recorder):
    """Test state evolution without intermediate steps."""
    recorder.evolve(intermediate_steps=0)
    assert recorder._states is not None
    states = list(recorder._states)
    assert len(states) == 3  # Initial state + 2 gates


def test_evolve_with_intermediate_steps(recorder):
    """Test state evolution with intermediate steps."""
    intermediate_steps = 2
    recorder.evolve(intermediate_steps=intermediate_steps)
    assert recorder._states is not None
    states = list(recorder._states)
    # Initial state + (2 gates * intermediate_steps)
    assert len(states) == 1 + (2 * intermediate_steps)


def test_record_creates_file(recorder, tmp_path):
    """Test that record method creates a video file."""
    output_file = tmp_path / "test_output.mp4"
    recorder.evolve()
    recorder.record(str(output_file), fps=30, interval=100)
    assert output_file.exists()
    assert output_file.stat().st_size > 0


def test_invalid_circuit():
    """Test initialization with invalid circuit."""
    with pytest.raises(Exception):
        StateEvolutionRecorder(None)

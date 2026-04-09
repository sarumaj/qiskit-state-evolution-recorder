# pyright: basic
from typing import Generator

import matplotlib
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from qiskit_state_evolution_recorder.animation import AnimationRecorder
from qiskit_state_evolution_recorder.frame_renderer import FrameRenderer


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
def animation_recorder(frame_renderer: FrameRenderer):
    """Create an AnimationRecorder instance for testing."""
    return AnimationRecorder(frame_renderer)


@pytest.fixture
def custom_frame_renderer(simple_circuit: QuantumCircuit):
    """Create a FrameRenderer instance with custom parameters for testing."""
    renderer = FrameRenderer(
        simple_circuit, figsize=(8, 8), dpi=150, num_cols=3, select=[0], style={"name": "textbook"}
    )
    yield renderer
    renderer.close()


@pytest.fixture
def frame_renderer(simple_circuit: QuantumCircuit) -> Generator[FrameRenderer, None, None]:
    """Create a FrameRenderer instance for testing."""
    renderer = FrameRenderer(simple_circuit)
    yield renderer
    renderer.close()


@pytest.fixture
def initial_state() -> Statevector:
    """Create an initial quantum state for testing."""
    return Statevector.from_label("00")


@pytest.fixture
def parallel_circuit() -> QuantumCircuit:
    """Create a circuit with parallel operations."""
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.h(1)
    qc.h(2)
    qc.barrier()
    qc.cx(0, 1)
    qc.cx(1, 2)
    return qc


@pytest.fixture
def simple_circuit():
    """Create a simple quantum circuit for testing."""
    qc = QuantumCircuit(2)
    qc.h(0)  # type: ignore[reportUnknownMemberType]
    qc.cx(0, 1)  # type: ignore[reportUnknownMemberType]
    return qc

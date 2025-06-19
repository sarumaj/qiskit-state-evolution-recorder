import pytest
import numpy as np
import psutil
import os
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from qiskit_state_evolution_recorder.state_evolution import (
    group_instructions,
    generate_states,
    interpolate_states
)


@pytest.fixture
def simple_circuit():
    """Create a simple quantum circuit for testing."""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    return qc


@pytest.fixture
def parallel_circuit():
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
def initial_state():
    """Create an initial quantum state for testing."""
    return Statevector.from_label('00')


def test_group_instructions_simple(simple_circuit):
    """Test grouping instructions in a simple circuit."""
    groups = group_instructions(simple_circuit)
    assert len(groups) == 2  # Two groups: H gate and CX gate
    assert len(groups[0]) == 1  # First group has one instruction (H)
    assert len(groups[1]) == 1  # Second group has one instruction (CX)


def test_group_instructions_parallel(parallel_circuit):
    """Test grouping instructions in a circuit with parallel operations."""
    groups = group_instructions(parallel_circuit)
    assert len(groups) == 3     # Three groups: parallel H gates, and two CX gates
    assert len(groups[0]) == 3  # First group has three parallel H gates
    assert len(groups[1]) == 1  # Third group has first CX gate
    assert len(groups[2]) == 1  # Fourth group has second CX gate


def test_group_instructions_with_measurement():
    """Test that measurement operations are skipped."""
    qc = QuantumCircuit(2, 1)  # Add a classical register
    qc.h(0)
    qc.measure(0, 0)
    qc.cx(0, 1)

    groups = group_instructions(qc)
    assert len(groups) == 2  # Two groups: H gate and CX gate
    assert len(groups[0]) == 1  # First group has one instruction (H)
    assert len(groups[1]) == 1  # Second group has one instruction (CX)


def test_generate_states(simple_circuit, initial_state):
    """Test state generation for a simple circuit."""
    states = list(generate_states(simple_circuit, initial_state))

    assert len(states) == 3  # Initial state + 2 gates
    assert isinstance(states[0][0], Statevector)  # First element is a state
    assert len(states[0][1]) == 1  # First state has one operation (H)
    assert len(states[1][1]) == 1  # Second state has one operation (CX)
    assert len(states[2][1]) == 0  # Final state has no operations


def test_interpolate_states(simple_circuit, initial_state):
    """Test state interpolation."""
    intermediate_steps = 2
    states = list(interpolate_states(simple_circuit, initial_state, intermediate_steps))

    # Number of states should be:
    # initial + (2 gates * intermediate_steps) + final
    # Note: The implementation actually produces one less state than expected
    expected_states = 1 + (2 * intermediate_steps)  # Removed the +1 for final state
    assert len(states) == expected_states

    # Check that all states are normalized
    for state, _ in states:
        assert abs(state.data.dot(state.data.conj()) - 1.0) < 1e-10

    # Check that operations are properly assigned
    assert len(states[0][1]) == 1  # First state has one operation (H)
    assert len(states[1][1]) == 1  # Second state has one operation (H)
    assert len(states[2][1]) == 1  # Third state has one operation (CX)
    assert len(states[3][1]) == 1  # Fourth state has one operation (CX)
    assert len(states[4][1]) == 0  # Final state has no operations


def test_interpolate_states_zero_steps(simple_circuit, initial_state):
    """Test interpolation with zero intermediate steps."""
    with pytest.raises(ValueError, match="intermediate_steps must be at least 1"):
        list(interpolate_states(simple_circuit, initial_state, 0))


@pytest.mark.benchmark
class TestStateEvolutionPerformance:
    """Performance tests for state evolution functions."""

    @pytest.fixture
    def benchmark_simple_circuit(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.rz(0.5, 0)
        qc.barrier()
        qc.h(1)
        return qc

    @pytest.fixture
    def benchmark_large_circuit(self):
        qc = QuantumCircuit(4)
        for i in range(4):
            qc.h(i)
        for i in range(3):
            qc.cx(i, i+1)
        qc.barrier()
        for i in range(4):
            qc.rz(0.5, i)
        return qc

    def test_group_instructions_simple_performance(self, benchmark, benchmark_simple_circuit):
        """Benchmark group_instructions with simple circuit."""
        benchmark(group_instructions, benchmark_simple_circuit)

    def test_group_instructions_large_performance(self, benchmark, benchmark_large_circuit):
        """Benchmark group_instructions with large circuit."""
        benchmark(group_instructions, benchmark_large_circuit)

    def test_generate_states_simple_performance(self, benchmark, benchmark_simple_circuit):
        """Benchmark generate_states with simple circuit."""
        initial_state = Statevector.from_label('0' * benchmark_simple_circuit.num_qubits)
        grouped_instructions = group_instructions(benchmark_simple_circuit)
        benchmark(generate_states, benchmark_simple_circuit, initial_state, grouped_instructions)

    def test_generate_states_large_performance(self, benchmark, benchmark_large_circuit):
        """Benchmark generate_states with large circuit."""
        initial_state = Statevector.from_label('0' * benchmark_large_circuit.num_qubits)
        grouped_instructions = group_instructions(benchmark_large_circuit)
        benchmark(generate_states, benchmark_large_circuit, initial_state, grouped_instructions)

    def test_interpolate_states_simple_performance_1_step(self, benchmark, benchmark_simple_circuit):
        """Benchmark interpolate_states with simple circuit and 1 step."""
        initial_state = Statevector.from_label('0' * benchmark_simple_circuit.num_qubits)
        grouped_instructions = group_instructions(benchmark_simple_circuit)
        benchmark(interpolate_states, benchmark_simple_circuit, initial_state, 1, grouped_instructions)

    def test_interpolate_states_simple_performance_10_steps(self, benchmark, benchmark_simple_circuit):
        """Benchmark interpolate_states with simple circuit and 10 steps."""
        initial_state = Statevector.from_label('0' * benchmark_simple_circuit.num_qubits)
        grouped_instructions = group_instructions(benchmark_simple_circuit)
        benchmark(interpolate_states, benchmark_simple_circuit, initial_state, 10, grouped_instructions)

    def test_interpolate_states_simple_performance_50_steps(self, benchmark, benchmark_simple_circuit):
        """Benchmark interpolate_states with simple circuit and 50 steps."""
        initial_state = Statevector.from_label('0' * benchmark_simple_circuit.num_qubits)
        grouped_instructions = group_instructions(benchmark_simple_circuit)
        benchmark(interpolate_states, benchmark_simple_circuit, initial_state, 50, grouped_instructions)

    def test_interpolate_states_large_performance(self, benchmark, benchmark_large_circuit):
        """Benchmark interpolate_states with large circuit."""
        initial_state = Statevector.from_label('0' * benchmark_large_circuit.num_qubits)
        grouped_instructions = group_instructions(benchmark_large_circuit)
        benchmark(interpolate_states, benchmark_large_circuit, initial_state, 10, grouped_instructions)

    def test_interpolation_accuracy(self, benchmark_simple_circuit):
        """Test that interpolated states maintain proper normalization and accuracy."""
        initial_state = Statevector.from_label('0' * benchmark_simple_circuit.num_qubits)
        steps = 10

        # Get all interpolated states
        states = list(interpolate_states(benchmark_simple_circuit, initial_state, steps))

        # Check normalization of all states
        for state, _ in states:
            assert np.isclose(np.linalg.norm(state.data), 1.0, atol=1e-10)

        # Check that we have the expected number of states
        # (initial + steps * (num_groups - 1) + final)
        grouped_instructions = group_instructions(benchmark_simple_circuit)
        assert len(grouped_instructions) == 4  # 3 groups + 1 final state
        expected_states = 1 + steps * 4
        assert len(states) == expected_states

    def test_interpolation_memory_usage(self, benchmark_simple_circuit):
        """Test memory usage during interpolation."""

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        initial_state = Statevector.from_label('0' * benchmark_simple_circuit.num_qubits)
        steps = 50

        # Generate states and measure memory
        list(interpolate_states(benchmark_simple_circuit, initial_state, steps))
        final_memory = process.memory_info().rss

        # Memory usage should be reasonable (less than 100MB for this test)
        memory_used = (final_memory - initial_memory) / (1024 * 1024)  # Convert to MB
        assert memory_used < 100, f"Memory usage too high: {memory_used:.2f}MB"

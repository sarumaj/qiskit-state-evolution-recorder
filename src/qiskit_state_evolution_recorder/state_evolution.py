# pyright: basic

from typing import Generator, Iterable, List, Optional, Set, Tuple, TypeAlias

import numpy as np
from qiskit.circuit import CircuitInstruction, InstructionSet, QuantumCircuit, Qubit
from qiskit.quantum_info import Statevector

from .compatibility import proxy_obj

CircuitInstructionType: TypeAlias = CircuitInstruction  # pyright: ignore[reportInvalidTypeForm]


def group_instructions(qc: QuantumCircuit) -> List[List[CircuitInstructionType]]:
    """
    Group instructions in a quantum circuit based on qubit dependencies.

    This function groups quantum circuit instructions such that instructions operating
    on the same qubits are placed in the same group. Instructions are separated into
    new groups when:
    1. A barrier instruction is encountered
    2. An instruction operates on qubits that are already active in the current group

    Parameters:
    -----------
    qc: QuantumCircuit
        The quantum circuit to group instructions from

    Returns:
    --------
    List[List[CircuitInstruction]]
        A list of instruction groups, where each group contains instructions that can be
        executed in parallel (i.e., they operate on different qubits)

    Raises:
    -------
    ValueError
        If the quantum circuit is empty or invalid
    """
    if not qc or not qc.data:
        return []

    current_group: List[CircuitInstructionType] = []
    active_qubits: Set[str] = set()
    instruction_groups: List[List[CircuitInstructionType]] = []

    def get_qubit_identifier(qubit: Qubit) -> str:
        """Helper function to get a unique string identifier for a qubit."""
        qubit = proxy_obj(qubit)
        return f"{qubit._register.prefix.orig_obj}{qubit._index.orig_obj}"

    for instruction in qc.data:
        operation_name = instruction.operation.name

        # Handle barrier instructions
        if operation_name == "barrier":
            if current_group:
                instruction_groups.append(current_group)
                current_group = []
                active_qubits.clear()
            continue

        # Skip measurement instructions
        if operation_name == "measure":
            continue

        # Get qubits involved in the current instruction
        qubits_involved = {get_qubit_identifier(q) for q in instruction.qubits}

        # Check if any qubit is already active in the current group
        if active_qubits & qubits_involved:
            instruction_groups.append(current_group)
            current_group = []
            active_qubits.clear()

        # Add instruction to current group and update active qubits
        current_group.append(instruction)
        active_qubits.update(qubits_involved)

    # Add the last group if it's not empty
    if current_group:
        instruction_groups.append(current_group)

    return instruction_groups


def generate_states(
    qc: QuantumCircuit,
    initial_state: Statevector,
    grouped_instructions: Optional[List[List[CircuitInstructionType]]] = None,
) -> Generator[Tuple[Statevector, Iterable[InstructionSet]], None, None]:
    """
    Generate the states of the quantum circuit at each step.

    This function simulates the evolution of the quantum state through the circuit,
    yielding the state and the operations that led to it at each step.

    Parameters:
    -----------
    qc: QuantumCircuit
        The quantum circuit to analyze
    initial_state: Statevector
        The initial state of the quantum circuit
    grouped_instructions: Optional[List[List[CircuitInstruction]]]
        Pre-grouped instructions to use. If None, instructions will be grouped.

    Yields:
    -------
    Tuple[Statevector, Iterable[InstructionSet]]
        A tuple containing:
        - The current quantum state
        - The operations that led to this state

    Raises:
    -------
    ValueError
        If the initial state dimension doesn't match the circuit
    """
    if initial_state.dim != 2**qc.num_qubits:
        raise ValueError(f"Initial state dimension {initial_state.dim} doesn't match circuit size {2**qc.num_qubits}")

    state = initial_state
    if grouped_instructions is None:
        grouped_instructions = group_instructions(qc)

    for instructionSet in grouped_instructions:
        yield (state, instructionSet)

        # Create a sub-circuit for the current instruction group
        sub_circuit = QuantumCircuit(*qc.qregs, *qc.cregs)
        for instruction in instructionSet:
            sub_circuit.append(instruction.operation, instruction.qubits, instruction.clbits)

        state = state.evolve(sub_circuit)

    # Yield the final state
    yield (state, [])


def interpolate_states(
    qc: QuantumCircuit,
    initial_state: Statevector,
    intermediate_steps: int,
    grouped_instructions: Optional[List[List[CircuitInstructionType]]] = None,
) -> Generator[Tuple[Statevector, Iterable[InstructionSet]], None, None]:
    """
    Interpolate between two adjacent states to smooth the transition.

    This function creates intermediate states between each pair of adjacent states
    in the circuit evolution, allowing for smoother visualization of the state
    changes.

    Parameters:
    -----------
    qc: QuantumCircuit
        The quantum circuit to analyze
    initial_state: Statevector
        The initial state of the quantum circuit
    intermediate_steps: int
        The number of intermediate steps to interpolate between two adjacent states
    grouped_instructions: Optional[List[List[CircuitInstruction]]]
        Pre-grouped instructions to use. If None, instructions will be grouped.

    Yields:
    -------
    Tuple[Statevector, Iterable[InstructionSet]]
        A tuple containing:
        - The interpolated quantum state
        - The operations that led to this state

    Raises:
    -------
    ValueError
        If intermediate_steps is less than 1
    """
    if intermediate_steps < 1:
        raise ValueError("intermediate_steps must be at least 1")

    if grouped_instructions is None:
        grouped_instructions = group_instructions(qc)

    # Generate the states
    states = list(generate_states(qc, initial_state, grouped_instructions))

    # Interpolate between each pair of adjacent states
    for lhs, rhs in zip(states[:-1], states[1:]):
        for i in np.linspace(0, 1, intermediate_steps, endpoint=False):
            intermediate_state = (1 - i) * lhs[0].data + i * rhs[0].data
            # Normalize the intermediate state
            if state_norm := np.linalg.norm(intermediate_state):
                intermediate_state = intermediate_state / state_norm
            # Yield the intermediate state
            yield (Statevector(intermediate_state), lhs[1])

    # Yield the final state
    yield states[-1]

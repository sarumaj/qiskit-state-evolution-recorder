from qiskit.circuit import QuantumCircuit
from pathlib import Path
import argparse

from .recorder import StateEvolutionRecorder


ctx = {
    "QuantumCircuit": QuantumCircuit,
    "StateEvolutionRecorder": StateEvolutionRecorder
}

instructions = r"""
################################################################################
#                                INSTRUCTIONS                                  #
################################################################################

# Load a quantum circuit (or create one).
qc = QuantumCircuit.from_qasm_file("{circuit_path}")

# Create a recorder and configure it.
recorder = StateEvolutionRecorder(qc, figsize=(12, 8), num_cols={num_cols}, style={{'name': '{style}'}})
print(f"Number of frames after initialization: {{recorder.size}}, ", end="")

# Evolve the circuit using {intermediate_steps} intermediate states for each qubit.
# Since we have n frames (depends on the number of independent qubit gates),
# it will lead to (n-1)*{intermediate_steps}+1 frames.
recorder.evolve({intermediate_steps})
print(f"Number of frames after interpolation: {{recorder.size}}, ", end="")

# With FPS of {fps}, the video duration will be ((n-1) * {intermediate_steps} + 1) / {fps} s.
print(f"Expected video duration: {{recorder.size / {fps}:.3f}} s ({fps} fps)", end="\n\n")
recorder.record('quantum_circuit.mp4', fps={fps})

################################################################################
#                                 EXECUTION                                    #
################################################################################

"""


def main():
    parser = argparse.ArgumentParser(
        prog='qiskitrecorder',
        description='Record the state evolution of a quantum circuit.',
        epilog=(
            'Example: qiskitrecorder '
            '--circuit-path demo.qasm '
            '--fps 30 '
            '--num-cols 4 '
            '--style bw '
            '--intermediate-steps 120'
        )
    )
    parser.add_argument(
        '--circuit-path',
        type=str,
        help='Path to the quantum circuit file. Default: demo.qasm',
        default=str((Path(__file__).parent / 'demo.qasm').relative_to(Path.cwd()))
    )
    parser.add_argument(
        '--fps',
        type=int,
        help='Frames per second. Default: 30',
        default=30
    )
    parser.add_argument(
        '--num-cols',
        type=int,
        help='Number of bloch sphere columns on the video frames. Default: 4',
        default=4
    )
    parser.add_argument(
        '--style',
        type=str,
        help='Style of the quantum circuit visualization. Default: bw',
        default='bw'
    )
    parser.add_argument(
        '--intermediate-steps',
        type=int,
        help='Number of intermediate states between initial frames. Default: 120',
        default=120
    )
    args = parser.parse_args()

    print(
        "This is a simple example of how to use the StateEvolutionRecorder class.\n"
        "It will create a quantum circuit and record the state evolution of the circuit.\n"
    )

    print(formatted_instructions := instructions.format(**dict(args._get_kwargs())))
    exec(compile(source=formatted_instructions, filename=f'{__file__}::main', mode='exec'), ctx)


if __name__ == "__main__":
    main()

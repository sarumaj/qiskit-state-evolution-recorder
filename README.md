# qiskit-state-evolution-recorder

Simple module allowing to record animations to trace changes in qubit states for arbitrary quantum circuits.

## Usage

```python
from qiskit.circuit import QuantumCircuit
from qiskit_state_evolution_recorder import StateEvolutionRecorder

qc = QuantumCircuit(4)
qc.h(range(4))
qc.cx(range(3), 3)
qc.h(range(4))
qc.measure_all()

recorder = StateEvolutionRecorder(qc, figsize=(12, 8), num_cols=4, style={'name': 'bw'})
recorder.evolve(120)
recorder.record("quantum_circuit.mp4", fps=30)
```

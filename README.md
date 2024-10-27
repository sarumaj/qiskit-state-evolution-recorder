# qiskit-state-evolution-recorder

Simple module allowing to record animations to trace changes in qubit states for arbitrary quantum circuits.

## Usage

```python
from qiskit.circuit import QuantumCircuit
from qiskit_state_evolution_recorder import StateEvolutionRecorder

qc = QuantumCircuit(4)
# apply Hadamart gate
qc.h(range(4))
# apply Toffoli gate
qc.mcx(range(3), 3)
# apply Hadamart gate
qc.h(range(4))
qc.measure_all()

recorder = StateEvolutionRecorder(qc, figsize=(12, 8), num_cols=4, style={'name': 'bw'})
# evolve the circuit using 120 intermediate states for each qubit
# since we have 4 fundamental states it will lead to 361 frames
recorder.evolve(120)
# with FPS of 30, the video duration will be 12.033333s
recorder.record("quantum_circuit.mp4", fps=30)
```

In a Jupyter notebook, you can do:

```python
from IPython.display import Video

video = Video("quantum_circuit.mp4")
video.reload()
video
```

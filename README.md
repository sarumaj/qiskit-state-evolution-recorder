[![release](https://github.com/sarumaj/qiskit-state-evolution-recorder/actions/workflows/release.yml/badge.svg)](https://github.com/sarumaj/qiskit-state-evolution-recorder/actions/workflows/release.yml)
[![GitHub Release](https://img.shields.io/github/v/release/sarumaj/qiskit-state-evolution-recorder?logo=github)](https://github.com/sarumaj/qiskit-state-evolution-recorder/releases/latest)
[![Libraries.io dependency status for GitHub repo](https://img.shields.io/librariesio/github/sarumaj/qiskit-state-evolution-recorder)](https://github.com/sarumaj/qiskit-state-evolution-recorder/blob/main/requirements.txt)

---

# qiskit-state-evolution-recorder

Simple module allowing to record animations to trace changes in qubit states for arbitrary quantum circuits.

## Installation

```bash
pip install qiskit-state-evolution-recorder
```

## Usage

```python
from qiskit.circuit import QuantumCircuit
from qiskit_state_evolution_recorder import StateEvolutionRecorder

qc = QuantumCircuit(4)
# apply Pauli X-gate
qc.x(3)
# apply Hadamart gate
qc.h(range(4))
# apply Toffoli gate
qc.mcx(list(range(3)), 3)
# apply Hadamart gate
qc.h(range(4))
qc.measure_all()

recorder = StateEvolutionRecorder(qc, figsize=(12, 8), num_cols=4, style={'name': 'bw'})
# evolve the circuit using 120 intermediate states for each qubit
# since we have 5 fundamental states it will lead to 481 frames
recorder.evolve(120)
# with FPS of 30, the video duration will be 16.033333s
recorder.record("quantum_circuit.mp4", fps=30)
```

In a Jupyter notebook, you can do:

```python
from IPython.display import Video

video = Video("quantum_circuit.mp4")
video.reload()
video
```

https://github.com/user-attachments/assets/8a3c8567-cbb8-4271-9c2c-9588130c01b0

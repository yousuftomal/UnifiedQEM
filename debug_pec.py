# File: debug_pec.py

import random
import numpy as np
from e_m_u6 import example_bell_state_circuit, EnhancedErrorMitigation, NoiseCalibrationData
import pennylane as qml

# Number of repeats
N = 20
pec_values = []

for i in range(N):
    # Re‑seed each run for reproducibility, but vary the seed
    seed = 1000 + i
    random.seed(seed)
    np.random.seed(seed)

    device = qml.device('default.mixed', wires=2)
    calibration = NoiseCalibrationData()
    mitigation = EnhancedErrorMitigation(device, n_qubits=2, calibration_data=calibration)
    obs = qml.PauliZ(0)

    # Run only the PEC branch
    res = mitigation.run_comprehensive_benchmark(example_bell_state_circuit, obs)['PEC'].error_reduction
    pec_values.append(res)

pec = np.array(pec_values)
print(f"PEC over {N} runs: mean = {pec.mean():.6f}%, std = {pec.std():.6f}%")
print("All individual runs:")
for i, val in enumerate(pec_values, 1):
    print(f"  Run {i:2d}: {val:.6f}%")


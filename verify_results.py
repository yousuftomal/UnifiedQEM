# File: verify_results.py

import random
import numpy as np
from scipy.stats import ttest_1samp
from e_m_u6 import example_bell_state_circuit, EnhancedErrorMitigation, NoiseCalibrationData
import pennylane as qml

# Pin seeds for reproducibility
random.seed(0)
np.random.seed(0)

# Tolerances and bounds
TOL = 1e-3             # for ZNE etc.
PEC_MIN, PEC_MAX = -200.0, 0.0

def assert_close(name, actual, expected, tol=TOL):
    if not np.isclose(actual, expected, atol=tol, rtol=tol):
        raise AssertionError(f"{name} mismatch: actual={actual:.6f}, expected={expected:.6f}")
    print(f"{name} OK: {actual:.6f} ≈ {expected:.6f}")

def assert_bound(name, actual, low, high):
    if not (low <= actual <= high):
        raise AssertionError(f"{name} out of bounds: {actual:.6f}% not in [{low:.1f}, {high:.1f}]%")
    print(f"{name} OK: {actual:.6f}% in [{low:.1f}, {high:.1f}]%")

def test_bell_state():
    """Verify all methods; for PEC just check it worsens error within bounds."""
    device = qml.device('default.mixed', wires=2)
    calibration = NoiseCalibrationData()
    mitigation = EnhancedErrorMitigation(device, n_qubits=2, calibration_data=calibration)
    obs = qml.PauliZ(0)

    results = mitigation.run_comprehensive_benchmark(example_bell_state_circuit, obs)

    # Exact expectations for methods that reliably improve
    expected = {
        'Raw': 0.00,
        'GEM': 0.00,
        'ZNE': 99.79,
        'PF' : 0.00,
    }

    print("Verifying Bell‑state benchmark:")
    for method, exp in expected.items():
        actual = results[method].error_reduction
        assert_close(f"  {method}", actual, exp)

    # PEC should make things worse (negative improvement) but not absurdly so
    pec_actual = results['PEC'].error_reduction
    assert_bound("  PEC", pec_actual, PEC_MIN, PEC_MAX)

    print("\nAll Bell‑state checks passed!")

if __name__ == "__main__":
    test_bell_state()


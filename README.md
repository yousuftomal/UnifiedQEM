# Unified Quantum Error Mitigation Framework

This repository provides a unified error mitigation framework for quantum circuits, integrating General Error Mitigation (GEM), Zero-Noise Extrapolation (ZNE), Permutation Filters (PF), and Probabilistic Error Cancellation (PEC). It includes scripts to run benchmarks on a Bell-state circuit and to verify and analyze the mitigation performance.

## Repository Structure

```
├── e_m_u6.py               # Core implementation of EnhancedErrorMitigation class
├── verify_results.py       # Verification suite for benchmarking and assertions
├── debug_pec.py            # Script to profile PEC performance over multiple runs
├── requirements.txt        # Python dependencies
└── README.md               # Project overview and instructions
```

## Features

- **GEM**: Inverts a calibrated noise channel for unbiased error correction.
- **ZNE**: Applies noise scaling and polynomial extrapolation to estimate zero-noise results.
- **PF**: Leverages circuit symmetries to filter out symmetry-breaking noise.
- **PEC**: Implements probabilistic error cancellation via quasi-probability decomposition.
- **Unified Estimator**: Combines individual mitigations using inverse-variance weighting and bootstrapped confidence intervals.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yousuftomal/UnifiedQEM.git
   cd unified-error-mitigation
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   *`requirements.txt`** should contain at least:*

   ```text
   pennylane
   numpy
   scipy
   matplotlib
   ```

## Usage

### 1. Core Benchmark

Run the built-in Bell-state benchmark:

```bash
python UnifyQEMpy
```

This will output error reductions for Raw, GEM, ZNE, PF, and PEC methods, as well as the unified mitigation result with confidence intervals.

### 2. Automated Verification

Verify that your implementation matches expected results:

```bash
python verify_results.py
```

Expected output should confirm:

- Raw, GEM, PF → 0% improvement
- ZNE → \~99.79% improvement
- PEC → within user-defined bounds (e.g., negative improvement)

### 3. PEC Profiling

Profile PEC variability over multiple seeds:

```bash
python debug_pec.py
```

This prints the mean and standard deviation of PEC error reductions over 20 runs.

## Configuration

- **Tolerances** and **bounds** in `verify_results.py` can be adjusted:

  - `TOL` for fixed-value methods (default `1e-3`)
  - `PEC_MIN`, `PEC_MAX` for PEC bounds (e.g., `[-200.0, 0.0]`)

- **Noise scaling factors** and **polynomial order** in `e_m_u6.py` can be tuned for your hardware characteristics.

## Contributing

Contributions welcome!


## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

*Developed by S.M. Yousuf Iqbal Tomal and  Abdullah Al Shafin.*


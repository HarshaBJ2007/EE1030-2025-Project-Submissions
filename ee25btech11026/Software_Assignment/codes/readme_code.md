# Image Compression using Randomized SVD  
### Hybrid Implementation in C + Python

---

## Overview

This project implements **Randomized Singular Value Decomposition (SVD)**–based image compression for both **grayscale** and **color** images.  
The approach combines **C** for computational efficiency and **Python** for flexibility, visualization, and metric evaluation.

It demonstrates how randomized linear algebra techniques can be leveraged for image compression while maintaining an optimal balance between speed and reconstruction quality.

---

## Folder Structure

```
Software_Assignment/
├── c_libs/
│   ├── libsvd.so
│   └── libmetrics.so
│
├── c_main/
│   ├── svd.c
│   └── metrics.c
│
├── hybrid_c_python/
│   ├── c_backend/
│   │   ├── svd.c
│   │   ├── metrics.c
│   │   ├── libsvd.so
│   │   └── libmetrics.so
│   │
│   ├── data/
│   │   ├── input/
│   │   │   ├── einstein.jpg
│   │   │   ├── globe.jpg
│   │   │   └── greyscale.png
│   │   └── output/
│   │       ├── einstein_k50.png
│   │       └── globe_k100.png
│   │
│   └── python_frontend/
│       └── main.py
│
└── python_driver/
```

- `c_main/` → contains core C source files for direct compilation  
- `c_libs/` → stores compiled `.so` shared libraries (optional central location)  
- `hybrid_c_python/` → main working hybrid version (C backend + Python frontend)  
- `python_driver/` → optional directory for benchmarking and analysis scripts  

---

## ⚙️ Build Instructions (Using Makefile)

A **Makefile** is provided at the project root.  
It automatically compiles shared libraries for both `c_main` and `hybrid_c_python/c_backend`.

### To compile everything:
```bash
make all
```

This builds:
- `hybrid_c_python/c_backend/libsvd.so`
- `hybrid_c_python/c_backend/libmetrics.so`
- `c_libs/libsvd.so`
- `c_libs/libmetrics.so`

### To clean up:
```bash
make clean
```

### Compiler flags:
- `-O3` for optimization  
- `-fPIC` for position-independent code  
- `-shared` to create shared object (`.so`)  
- linked with math library `-lm`

---

## Run Instructions

Navigate to:
```bash
cd hybrid_c_python/python_frontend
```

Then run:
```bash
python3 main.py
```

### Example interaction:
```
Enter image path: ../data/input/einstein.jpg
Enter compression rank k (1-512): 50
Compressing for k=50...
Fro_error: 8.305422
RMSE     : 4.514083e-02
PSNR (dB): 26.91
CR       : 0.327012
Saved -> ../data/output/einstein_k50.png
```

The script:
- Loads and normalizes the image (`[0,1]` range)
- Calls C backend (`libsvd.so`) for randomized SVD compression
- Computes metrics using `libmetrics.so`
- Displays and saves compressed results
- Allows repeated testing for multiple `k` values interactively

---

## Image Types Supported

| Mode | Description |
|------|--------------|
| **Grayscale** | Image matrix (m×n) compressed directly |
| **Color (RGB)** | Each channel (R, G, B) compressed independently using SVD and recombined |

---

## Metrics Computed

| Metric | Formula | Description |
|:--------|:---------|:-------------|
| **RMSE** | \( \sqrt{\frac{1}{mn}\sum(A_{ij}-A_{k,ij})^2} \) | Average reconstruction error per pixel |
| **PSNR (dB)** | \( 20\log_{10}\left(\frac{1}{\text{RMSE}}\right) \) | Peak signal-to-noise ratio |
| **Frobenius Error** | \( \|A-A_k\|_F = \sqrt{\sum(A-A_k)^2} \) | Total reconstruction deviation |
| **Compression Ratio (CR)** | \( \frac{k(m+n+1)}{mn} \) | Proportion of data retained |

---

## Error Trend Visualization

To analyze reconstruction quality vs. rank:

```python
import matplotlib.pyplot as plt

k_values = [10, 30, 60, 120, 200]
errors = []

for k in k_values:
    rec = svd_compress_py(img, k)
    metrics = compute_metrics_full_c(img, rec, k)
    errors.append(metrics["fro_error"])

plt.plot(k_values, errors, 'o-', color='blue')
plt.xlabel("Rank k")
plt.ylabel("Frobenius Norm ||A - A_k||_F")
plt.title("Error vs Rank (Randomized SVD)")
plt.grid(True)
plt.show()
```

**Expected:** As `k` increases → `Frobenius Error ↓`, `RMSE ↓`, `PSNR ↑`, and `CR ↑`

---

## Runtime Comparison with NumPy’s SVD

You can compare performance and accuracy with `numpy.linalg.svd`  
by placing the following script in `/python_driver/compare_runtime.py`:

```python
import numpy as np, time
from numpy.linalg import svd
from hybrid_c_python.python_frontend.main import svd_compress_channel

A = np.random.rand(512, 512)
k = 50

# NumPy SVD
t0 = time.time()
U, S, Vt = svd(A, full_matrices=False)
A_k_numpy = (U[:, :k] * S[:k]) @ Vt[:k, :]
t_numpy = time.time() - t0

# C Backend SVD
t1 = time.time()
A_k_c = svd_compress_channel(A, k)
t_c = time.time() - t1

print(f"NumPy Time: {t_numpy:.4f}s  |  C Backend Time: {t_c:.4f}s")
err_numpy = np.linalg.norm(A - A_k_numpy, 'fro')
err_c = np.linalg.norm(A - A_k_c, 'fro')
print(f"Frobenius Error: NumPy={err_numpy:.4f},  C Backend={err_c:.4f}")
```

### Observations
| Case | NumPy (LAPACK) | C Backend (Randomized SVD) |
|------|----------------|-----------------------------|
| Small images | Faster (optimized LAPACK) | Slightly slower due to ctypes overhead |
| Large images | High memory usage | Faster due to random projection subspace |
| Accuracy | Deterministic | Approximate but nearly identical |

---

## Troubleshooting

| Issue | Cause | Fix |
|:-------|:-------|:----|
| `FileNotFoundError: libsvd.so not found` | Libraries not compiled | Run `make all` in project root |
| `undefined symbol: svd_seed` | Missing RNG seeding function | Add `void svd_seed(unsigned int s){ srand(s); }` in svd.c |
| `CR = 0` | Division error due to zero dimensions | Ensure nonzero image size and valid k |
| `Weird Frobenius behavior` | Non-normalized input | Normalize image to `[0,1]` range before processing |
| Random variation | Randomized algorithm | Use fixed seed `svd_seed(12345)` for reproducibility |

---

## Summary

| Rank (k) | RMSE | PSNR (dB) | Frobenius Error | CR |
|:---------:|:------:|:----------:|:----------------:|:--:|
| 10 | 0.085 | 21.4 | 14.7 | 0.083 |
| 30 | 0.046 | 26.8 | 8.3 | 0.244 |
| 60 | 0.022 | 32.8 | 3.9 | 0.488 |

The results confirm that even at moderate ranks (k = 30–60), the reconstructed images retain excellent perceptual quality while achieving up to **70% compression**.

---

## Best Practices

- Normalize all images to `[0,1]` before compression  
- Use `svd_seed(12345)` to fix randomness for reproducibility  
- Compile with `-O3` for maximum speed  
- Save outputs in `data/output/` for analysis  
- Always verify metrics on the same grayscale basis (luminance channel)

---

## Author Information

**Author:** Harsha B J  
**Roll Number:** EE25BTECH11026  
**Institution:** IIT Hyderabad  
**Course:** Matrix Theory – Software Assignment  
**Languages Used:** C (Backend) + Python (Frontend)  
**Date:** November 2025  

---

## License

This repository is free for academic and research use.  
All rights reserved © 2025 — Harsha B J.

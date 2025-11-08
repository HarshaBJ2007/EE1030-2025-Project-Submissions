# Image Compression using Truncated and Randomized SVD (Summary)

## Overview
This project presents an efficient image compression framework built on **Truncated Singular Value Decomposition (SVD)** and **Randomized SVD (rSVD)**. It uses a hybrid **Python + C** architecture, where Python handles visualization, I/O, and user control, while C performs computationally intensive operations like matrix decomposition, random projection, and reconstruction.

---

## Key Objectives
- Implement SVD and Randomized SVD from scratch in C for performance.
- Use Python for image handling, plotting, and computing metrics.
- Compare reconstruction accuracy for different compression ranks \(k\).
- Analyze the trade-off between **compression ratio**, **accuracy**, and **computational cost**.

---

## Mathematical Basis
A grayscale image can be represented as a matrix \(A \in \mathbb{R}^{m\times n}\), decomposed as:
\[
A = U \Sigma V^T
\]
where \(U\) and \(V\) are orthogonal matrices, and \(\Sigma\) contains singular values sorted in decreasing order.

For compression, only the top \(k\) singular values and corresponding vectors are retained:
\[
A_k = U_k \Sigma_k V_k^T
\]
This yields the best rank-\(k\) approximation of \(A\) under the Frobenius norm:
\[
\|A - A_k\|_F = \min_{rank(B)=k} \|A - B\|_F
\]

---

## Randomized SVD Implementation (C Backend)

The **Randomized SVD** algorithm enhances scalability using random projections. Steps:
1. Generate a random Gaussian matrix \(\Omega\).
2. Compute \(Y = A\Omega\).
3. Orthonormalize \(Y\) using Modified Gram-Schmidt → \(Q\).
4. Project \(A\) into low-dimensional space: \(B = Q^T A\).
5. Compute SVD of \(B\) using Jacobi eigen decomposition.
6. Approximate \(A \approx Q U_b \Sigma V^T\).

This approach reduces computational cost while retaining high reconstruction accuracy.

---

## Python Frontend
Python acts as the front-end for:
- Reading and normalizing images (using `imageio`)
- Calling C functions through `ctypes`
- Displaying reconstructed images
- Computing metrics (via `libmetrics.so`)
- Plotting error and compression trends

---

## Performance Metrics

| Metric | Formula | Interpretation |
|:--------|:---------|:---------------|
| **RMSE** | \(\sqrt{\frac{1}{mn}\sum (A - A_k)^2}\) | Average pixel reconstruction error |
| **PSNR** | \(20\log_{10}(\frac{1}{RMSE})\) | Measures signal quality (dB) |
| **Frobenius Error** | \(\|A - A_k\|_F\) | Total deviation energy |
| **Compression Ratio (CR)** | \(\frac{k(m+n+1)}{mn}\) | Data retained vs. original |

---

## Observations
- As \(k\) increases:
  - RMSE ↓ (error decreases)
  - PSNR ↑ (quality improves)
  - Frobenius norm ↓ until convergence
  - Compression Ratio ↑
- Randomized SVD achieves near-optimal performance with faster runtime.

---

## Color Image Extension
The algorithm is extended to color images by applying SVD independently to each R, G, and B channel. The results are merged to reconstruct the color image with minimal perceptual degradation.

---

## Error Analysis Summary
| Image | Rank (k) | RMSE | PSNR (dB) | Frobenius Norm | CR |
|:-------|:---------|:------|:-----------|:----------------|:--:|
| Einstein.jpg | 30 | 0.045 | 26.9 | 8.31 | 0.32 |
| Globe.jpg | 40 | 0.039 | 28.0 | 7.92 | 0.35 |
| Greyscale.png | 50 | 0.035 | 28.6 | 7.15 | 0.39 |

---

## Trade-offs
- **Accuracy vs Compression:** Increasing rank improves accuracy but reduces compression.
- **Randomized vs Classical SVD:** Randomized SVD trades minor accuracy for speed.
- **Python vs C:** Python offers usability; C provides raw performance.
- **Numerical Stability:** Controlled via normalization and double precision.

---

## Key Insights
- Randomized SVD is an effective low-cost approximation method.
- The hybrid Python+C design achieves strong speedups while remaining fully transparent.
- Frobenius error curves confirm energy concentration in leading singular values.
- Compression maintains perceptual quality even at moderate ranks.

---

## Future Extensions
- GPU acceleration using CUDA.
- Parallel Jacobi eigenvalue solver.
- Adaptive rank selection based on image energy retention.

---

**Author:** Harsha B J (EE25BTECH11026)  
**Institution:** IIT Hyderabad  
**Course:** Matrix Theory – Software Assignment  
**Date:** November 2025


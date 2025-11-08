import ctypes
import numpy as np
import imageio.v2 as iio
import matplotlib.pyplot as plt
import os
import math
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))

# Locate C backend directory relative to this script:
c_backend_dir = os.path.abspath(os.path.join(script_dir, "..", "C_backend"))
libsvd_path = os.path.join(c_backend_dir, "libsvd.so")
libmetrics_path = os.path.join(c_backend_dir, "libmetrics.so")

if not os.path.exists(libsvd_path):
    raise FileNotFoundError(f"libsvd.so not found at: {libsvd_path}")
if not os.path.exists(libmetrics_path):
    raise FileNotFoundError(f"libmetrics.so not found at: {libmetrics_path}")

# Load libsvd
try:
    libsvd = ctypes.CDLL(libsvd_path)
except OSError as e:
    raise RuntimeError(f"Failed to load {libsvd_path}: {e}")

# optional seed function: call if available
try:
    libsvd.svd_seed.argtypes = [ctypes.c_uint]
    # call with fixed seed for deterministic runs; ignore failures at runtime
    try:
        libsvd.svd_seed(ctypes.c_uint(12345))
    except Exception:
        # some builds may not expose svd_seed symbol — that's fine
        pass
except AttributeError:
    # symbol not present; continue
    pass

# svd_compress signature
libsvd.svd_compress.argtypes = [
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_int, ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_double)
]
libsvd.svd_compress.restype = None

# Load metrics lib
try:
    libmetrics = ctypes.CDLL(libmetrics_path)
except OSError as e:
    raise RuntimeError(f"Failed to load {libmetrics_path}: {e}")

libmetrics.compute_metrics.argtypes = [
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_int, ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_double)
]
libmetrics.compute_metrics.restype = None


def svd_compress_py(A: np.ndarray, k: int) -> np.ndarray:
    m, n = A.shape
    A_flat = A.astype(np.float64).ravel()
    out = np.zeros_like(A_flat)
    libsvd.svd_compress(
        A_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(m),
        ctypes.c_int(n),
        ctypes.c_int(k),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    )
    return out.reshape(m, n)


def compute_metrics_full_c(original: np.ndarray, reconstructed: np.ndarray, k: int):
    if original.shape != reconstructed.shape:
        raise ValueError("original and reconstructed must have same shape")
    m, n = original.shape
    out_arr = (ctypes.c_double * 4)()
    orig_flat = original.astype(np.float64).ravel()
    rec_flat = reconstructed.astype(np.float64).ravel()

    libmetrics.compute_metrics(
        orig_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        rec_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(m), ctypes.c_int(n),
        ctypes.c_int(k),
        out_arr
    )

    return {
        "rmse": float(out_arr[0]),
        "psnr": float(out_arr[1]),
        "fro_error": float(out_arr[2]),
        "cr": float(out_arr[3])
    }


def grid_shape(n):
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    return rows, cols


def nice_print_metrics(k, metrics):
    print(f"Metrics for k={k}:")
    print(f"  Fro_error (||A - A_k||_F): {metrics['fro_error']:.6f}")
    print(f"  RMSE                      : {metrics['rmse']:.6e}")
    print(f"  PSNR (dB)                : {metrics['psnr']:.2f}")
    print(f"  Compression Ratio (CR)   : {metrics['cr']:.6f}")


def main():
    if len(sys.argv) >= 2:
        img_path = sys.argv[1]
    else:
        img_path = input("Enter grayscale image path: ").strip()

    if not os.path.exists(img_path):
        print("Image not found:", img_path)
        return

    img = iio.imread(img_path, pilmode="L").astype(np.float64) / 255.0
    if img.ndim != 2:
        raise ValueError("Image must be grayscale (2D).")

    m, n = img.shape
    print(f"Loaded image: {m} x {n}")

    max_k = min(m, n)
    results = []

    while True:
        try:
            k_raw = input(f"Enter compression rank k (1 - {max_k}) [enter for {max_k}]: ").strip()
            k = int(k_raw) if k_raw else max_k
        except ValueError:
            print("Invalid integer. Try again.")
            continue
        k = max(1, min(k, max_k))

        print(f"\nCompressing with k={k} ...")
        rec = svd_compress_py(img, k)

        rec = np.nan_to_num(rec, nan=0.0, posinf=1.0, neginf=0.0)
        rec = np.clip(rec, 0.0, 1.0)

        metrics = compute_metrics_full_c(img, rec, k)
        nice_print_metrics(k, metrics)

        # output path: data/output relative to repository root (two levels up from script_dir)
        default_out = os.path.abspath(os.path.join(script_dir, "..", "..", "data", "output"))
        out_dir = default_out if os.path.exists(os.path.dirname(default_out)) else os.path.join(script_dir, "output")
        os.makedirs(out_dir, exist_ok=True)

        base = os.path.splitext(os.path.basename(img_path))[0]
        out_path = os.path.join(out_dir, f"{base}_k{k}.png")
        iio.imwrite(out_path, (rec * 255.0 + 0.5).astype(np.uint8))
        print("Saved ->", out_path)

        results.append((k, rec, metrics))

        yn = input("Add another k? (y/n): ").strip().lower()
        if yn not in ("y", "yes"):
            break

    total = len(results) + 1
    rows, cols = grid_shape(total)
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = np.array(axes).reshape(-1)

    axes[0].imshow(img, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Original")
    axes[0].axis("off")

    for idx, (k, rec, metrics) in enumerate(results, start=1):
        ax = axes[idx]
        ax.imshow(rec, cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"k={k}")
        ax.axis("off")
        txt = (f"Fro_err={metrics['fro_error']:.4f}\n"
               f"PSNR={metrics['psnr']:.2f} dB\n"
               f"CR={metrics['cr']:.3f}")
        ax.text(0.5, -0.12, txt, transform=ax.transAxes, ha='center', fontsize=8)

    for j in range(1 + len(results), rows * cols):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()

    print("\nSummary:")
    print("k\tFro_err\t\tRMSE\t\tPSNR\tCR")
    for k, rec, metrics in results:
        print(f"{k}\t{metrics['fro_error']:.6f}\t{metrics['rmse']:.6e}\t{metrics['psnr']:.2f}\t{metrics['cr']:.6f}")


if __name__ == "__main__":
    main()


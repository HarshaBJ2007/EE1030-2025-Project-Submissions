import os
import ctypes
import numpy as np
import imageio.v2 as iio
import matplotlib.pyplot as plt
import math
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(script_dir, ".."))
C_BACKEND = os.path.join(BASE_DIR, "c_backend")
TOP_C_LIBS = os.path.abspath(os.path.join(BASE_DIR, "..", "c_libs"))
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def find_lib(name):
    candidates = [
        os.path.join(C_BACKEND, name),
        os.path.join(TOP_C_LIBS, name)
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None

libsvd_path = find_lib("libsvd.so")
libmetrics_path = find_lib("libmetrics.so")

if libsvd_path is None:
    raise FileNotFoundError("libsvd.so not found in hybrid_c_python/c_backend or ../c_libs")
if libmetrics_path is None:
    raise FileNotFoundError("libmetrics.so not found in hybrid_c_python/c_backend or ../c_libs")

try:
    libsvd = ctypes.CDLL(libsvd_path)
except OSError as e:
    raise RuntimeError(f"Failed to load {libsvd_path}: {e}")

# optional seed function
if hasattr(libsvd, "svd_seed"):
    try:
        libsvd.svd_seed.argtypes = [ctypes.c_uint]
        libsvd.svd_seed(ctypes.c_uint(12345))
    except Exception:
        pass

libsvd.svd_compress.argtypes = [
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_int, ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_double)
]
libsvd.svd_compress.restype = None

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

def safe_imread(path):
    img = iio.imread(path)
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[..., :3]
    img = img.astype(np.float64)
    if img.max() > 1.5:
        img /= 255.0
    return img

def luma_from_rgb(img):
    if img.ndim == 2:
        return img
    R, G, B = img[..., 0], img[..., 1], img[..., 2]
    return 0.2989 * R + 0.5870 * G + 0.1140 * B

def svd_compress_channel(A, k):
    m, n = A.shape
    A_flat = np.ascontiguousarray(A.astype(np.float64).ravel())
    out_flat = np.zeros_like(A_flat, dtype=np.float64)
    libsvd.svd_compress(
        A_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(m), ctypes.c_int(n), ctypes.c_int(k),
        out_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    )
    return out_flat.reshape(m, n)

def compress_image(img, k):
    if img.ndim == 2:
        rec = svd_compress_channel(img, k)
        return np.clip(np.nan_to_num(rec), 0.0, 1.0)
    if img.ndim == 3 and img.shape[2] == 3:
        recs = []
        for ch in range(3):
            rec_ch = svd_compress_channel(img[..., ch], k)
            recs.append(np.clip(np.nan_to_num(rec_ch), 0.0, 1.0))
        return np.stack(recs, axis=2)
    raise ValueError("Unsupported image format")

def compute_metrics_c(orig_luma, rec_luma, k):
    m, n = orig_luma.shape
    orig_flat = np.ascontiguousarray(orig_luma.astype(np.float64).ravel())
    rec_flat = np.ascontiguousarray(rec_luma.astype(np.float64).ravel())
    out_arr = (ctypes.c_double * 4)()
    libmetrics.compute_metrics(
        orig_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        rec_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(m), ctypes.c_int(n), ctypes.c_int(k),
        out_arr
    )
    v0 = float(out_arr[0])
    psnr = float(out_arr[1])
    fro_error = float(out_arr[2])
    cr = float(out_arr[3])
    if v0 > 1.0:
        mse = v0
        rmse = math.sqrt(mse)
    else:
        rmse = v0
        mse = rmse * rmse
    return {"m": m, "n": n, "mse": mse, "rmse": rmse, "psnr": psnr,
            "fro_error": fro_error, "cr": cr}

def grid_shape(n):
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    return rows, cols

def ensure_axes_array(axes, total):
    axes = np.array(axes).reshape(-1)
    if axes.size < total:
        new_axes = np.empty(total, dtype=object)
        new_axes[:] = None
        new_axes[:axes.size] = axes
        axes = new_axes
    return axes

def main():
 
    img_path = sys.argv[1] if len(sys.argv) > 1 else input("Enter image path: ").strip()
    if not os.path.isabs(img_path):
        img_path = os.path.join(BASE_DIR, img_path)
    if not os.path.exists(img_path):
        print("Image not found:", img_path)
        return

    img = safe_imread(img_path)
    m, n = img.shape[:2]
    max_k = min(m, n)
    results = []

    while True:
        try:
            k_raw = input(f"Enter compression rank k (1-{max_k}) [default={max_k}]: ").strip()
            k = int(k_raw) if k_raw else max_k
        except ValueError:
            print("Invalid integer")
            continue
        k = max(1, min(k, max_k))

        print(f"\nCompressing for k={k}...")
        rec = compress_image(img, k)
        orig_l, rec_l = luma_from_rgb(img), luma_from_rgb(rec)
        metrics = compute_metrics_c(orig_l, rec_l, k)

        print(f"Image size: {metrics['m']}x{metrics['n']}")
        print(f"Fro_error: {metrics['fro_error']:.6f}")
        print(f"RMSE     : {metrics['rmse']:.6e}")
        print(f"PSNR (dB): {metrics['psnr']:.2f}")
        print(f"CR       : {metrics['cr']:.6f}")

        base = os.path.splitext(os.path.basename(img_path))[0]
        out_path = os.path.join(OUTPUT_DIR, f"{base}_k{k}.jpg")
        iio.imwrite(out_path, (np.clip(rec, 0, 1) * 255 + 0.5).astype(np.uint8))
        print("Saved ->", out_path)

        results.append((k, rec, metrics))
        yn = input("Add another k? (y/n): ").strip().lower()
        if yn not in ("y", "yes"):
            break

    total = 1 + len(results)
    rows, cols = grid_shape(total)
    fig, axes_raw = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = ensure_axes_array(axes_raw, rows * cols)

    axes[0].imshow(np.clip(img, 0, 1))
    axes[0].set_title("Original")
    axes[0].axis("off")

    for i, (k, rec, metrics) in enumerate(results, start=1):
        ax = axes[i]
        ax.imshow(np.clip(rec, 0, 1))
        ax.set_title(f"k={k}")
        ax.axis("off")
        txt = (f"Fro={metrics['fro_error']:.4f}\n"
               f"RMSE={metrics['rmse']:.4e}\n"
               f"PSNR={metrics['psnr']:.2f}  CR={metrics['cr']:.3f}")
        ax.text(0.5, -0.12, txt, transform=ax.transAxes, ha='center', fontsize=8)

    for j in range(1 + len(results), rows * cols):
        if axes[j] is not None:
            axes[j].axis("off")

    plt.tight_layout()
    plt.show()

    print("\nSummary (k, Fro_err, RMSE, PSNR, CR):")
    print("K\tFro_err\t\tRMSE\t\t\tPSNR\t\tCR")
    for k, _, metrics in results:
        print(f"{k}\t{metrics['fro_error']:.6f}\t"
              f"{metrics['rmse']:.6e}\t{metrics['psnr']:.2f}\t{metrics['cr']:.6f}")

if __name__ == "__main__":
    main()


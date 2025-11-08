#include <math.h>
#include <stddef.h>
#include <float.h>
#include <stdio.h>

void compute_metrics(const double *orig_flat, const double *rec_flat, int m, int n, int k, double *out) {
    if (!orig_flat || !rec_flat || !out || m <= 0 || n <= 0 || k <= 0) {
        if (out) {
            for (int i = 0; i < 4; ++i) out[i] = NAN;
        }
        return;
    }

    long long N = (long long)m * (long long)n;
    double sse = 0.0;
    long long valid_count = 0;

    for (long long i = 0; i < N; ++i) {
        double a = orig_flat[i];
        double b = rec_flat[i];

        if (!isfinite(a) || !isfinite(b)) continue;

        double d = a - b;
        sse += d * d;
        ++valid_count;
    }

    if (valid_count <= 0) {
        out[0] = NAN; out[1] = NAN; out[2] = NAN; out[3] = NAN;
        return;
    }

    double mse = sse / (double)valid_count;
    double rmse = sqrt(mse);
    double fro_error = 255.0*sqrt(sse);

    double psnr;
    if (rmse <= 0.0) psnr = INFINITY;
    else psnr = 20.0 * log10(1.0 / rmse); 

    double cr = (double)k * ((double)m + (double)n + 1.0) / ((double)m * (double)n);

    out[0] = rmse;
    out[1] = psnr;
    out[2] = fro_error;
    out[3] = cr;
}


// Randomized SVD implementation for image compression.


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define EPS 1e-12
#define MAX_JACOBI_ITER 2000

double **alloc_matrix(int rows, int cols) {
    double **m = malloc(rows * sizeof(double*));
    if (!m) return NULL;
    for (int i=0;i<rows;i++) {
        m[i] = calloc(cols, sizeof(double));
        if (!m[i]) { for (int r=0;r<i;r++) free(m[r]); free(m); return NULL; }
    }
    return m;
}
void free_matrix(double **m, int rows) {
    if (!m) return;
    for (int i=0;i<rows;i++) free(m[i]);
    free(m);
}
void matmul(double **A, double **B, double **C, int m, int n, int p) {
    for (int i=0;i<m;i++) {
        for (int j=0;j<p;j++) {
            double s = 0.0;
            for (int t=0;t<n;t++) s += A[i][t]*B[t][j];
            C[i][j] = s;
        }
    }
}
double **transpose(double **A, int m, int n) {
    double **T = alloc_matrix(n, m);
    for (int i=0;i<m;i++) for (int j=0;j<n;j++) T[j][i] = A[i][j];
    return T;
}


static double randn() {
    // thread-unsafe but fine here
    static int have = 0;
    static double spare;
    if (have) { have = 0; return spare; }
    double u, v, s;
    do {
        u = 2.0 * ((double)rand()/RAND_MAX) - 1.0;
        v = 2.0 * ((double)rand()/RAND_MAX) - 1.0;
        s = u*u + v*v;
    } while (s == 0.0 || s >= 1.0);
    double mul = sqrt(-2.0 * log(s) / s);
    spare = v * mul;
    have = 1;
    return u * mul;
}


void mgs_orthonormalize(double **Y, int m, int l) {
    for (int j=0;j<l;j++) {
        // subtract projections onto previous q's
        for (int i=0;i<j;i++) {
            double dot = 0.0;
            for (int r=0;r<m;r++) dot += Y[r][i]*Y[r][j];
            for (int r=0;r<m;r++) Y[r][j] -= dot * Y[r][i];
        }
        // normalize
        double norm = 0.0;
        for (int r=0;r<m;r++) norm += Y[r][j]*Y[r][j];
        norm = sqrt(norm);
        if (norm < 1e-15) { // set to zero column
            for (int r=0;r<m;r++) Y[r][j] = 0.0;
        } else {
            for (int r=0;r<m;r++) Y[r][j] /= norm;
        }
    }
}


void jacobi_eigen_small(double **A, double *eigenvals, double **evecs, int n) {
    // init evecs = identity
    for (int i=0;i<n;i++) for (int j=0;j<n;j++) evecs[i][j] = (i==j) ? 1.0 : 0.0;

    for (int iter=0; iter<MAX_JACOBI_ITER; ++iter) {
        int p=-1,q=-1;
        double maxv = 0.0;
        for (int i=0;i<n;i++) for (int j=i+1;j<n;j++) {
            double val = fabs(A[i][j]);
            if (val > maxv) { maxv = val; p=i; q=j; }
        }
        if (maxv < 1e-12) break;
        double app = A[p][p], aqq = A[q][q], apq = A[p][q];
        double theta = 0.5 * atan2(2.0*apq, aqq - app);
        double c = cos(theta), s = sin(theta);
        double new_app = c*c*app - 2*c*s*apq + s*s*aqq;
        double new_aqq = s*s*app + 2*c*s*apq + c*c*aqq;
        A[p][p] = new_app; A[q][q] = new_aqq; A[p][q] = A[q][p] = 0.0;
        for (int k=0;k<n;k++) if (k!=p && k!=q) {
            double akp = A[k][p], akq = A[k][q];
            A[k][p] = A[p][k] = c*akp - s*akq;
            A[k][q] = A[q][k] = s*akp + c*akq;
        }
        // update eigenvectors
        for (int r=0;r<n;r++) {
            double vip = evecs[r][p], viq = evecs[r][q];
            evecs[r][p] = c*vip - s*viq;
            evecs[r][q] = s*vip + c*viq;
        }
    }
    for (int i=0;i<n;i++) eigenvals[i] = A[i][i];
}


void svd_compress(double *A_flat, int m, int n, int k, double *out_flat) {
    if (m<=0 || n<=0 || k<=0) return;
    if (k > (m<n?m:n)) k = (m<n?m:n);
    int p = 10;                      // oversampling
    int l = (k + p <= (n< m? n: m) ? k + p : (n< m? n: m));

    // Build A as matrix
    double **A = alloc_matrix(m, n);
    for (int i=0;i<m;i++) for (int j=0;j<n;j++) A[i][j] = A_flat[i*n + j];

    // 1) Form random Gaussian matrix Omega (n x l)
    double **Omega = alloc_matrix(n, l);
    srand((unsigned)time(NULL));
    for (int i=0;i<n;i++) for (int j=0;j<l;j++) Omega[i][j] = randn();

    // 2) Y = A * Omega  (m x l)
    double **Y = alloc_matrix(m, l);
    matmul(A, Omega, Y, m, n, l);

    // 3) Orthonormalize Y -> Q (in-place)
    mgs_orthonormalize(Y, m, l); // now Y contains orthonormal columns Q

    // 4) B = Q^T * A  (l x n). compute transpose(Q) = Y^T
    double **Qtrans = transpose(Y, m, l);
    double **B = alloc_matrix(l, n);
    matmul(Qtrans, A, B, l, m, n);

    // 5) Compute small SVD of B (we compute eigen of B * B^T which is (l x l))
    double **BBt = alloc_matrix(l, l);
    matmul(B, transpose(B, l, n), BBt, l, n, l); // B * B^T
    // Note: matmul(B, B^T) -> B (l x n) * B^T (n x l) -> l x l

    // Prepare storage for small eigen
    double *eig = calloc(l, sizeof(double));
    double **Ub = alloc_matrix(l, l);
    jacobi_eigen_small(BBt, eig, Ub, l);
    // Sort eigenvalues & Ub columns descending
    for (int i=0;i<l-1;i++) for (int j=i+1;j<l;j++) {
        if (eig[i] < eig[j]) {
            double tmp = eig[i]; eig[i] = eig[j]; eig[j] = tmp;
            for (int r=0;r<l;r++) {
                double t = Ub[r][i]; Ub[r][i] = Ub[r][j]; Ub[r][j] = t;
            }
        }
    }
    // singular values (for B)
    double *sigma = calloc(l, sizeof(double));
    for (int i=0;i<l;i++) sigma[i] = (eig[i] > 0.0) ? sqrt(eig[i]) : 0.0;

    // Compute V_small: V = B^T * Ub / sigma  (n x l)
    double **Vsmall = alloc_matrix(n, l);
    // First compute B^T (n x l)
    double **Bt = transpose(B, l, n);
    for (int i=0;i<l;i++) {
        if (sigma[i] < 1e-14) {
            for (int r=0;r<n;r++) Vsmall[r][i] = 0.0;
        } else {
            for (int r=0;r<n;r++) {
                double sum = 0.0;
                for (int t=0;t<l;t++) sum += Bt[r][t] * Ub[t][i];
                Vsmall[r][i] = sum / sigma[i];
            }
        }
    }

    // Orthonormalize Vsmall columns
    mgs_orthonormalize(Vsmall, n, l);

    // Compute final left singular vectors U_k = Q * Ub (m x l) -> take first k columns
    double **Ufull = alloc_matrix(m, l);
    matmul(Y, Ub, Ufull, m, l, l); // Y is Q (m x l)

    // Reconstruct A_k using top-k singulars (use Ufull[:,i], sigma[i], Vsmall[:,i])
    double **Ak = alloc_matrix(m, n);
    for (int i=0;i<k;i++) {
        double s = sigma[i];
        if (!(s > 0.0)) continue;
        for (int r=0;r<m;r++) {
            double u = Ufull[r][i];
            double coef = s * u;
            for (int c=0;c<n;c++) Ak[r][c] += coef * Vsmall[c][i];
        }
    }

    // Copy to out_flat; clip to [0,1] (assume input ∈ [0,1])
    for (int r=0;r<m;r++) {
        for (int c=0;c<n;c++) {
            double v = Ak[r][c];
            if (!isfinite(v)) v = 0.0;
            if (v < 0.0) v = 0.0;
            if (v > 1.0) v = 1.0;
            out_flat[r*n + c] = v;
        }
    }

    // cleanup
    free_matrix(A, m);
    free_matrix(Omega, n);
    free_matrix(Y, m);
    free_matrix(Qtrans, l);
    free_matrix(B, l);
    free_matrix(BBt, l);
    free(eig);
    free_matrix(Ub, l);
    free(sigma);
    free_matrix(Vsmall, n);
    free_matrix(Bt, n);
    free_matrix(Ufull, m);
    free_matrix(Ak, m);
}


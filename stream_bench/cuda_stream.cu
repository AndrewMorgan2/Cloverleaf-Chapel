// cuda_stream.cu — GPU STREAM benchmark (device memory, CUDA events for timing)
// Matches the Chapel version: N=40M doubles, best-of-NTIMES per operation.

#include <stdio.h>
#include <float.h>
#include <cuda_runtime.h>

#define N       40000000
#define NTIMES  20
#define SCALAR  3.0

#define CHECK(x) do { \
    cudaError_t _e = (x); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
        exit(1); \
    } \
} while(0)

__global__ void init_kernel(double* A, double* B, double* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { A[i] = 1.0; B[i] = 2.0; C[i] = 0.0; }
}

__global__ void copy_kernel(double* __restrict__ C, const double* __restrict__ A, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) C[i] = A[i];
}

__global__ void scale_kernel(double* __restrict__ B, const double* __restrict__ C,
                              double s, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) B[i] = s * C[i];
}

__global__ void add_kernel(double* __restrict__ C,
                           const double* __restrict__ A,
                           const double* __restrict__ B, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) C[i] = A[i] + B[i];
}

__global__ void triad_kernel(double* __restrict__ A,
                              const double* __restrict__ B,
                              const double* __restrict__ C,
                              double s, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) A[i] = B[i] + s * C[i];
}

static double time_kernel(cudaEvent_t ev0, cudaEvent_t ev1) {
    float ms;
    CHECK(cudaEventSynchronize(ev1));
    CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
    return ms / 1000.0;   // seconds
}

int main(void) {
    double *d_A, *d_B, *d_C;
    size_t bytes = (size_t)N * sizeof(double);

    printf("CUDA STREAM  N=%d  NTIMES=%d\n", N, NTIMES);
    printf("Array size: %.1f MB each\n\n", bytes / 1024.0 / 1024.0);

    CHECK(cudaMalloc(&d_A, bytes));
    CHECK(cudaMalloc(&d_B, bytes));
    CHECK(cudaMalloc(&d_C, bytes));

    const int BLK = 1024;
    const int GRD = (N + BLK - 1) / BLK;

    cudaEvent_t ev0, ev1;
    CHECK(cudaEventCreate(&ev0));
    CHECK(cudaEventCreate(&ev1));

    // Initialise
    init_kernel<<<GRD, BLK>>>(d_A, d_B, d_C, N);
    CHECK(cudaDeviceSynchronize());

    // Warmup
    copy_kernel<<<GRD, BLK>>>(d_C, d_A, N);
    CHECK(cudaDeviceSynchronize());

    // ---- Copy ----
    double bestCopy = DBL_MAX;
    for (int r = 0; r < NTIMES; r++) {
        CHECK(cudaEventRecord(ev0));
        copy_kernel<<<GRD, BLK>>>(d_C, d_A, N);
        CHECK(cudaEventRecord(ev1));
        double t = time_kernel(ev0, ev1);
        if (t < bestCopy) bestCopy = t;
    }
    printf("Copy:  %.1f GB/s\n", 2.0 * 8 * N / bestCopy / 1e9);

    // ---- Scale ----
    double bestScale = DBL_MAX;
    for (int r = 0; r < NTIMES; r++) {
        CHECK(cudaEventRecord(ev0));
        scale_kernel<<<GRD, BLK>>>(d_B, d_C, SCALAR, N);
        CHECK(cudaEventRecord(ev1));
        double t = time_kernel(ev0, ev1);
        if (t < bestScale) bestScale = t;
    }
    printf("Scale: %.1f GB/s\n", 2.0 * 8 * N / bestScale / 1e9);

    // ---- Add ----
    double bestAdd = DBL_MAX;
    for (int r = 0; r < NTIMES; r++) {
        CHECK(cudaEventRecord(ev0));
        add_kernel<<<GRD, BLK>>>(d_C, d_A, d_B, N);
        CHECK(cudaEventRecord(ev1));
        double t = time_kernel(ev0, ev1);
        if (t < bestAdd) bestAdd = t;
    }
    printf("Add:   %.1f GB/s\n", 3.0 * 8 * N / bestAdd / 1e9);

    // ---- Triad ----
    double bestTriad = DBL_MAX;
    for (int r = 0; r < NTIMES; r++) {
        CHECK(cudaEventRecord(ev0));
        triad_kernel<<<GRD, BLK>>>(d_A, d_B, d_C, SCALAR, N);
        CHECK(cudaEventRecord(ev1));
        double t = time_kernel(ev0, ev1);
        if (t < bestTriad) bestTriad = t;
    }
    printf("Triad: %.1f GB/s\n", 3.0 * 8 * N / bestTriad / 1e9);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaEventDestroy(ev0); cudaEventDestroy(ev1);
    return 0;
}

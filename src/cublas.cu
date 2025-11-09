#include <assert.h>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <cublas_v2.h>

using namespace cute;

// CUDA API error checking
#define CUDA_CHECK(err)                                                                            \
    do {                                                                                           \
        cudaError_t err_ = (err);                                                                  \
        if (err_ != cudaSuccess) {                                                                 \
            std::printf("CUDA error %d at %s:%d\n", err_, __FILE__, __LINE__);                     \
            throw std::runtime_error("CUDA error");                                                \
        }                                                                                          \
    } while (0)

// cublas API error checking
#define CUBLAS_CHECK(err)                                                                          \
    do {                                                                                           \
        cublasStatus_t err_ = (err);                                                               \
        if (err_ != CUBLAS_STATUS_SUCCESS) {                                                       \
            std::printf("cublas error %d at %s:%d\n", err_, __FILE__, __LINE__);                   \
            throw std::runtime_error("cublas error");                                              \
        }                                                                                          \
    } while (0)

double alpha = 1.0;
double beta = 0.0;

constexpr int N_BENCHMARK_ITERS = 10;

// int main() {
//     cublasHandle_t cublasH = NULL;

//     CUBLAS_CHECK(cublasCreate(&cublasH));

//     int m = 2048; // 449*8
//     int n = 2048;
//     int k = 2048;

//     srand(42);

//     // allocate and initialize host buffers
//     double *a_host = (double *)std::malloc(m * k * sizeof(double));
//     double *b_host = (double *)std::malloc(k * n * sizeof(double));
//     double *c_host = (double *)std::malloc(m * n * sizeof(double));
//     for (int i = 0; i < m * k; i++)
//         a_host[i] = static_cast<double>(2 * (rand() / double(RAND_MAX)) - 1);
//     for (int i = 0; i < n * k; i++)
//         b_host[i] = static_cast<double>(2 * (rand() / double(RAND_MAX)) - 1);

//     // allocate and initialize device buffers
//     double *a_dev;
//     double *b_dev;
//     double *c_dev;
//     cudaMalloc(&a_dev, m * k * sizeof(double));
//     cudaMalloc(&b_dev, n * k * sizeof(double));
//     cudaMalloc(&c_dev, m * n * sizeof(double));
//     cudaMemcpy(a_dev, a_host, m * k * sizeof(double), cudaMemcpyHostToDevice);
//     cudaMemcpy(b_dev, b_host, n * k * sizeof(double), cudaMemcpyHostToDevice);

//     cudaEvent_t start, end;
//     cudaEventCreate(&start);
//     cudaEventCreate(&end);

//     cudaDeviceSynchronize();
//     cudaEventRecord(start);
//     for (int i = 0; i < N_BENCHMARK_ITERS; i++) {
//         // TN gemm
//         cublasDgemm_v2(cublasH, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alpha, a_dev, k, b_dev, k, &beta, c_dev, m);
//     }
//     cudaEventRecord(end);
//     cudaEventSynchronize(end);
//     cudaDeviceSynchronize();

//     cudaError_t err = cudaGetLastError();
//     if (err != cudaSuccess) {
//         printf("CUDA error: %s\n", cudaGetErrorString(err));
//     }
//     cudaMemcpy(c_host, c_dev, m * n * sizeof(double), cudaMemcpyDeviceToHost);

//     float elapsed_ms;
//     cudaEventElapsedTime(&elapsed_ms, start, end);
//     printf("elapsed ms %f\n", elapsed_ms);
//     double s = (double)elapsed_ms / N_BENCHMARK_ITERS / 1000.0;
//     double gflop = (2.0 * m * n * k) * 1e-9;
//     printf("%f GFLOP/S -- %.2f ms\n", gflop / s, s * 1e3);

//     // correctness test
//     double atol = 1e-7;
//     double rtol = 1e-7;
//     for (int i = 0; i < m; i++) {
//         for (int j = 0; j < n; j++) {
//             double acc = (double)0.0;
//             for (int _k = 0; _k < k; _k++) {
//                 acc += a_host[i * k + _k] * b_host[j * k + _k];
//             }

//             // instead of storing, just check against kernel result
//             if (fabs(acc - c_host[i + j * m]) > atol + rtol * fabs(acc)) {
//                 printf("FAILED! WRONG AT (%d,%d): expected %.20f, got %.20f "
//                        "(diff %.32f)\n",
//                        i, j, (float)acc, (float)c_host[i + j * m], fabs(acc - c_host[i + j * m]));
//                 return 1;
//             }
//         }
//     }
//     printf("PASSED!\n");

//     std::free(a_host);
//     std::free(b_host);
//     std::free(c_host);
//     cudaFree(a_dev);
//     cudaFree(b_dev);
//     cudaFree(c_dev);
//     return 0;
// }


double time_dgemm(cublasHandle_t *cublasH, int m, int n, int k) {
    // allocate and initialize host buffers
    double *a_host = (double *)std::malloc(m * k * sizeof(double));
    double *b_host = (double *)std::malloc(n * k * sizeof(double));
    double *c_host = (double *)std::malloc(m * n * sizeof(double));
    for (int i = 0; i < m * k; i++)
        a_host[i] = static_cast<double>(2 * (rand() / double(RAND_MAX)) - 1);
    for (int i = 0; i < n * k; i++)
        b_host[i] = static_cast<double>(2 * (rand() / double(RAND_MAX)) - 1);

    // allocate and initialize device buffers
    double *a_dev;
    double *b_dev;
    double *c_dev;
    cudaMalloc(&a_dev, m * k * sizeof(double));
    cudaMalloc(&b_dev, n * k * sizeof(double));
    cudaMalloc(&c_dev, m * n * sizeof(double));
    cudaMemcpy(a_dev, a_host, m * k * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(b_dev, b_host, n * k * sizeof(double), cudaMemcpyHostToDevice);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cudaDeviceSynchronize();
    cudaEventRecord(start);
    for (int i = 0; i < N_BENCHMARK_ITERS; i++) {
        cublasDgemm_v2(*cublasH, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alpha, a_dev, k, b_dev, k, &beta, c_dev, m);
    }
    // return 0;
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }
    cudaMemcpy(c_host, c_dev, m * n * sizeof(double), cudaMemcpyDeviceToHost);

    float elapsed_ms;
    cudaEventElapsedTime(&elapsed_ms, start, end);
    // printf("elapsed ms %f\n", elapsed_ms);
    double s = (double)elapsed_ms / N_BENCHMARK_ITERS / 1000.0;

    std::free(a_host);
    std::free(b_host);
    std::free(c_host);
    cudaFree(a_dev);
    cudaFree(b_dev);
    cudaFree(c_dev);
    return s;
}

int main() {
    cublasHandle_t cublasH = NULL;
    // cudaStream_t stream = NULL;

    CUBLAS_CHECK(cublasCreate(&cublasH));

    srand(42);

    printf("M;N;K;TIME_S;GFLOPS\n");
    for (int size = 64; size <= 8192; size *= 2) {
        double s = time_dgemm(&cublasH, size, size, size);
        double gflop = (2.0 * size * size * size) * 1e-9;
        // printf("%f GFLOP/S -- %.2f ms\n", gflop / s, s * 1e3);
        printf("%d;%d;%d;%f;%f\n", size, size, size, s, gflop / s);
    }

    CUBLAS_CHECK(cublasDestroy(cublasH));
    return 0;
}

#include <assert.h>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cute/tensor.hpp>

using namespace cute;

using TA = double;
using TB = double;
using TC = double;
double alpha = 1.0;
double beta = 0.0;

constexpr int WARP_SIZE = 32;
constexpr int N_BENCHMARK_ITERS = 10;

template <class ElementA, class ElementB, class SmemLayoutA, class SmemLayoutB>
struct SharedStorage
{
  cute::ArrayEngine<ElementA, cute::cosize_v<SmemLayoutA>> A;
  cute::ArrayEngine<ElementB, cute::cosize_v<SmemLayoutB>> B;
};

template <class CtaTilerShape, class TA, class TB, class TC, class TensorA, class TensorB, class TensorC,
          class TiledMMAC, class ASmemLayout, class BSmemLayout, class TiledCopyA, class TiledCopyB, class Alpha,
          class Beta>
__global__ void dgemm_kernel(
    CtaTilerShape cta_tiler, TensorA mA, TensorB mB, TensorC mC, TiledMMAC tiled_mma, ASmemLayout sA_layout,
    BSmemLayout sB_layout, TiledCopyA tiled_copy_a, TiledCopyB tiled_copy_b, Alpha alpha, Beta beta) {
    Copy_Atom<UniversalCopy<double>, double> s2r_atom_a;
    Copy_Atom<UniversalCopy<double>, double> s2r_atom_b;

    auto cta_coords = make_coord(blockIdx.x, blockIdx.y, _);
    Tensor gA = local_tile(mA, cta_tiler, cta_coords, Step<_1, X, _1>{}); // (BM, BK, k)
    Tensor gB = local_tile(mB, cta_tiler, cta_coords, Step<X, _1, _1>{}); // (BN, BK, k)
    Tensor gC = local_tile(mC, cta_tiler, cta_coords, Step<_1, _1, X>{}); // (BM, BN)

    // setup shared memory for loading A/B tiles into
    extern __shared__ char shared_memory[];
    using SharedStorage = SharedStorage<TA, TB, ASmemLayout, BSmemLayout>;
    SharedStorage &smem = *reinterpret_cast<SharedStorage *>(shared_memory);
    Tensor sA = make_tensor(make_smem_ptr(smem.A.begin()), sA_layout); // (BM,BK)
    Tensor sB = make_tensor(make_smem_ptr(smem.B.begin()), sB_layout); // (BN,BK)

    // partition `gA`, `gB`, `gC` by the partitioning pattern of the `tiled_mma` and create the fragments
    ThrMMA thr_mma = tiled_mma.get_slice(threadIdx.x);
    Tensor tCgA = thr_mma.partition_A(gA);                   // (THRMMA, MMA_M, MMA_K, k)
    Tensor tCgB = thr_mma.partition_B(gB);                   // (THRMMA, MMA_N, MMA_K, k)
    Tensor tCgC = thr_mma.partition_C(gC);                   // (THRMMA, MMA_M, MMA_N)
    Tensor tCrA = thr_mma.make_fragment_A(tCgA(_, _, _, 0)); // (THRMMA, MMA_M, MMA_K)
    Tensor tCrB = thr_mma.make_fragment_B(tCgB(_, _, _, 0)); // (THRMMA, MMA_N, MMA_K)
    Tensor tCrC = thr_mma.make_fragment_C(tCgC);             // (THRMMA, MMA_M, MMA_N)
    // set accumulator to 0s
    clear(tCrC);

    // setup global to shared copies
    ThrCopy thr_copy_a = tiled_copy_a.get_slice(threadIdx.x);
    Tensor tAgA = thr_copy_a.partition_S(gA); // (THRCPY, CPY_M, CPY_K, k)
    Tensor tAsA = thr_copy_a.partition_D(sA); // (THRCPY, CPY_M, CPY_K)

    ThrCopy thr_copy_b = tiled_copy_b.get_slice(threadIdx.x);
    Tensor tBgB = thr_copy_b.partition_S(gB); // (THRCPY, CPY_N, CPY_K, k)
    Tensor tBsB = thr_copy_b.partition_D(sB); // (THRCPY, CPY_N, CPY_K)

    // setup shared to registers copies
    TiledCopy s2r_copy_a = make_tiled_copy_A(s2r_atom_a, tiled_mma);
    TiledCopy s2r_copy_b = make_tiled_copy_B(s2r_atom_b, tiled_mma);
    ThrCopy s2r_thr_copy_a = s2r_copy_a.get_slice(threadIdx.x);
    ThrCopy s2r_thr_copy_b = s2r_copy_b.get_slice(threadIdx.x);
    Tensor tXsA = s2r_thr_copy_a.partition_S(sA); // (CPY,MMA_M,MMA_K)
    Tensor tXrA = s2r_thr_copy_a.retile_D(tCrA);  // (CPY,MMA_M,MMA_K)
    Tensor tXsB = s2r_thr_copy_b.partition_S(sB); // (CPY,MMA_N,MMA_K)
    Tensor tXrB = s2r_thr_copy_b.retile_D(tCrB);  // (CPY,MMA_N,MMA_K)

    // MMA mainloop
    int nb_k_tiles = size<2>(gA);
    for (int k_tile_idx = 0; k_tile_idx < nb_k_tiles; k_tile_idx++) {
        // load A/B from gmem to smem
        copy(tiled_copy_a, tAgA(_, _, _, k_tile_idx), tAsA(_, _, _)); // start A tile async copy from gmem into smem
        copy(tiled_copy_b, tBgB(_, _, _, k_tile_idx), tBsB(_, _, _)); // start B tile async copy from gmem into smem
        cp_async_fence();   // commit the two async loads into a group of loads
        cp_async_wait<0>(); // wait until 0 groups of loads are pending anymore
        __syncthreads();    // fence to make sure all threads see the smem in the same state before reading it

        // load A/B from smem to rmem
        copy(s2r_atom_a, tXsA, tXrA);
        copy(s2r_atom_b, tXsB, tXrB);

        // call tensor core mma
        gemm(thr_mma, tCrC, tCrA, tCrB, tCrC);
        __syncthreads();
    }

    // store gC = alpha*acc + beta*gC
    axpby(alpha, tCrC, beta, tCgC);
}

void dgemm_tn(int m, int n, int k, double alpha, TA const *a, int lda, TB const *b, int ldb, double beta, TC *c,
              int ldc) {
    TiledMMA tiled_mma = make_tiled_mma(MMA_Atom<SM80_8x8x4_F64F64F64F64_TN>{},
        Layout<Shape<_2, _2>>{},
        Tile<_128,_128,_16>{}
    );
    // TiledMMA tiled_mma = make_tiled_mma(MMA_Atom<SM80_16x8x16_F16F16F16F16_TN>{});

    // print_latex(tiled_mma);
    // print(tiled_mma);
    // print("\n");
    // return;

    // compile time block sizes for tiling (FP64 tensor core shape for now)
    auto bM = tiled_mma.tile_size_mnk<0>();
    auto bN = tiled_mma.tile_size_mnk<1>();
    auto bK = tiled_mma.tile_size_mnk<2>();

    // create block shape (for blocking/tiling)
    auto cta_tiler = make_shape(bM, bN, bK);
    // print(cta_tiler);
    // print("\n");

    // static_assert(bM == _128{}, "BM is not equal to 16");
    // static_assert(bN == _128{}, "BN is not equal to 16");
    // static_assert(bK == _16{}, "BK is not equal to 16");

    assert(m % bM == 0 && "M has to be divisible by it's block size");
    assert(n % bN == 0 && "N has to be divisible by it's block size");
    assert(k % bK == 0 && "K has to be divisible by it's block size");

    // create a, b and c layouts
    auto a_layout = make_layout(make_shape(m, k), make_stride(lda, _1{})); // (M,K) row major
    auto b_layout = make_layout(make_shape(n, k), make_stride(ldb, _1{})); // (N,K) row major
    auto c_layout = make_layout(make_shape(m, n), make_stride(_1{}, ldc)); // (M,N) col major

    // create a, b and c tensors
    Tensor mA = make_tensor(make_gmem_ptr(a), a_layout);
    Tensor mB = make_tensor(make_gmem_ptr(b), b_layout);
    Tensor mC = make_tensor(make_gmem_ptr(c), c_layout);

    // note that here, majorness for value layouts of shape (1,2) doesn't matter:
    // Stride<_2, _1> => f(0, 0)=0, f(0, 1) = 1
    // Stride<_1, _1> => f(0, 0)=0, f(0, 1) = 1
    TiledCopy tiled_copy_a = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, TA>{},
                                             Layout<Shape<_16, _8>, Stride<_8, _1>>{}, // Thr layout 8x2 k-major
                                             Layout<Shape<_1, _2>>{}); // Val layout 1x2 k-major
    TiledCopy tiled_copy_b = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, TB>{},
                                             Layout<Shape<_16, _8>, Stride<_8, _1>>{},  // Thr layout 8x2 k-major
                                             Layout<Shape<_1, _2>>{}); // Val layout 1x2 k-major

    // print_latex(tiled_copy_a);
    // print_latex(tiled_copy_b);
    // print_latex(tiled_mma);
    // print("\n");
    // print(tiled_copy_a.get_layoutS_TV());
    // print(tiled_copy_a.get_layoutD_TV());
    // return;

    auto sA_layout = make_layout(make_shape(bM, bK), make_stride(bK, _1{}));
    auto sB_layout = make_layout(make_shape(bN, bK), make_stride(bK, _1{}));
    int dynamic_smem_size = sizeof(SharedStorage<TA, TB, decltype(sA_layout), decltype(sB_layout)>);
    // printf("dynamic_smem_size: %d bytes\n", dynamic_smem_size);

    auto dgemm_kernel_fnptr = dgemm_kernel<decltype(cta_tiler), TA, TB, TC, decltype(mA), decltype(mB), decltype(mC),
                                           decltype(tiled_mma), decltype(sA_layout), decltype(sB_layout), decltype(tiled_copy_a),
                                           decltype(tiled_copy_b), decltype(alpha), decltype(beta)>;
    // Set L1 to be SMEM only
    cudaFuncSetAttribute(dgemm_kernel_fnptr, cudaFuncAttributeMaxDynamicSharedMemorySize, dynamic_smem_size);
    cudaFuncSetAttribute(dgemm_kernel_fnptr, cudaFuncAttributePreferredSharedMemoryCarveout, 100);

    dim3 grid = dim3(m / bM, n / bN);
    dim3 block = dim3(size(tiled_mma));
    static_assert(size(tiled_mma) == 4*WARP_SIZE, "tiled_mma should be using 4 warps");
    // printf("block size: %d threads\n", block.x);
    dgemm_kernel_fnptr<<<grid, block, dynamic_smem_size>>>(cta_tiler, mA, mB, mC, tiled_mma, sA_layout, sB_layout, tiled_copy_a,
                                                           tiled_copy_b, alpha, beta);
}

// a, b and c pointers should be pointing to gmem
void dgemm(char transa, char transb, int m, int n, int k, double alpha, TA const *a, int lda, TB const *b, int ldb,
           double beta, TC *c, int ldc) {
    assert(transa == 'T' && transb == 'N');
    dgemm_tn(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

double time_dgemm(int m, int n, int k) {
    // allocate and initialize host buffers
    TA *a_host = (TA *)std::malloc(m * k * sizeof(TA));
    TB *b_host = (TB *)std::malloc(n * k * sizeof(TB));
    TC *c_host = (TC *)std::malloc(m * n * sizeof(TC));
    for (int i = 0; i < m * k; i++)
        a_host[i] = static_cast<TA>(2 * (rand() / double(RAND_MAX)) - 1);
    for (int i = 0; i < n * k; i++)
        b_host[i] = static_cast<TA>(2 * (rand() / double(RAND_MAX)) - 1);

    // allocate and initialize device buffers
    TA *a_dev;
    TB *b_dev;
    TC *c_dev;
    cudaMalloc(&a_dev, m * k * sizeof(TA));
    cudaMalloc(&b_dev, n * k * sizeof(TB));
    cudaMalloc(&c_dev, m * n * sizeof(TC));
    cudaMemcpy(a_dev, a_host, m * k * sizeof(TA), cudaMemcpyHostToDevice);
    cudaMemcpy(b_dev, b_host, n * k * sizeof(TB), cudaMemcpyHostToDevice);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cudaDeviceSynchronize();
    cudaEventRecord(start);
    for (int i = 0; i < N_BENCHMARK_ITERS; i++) {
        dgemm('T', 'N', m, n, k, alpha, a_dev, k, b_dev, k, beta, c_dev, m);
    }
    // return 0;
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }
    cudaMemcpy(c_host, c_dev, m * n * sizeof(TC), cudaMemcpyDeviceToHost);

    float elapsed_ms;
    cudaEventElapsedTime(&elapsed_ms, start, end);
    // printf("elapsed ms %f\n", elapsed_ms);
    double s = (double)elapsed_ms / N_BENCHMARK_ITERS / 1000.0;

    // // correctness test
    // double atol = 1e-7;
    // double rtol = 1e-7;
    // for (int i = 0; i < m; i++) {
    //     for (int j = 0; j < n; j++) {
    //         TC acc = (TC)0.0;
    //         for (int _k = 0; _k < k; _k++) {
    //             acc += a_host[i * k + _k] * b_host[j * k + _k];
    //         }
    //         acc *= alpha;

    //         // instead of storing, just check against kernel result
    //         if (fabs(acc - c_host[i + j * m]) > atol + rtol * fabs(acc)) {
    //             printf("FAILED! WRONG AT (%d,%d): expected %.20f, got %.20f "
    //                    "(diff %.32f)\n",
    //                    i, j, (float)acc, (float)c_host[i + j * m], fabs(acc - c_host[i + j * m]));
    //             return 1;
    //         }
    //     }
    // }
    // printf("PASSED!\n");

    std::free(a_host);
    std::free(b_host);
    std::free(c_host);
    cudaFree(a_dev);
    cudaFree(b_dev);
    cudaFree(c_dev);
    return s;
}

int main() {
    srand(42);

    printf("M;N;K;TIME_S;GFLOPS\n");
    for (int size = 128; size <= 8192; size *= 2) {
        double s = time_dgemm(size, size, size);
        double gflop = (2.0 * size * size * size) * 1e-9;
        // printf("%f GFLOP/S -- %.2f ms\n", gflop / s, s * 1e3);
        printf("%d;%d;%d;%f;%f\n", size, size, size, s, gflop / s);
    }
    return 0;
}

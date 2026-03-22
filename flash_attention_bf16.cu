#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CHECK_CUDA(call)                                                        \
    do {                                                                        \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess) {                                             \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,  \
                         cudaGetErrorString(err__));                             \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

#define CHECK_CUBLAS(call)                                                      \
    do {                                                                        \
        cublasStatus_t st__ = (call);                                           \
        if (st__ != CUBLAS_STATUS_SUCCESS) {                                    \
            std::fprintf(stderr, "cuBLAS error %s:%d: %d\n", __FILE__, __LINE__,\
                         static_cast<int>(st__));                               \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

constexpr int DEFAULT_N = 16;
constexpr int DEFAULT_D = 16;
constexpr int DEFAULT_WARMUP_ITERS = 200;
constexpr int DEFAULT_BENCH_ITERS = 2000;

__global__ void transpose_bf16_kernel(const __nv_bfloat16* in, __nv_bfloat16* out,
                                      int rows, int cols) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows || c >= cols) return;
    out[c * rows + r] = in[r * cols + c];
}

__global__ void float_to_bf16_kernel(const float* in, __nv_bfloat16* out, int sz) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < sz) out[idx] = __float2bfloat16(in[idx]);
}

__global__ void bf16_to_float_kernel(const __nv_bfloat16* in, float* out, int sz) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < sz) out[idx] = __bfloat162float(in[idx]);
}

__global__ void softmax_rows_kernel(const float* scores, float* probs, int n) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= n) return;

    __shared__ float smax[256];
    __shared__ float ssum[256];
    float local_max = -1e30f;
    for (int col = tid; col < n; col += blockDim.x) {
        float v = scores[row * n + col];
        local_max = fmaxf(local_max, v);
    }
    smax[tid] = local_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) smax[tid] = fmaxf(smax[tid], smax[tid + stride]);
        __syncthreads();
    }
    float row_max = smax[0];

    float local_sum = 0.0f;
    for (int col = tid; col < n; col += blockDim.x) {
        local_sum += expf(scores[row * n + col] - row_max);
    }
    ssum[tid] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) ssum[tid] += ssum[tid + stride];
        __syncthreads();
    }
    float denom = ssum[0] + 1e-9f;

    for (int col = tid; col < n; col += blockDim.x) {
        probs[row * n + col] = expf(scores[row * n + col] - row_max) / denom;
    }
}

static void print_tensor(const std::vector<float>& t, int rows, int cols,
                         const char* name) {
    std::cout << name << " (" << rows << "x" << cols << ")\n";
    for (int i = 0; i < rows; ++i) {
        std::cout << "[";
        for (int j = 0; j < cols; ++j) {
            std::cout << t[i * cols + j];
            if (j + 1 < cols) std::cout << ", ";
        }
        std::cout << "]\n";
    }
}

int main(int argc, char** argv) {
    int n = DEFAULT_N;
    int d = DEFAULT_D;
    int warmup_iters = DEFAULT_WARMUP_ITERS;
    int bench_iters = DEFAULT_BENCH_ITERS;
    int print_output = 1;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--n") == 0 && i + 1 < argc) n = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--d") == 0 && i + 1 < argc) d = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) warmup_iters = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--iters") == 0 && i + 1 < argc) bench_iters = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--print-output") == 0 && i + 1 < argc) print_output = std::atoi(argv[++i]);
    }
    if (n <= 0 || d <= 0 || warmup_iters < 0 || bench_iters <= 0) {
        std::fprintf(stderr, "Invalid args. Example: --n 16 --d 16 --warmup 200 --iters 2000 --print-output 1\n");
        return EXIT_FAILURE;
    }

    std::vector<float> h_hidden_f32(n * d);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < d; ++j) {
            h_hidden_f32[i * d + j] = std::sinf(i * 0.13f + j * 0.07f) + 0.01f * j;
        }
    }

    float *d_scores = nullptr, *d_probs = nullptr, *d_out_f32 = nullptr;
    __nv_bfloat16 *d_hidden_bf16 = nullptr, *d_hidden_t_bf16 = nullptr, *d_probs_bf16 = nullptr,
                  *d_out_bf16 = nullptr;
    float* d_hidden_f32 = nullptr;

    CHECK_CUDA(cudaMalloc(&d_hidden_f32, n * d * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_hidden_bf16, n * d * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&d_hidden_t_bf16, n * d * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&d_scores, n * n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_probs, n * n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_probs_bf16, n * n * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&d_out_f32, n * d * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_out_bf16, n * d * sizeof(__nv_bfloat16)));

    CHECK_CUDA(cudaMemcpy(d_hidden_f32, h_hidden_f32.data(), n * d * sizeof(float),
                          cudaMemcpyHostToDevice));
    int threads = 256;
    int blocks_hd = (n * d + threads - 1) / threads;
    float_to_bf16_kernel<<<blocks_hd, threads>>>(d_hidden_f32, d_hidden_bf16, n * d);

    dim3 block_t(16, 16);
    dim3 grid_t((d + block_t.x - 1) / block_t.x, (n + block_t.y - 1) / block_t.y);
    transpose_bf16_kernel<<<grid_t, block_t>>>(d_hidden_bf16, d_hidden_t_bf16, n, d);
    CHECK_CUDA(cudaDeviceSynchronize());

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

    const float alpha_qk = 1.0f / std::sqrt(static_cast<float>(d));
    const float alpha_pv = 1.0f;
    const float beta = 0.0f;

    dim3 block_softmax(256);
    dim3 grid_softmax(n);
    int blocks_probs = (n * n + threads - 1) / threads;

    for (int i = 0; i < warmup_iters; ++i) {
        CHECK_CUBLAS(cublasGemmEx(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            n,
            n,
            d,
            &alpha_qk,
            d_hidden_t_bf16,
            CUDA_R_16BF,
            n,
            d_hidden_bf16,
            CUDA_R_16BF,
            d,
            &beta,
            d_scores,
            CUDA_R_32F,
            n,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP));

        softmax_rows_kernel<<<grid_softmax, block_softmax>>>(d_scores, d_probs, n);
        float_to_bf16_kernel<<<blocks_probs, threads>>>(d_probs, d_probs_bf16, n * n);

        CHECK_CUBLAS(cublasGemmEx(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            d,
            n,
            n,
            &alpha_pv,
            d_hidden_bf16,
            CUDA_R_16BF,
            d,
            d_probs_bf16,
            CUDA_R_16BF,
            n,
            &beta,
            d_out_f32,
            CUDA_R_32F,
            d,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        float_to_bf16_kernel<<<blocks_hd, threads>>>(d_out_f32, d_out_bf16, n * d);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < bench_iters; ++i) {
        CHECK_CUBLAS(cublasGemmEx(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            n,
            n,
            d,
            &alpha_qk,
            d_hidden_t_bf16,
            CUDA_R_16BF,
            n,
            d_hidden_bf16,
            CUDA_R_16BF,
            d,
            &beta,
            d_scores,
            CUDA_R_32F,
            n,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP));

        softmax_rows_kernel<<<grid_softmax, block_softmax>>>(d_scores, d_probs, n);
        float_to_bf16_kernel<<<blocks_probs, threads>>>(d_probs, d_probs_bf16, n * n);

        CHECK_CUBLAS(cublasGemmEx(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            d,
            n,
            n,
            &alpha_pv,
            d_hidden_bf16,
            CUDA_R_16BF,
            d,
            d_probs_bf16,
            CUDA_R_16BF,
            n,
            &beta,
            d_out_f32,
            CUDA_R_32F,
            d,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        float_to_bf16_kernel<<<blocks_hd, threads>>>(d_out_f32, d_out_bf16, n * d);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, stop));
    float avg_ms = total_ms / bench_iters;

    std::vector<float> h_out(n * d);
    bf16_to_float_kernel<<<blocks_hd, threads>>>(d_out_bf16, d_out_f32, n * d);
    CHECK_CUDA(cudaMemcpy(h_out.data(), d_out_f32, n * d * sizeof(float),
                          cudaMemcpyDeviceToHost));

    if (print_output) {
        print_tensor(h_out, n, d, "BF16 output tensor");
    }
    std::cout << "N=" << n << "\n";
    std::cout << "D=" << d << "\n";
    std::cout << "BF16_TOTAL_MS=" << total_ms << "\n";
    std::cout << "BF16_AVG_MS=" << avg_ms << "\n";
    std::cout << "BF16_ITERS=" << bench_iters << "\n";

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK_CUDA(cudaFree(d_hidden_f32));
    CHECK_CUDA(cudaFree(d_hidden_bf16));
    CHECK_CUDA(cudaFree(d_hidden_t_bf16));
    CHECK_CUDA(cudaFree(d_scores));
    CHECK_CUDA(cudaFree(d_probs));
    CHECK_CUDA(cudaFree(d_probs_bf16));
    CHECK_CUDA(cudaFree(d_out_f32));
    CHECK_CUDA(cudaFree(d_out_bf16));
    return 0;
}

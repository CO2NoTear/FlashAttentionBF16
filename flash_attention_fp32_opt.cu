#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>

#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                        \
    do {                                                                        \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess) {                                             \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,  \
                         cudaGetErrorString(err__));                             \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

constexpr int DEFAULT_N = 16;
constexpr int DEFAULT_D = 16;
constexpr int DEFAULT_WARMUP_ITERS = 200;
constexpr int DEFAULT_BENCH_ITERS = 2000;

constexpr int TILE_M = 16;
constexpr int TILE_N = 16;
constexpr int TILE_K = 32;

__global__ void qk_matmul_tiled_kernel(const float* q, const float* k, float* scores,
                                       int n, int d, float scale) {
    int row = blockIdx.y * TILE_M + threadIdx.y;
    int col = blockIdx.x * TILE_N + threadIdx.x;

    __shared__ float q_tile[TILE_M][TILE_K];
    __shared__ float k_tile[TILE_N][TILE_K];

    float sum = 0.0f;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    for (int kk = 0; kk < d; kk += TILE_K) {
        for (int idx = tid; idx < TILE_M * TILE_K; idx += blockDim.x * blockDim.y) {
            int r_local = idx / TILE_K;
            int k_local = idx % TILE_K;
            int g_row = blockIdx.y * TILE_M + r_local;
            int g_k = kk + k_local;
            q_tile[r_local][k_local] = (g_row < n && g_k < d) ? q[g_row * d + g_k] : 0.0f;
        }
        for (int idx = tid; idx < TILE_N * TILE_K; idx += blockDim.x * blockDim.y) {
            int c_local = idx / TILE_K;
            int k_local = idx % TILE_K;
            int g_col = blockIdx.x * TILE_N + c_local;
            int g_k = kk + k_local;
            k_tile[c_local][k_local] = (g_col < n && g_k < d) ? k[g_col * d + g_k] : 0.0f;
        }
        __syncthreads();

        if (row < n && col < n) {
            int valid = min(TILE_K, d - kk);
            for (int t = 0; t < valid; ++t) {
                sum += q_tile[threadIdx.y][t] * k_tile[threadIdx.x][t];
            }
        }
        __syncthreads();
    }

    if (row < n && col < n) scores[row * n + col] = sum * scale;
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

__global__ void pv_matmul_tiled_kernel(const float* probs, const float* v, float* out,
                                       int n, int d) {
    int row = blockIdx.y * TILE_M + threadIdx.y;
    int col = blockIdx.x * TILE_N + threadIdx.x;

    __shared__ float p_tile[TILE_M][TILE_K];
    __shared__ float v_tile[TILE_K][TILE_N];
    float sum = 0.0f;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    for (int kk = 0; kk < n; kk += TILE_K) {
        for (int idx = tid; idx < TILE_M * TILE_K; idx += blockDim.x * blockDim.y) {
            int r_local = idx / TILE_K;
            int k_local = idx % TILE_K;
            int g_row = blockIdx.y * TILE_M + r_local;
            int g_k = kk + k_local;
            p_tile[r_local][k_local] = (g_row < n && g_k < n) ? probs[g_row * n + g_k] : 0.0f;
        }
        for (int idx = tid; idx < TILE_K * TILE_N; idx += blockDim.x * blockDim.y) {
            int k_local = idx / TILE_N;
            int c_local = idx % TILE_N;
            int g_k = kk + k_local;
            int g_col = blockIdx.x * TILE_N + c_local;
            v_tile[k_local][c_local] = (g_k < n && g_col < d) ? v[g_k * d + g_col] : 0.0f;
        }
        __syncthreads();

        if (row < n && col < d) {
            int valid = min(TILE_K, n - kk);
            for (int t = 0; t < valid; ++t) {
                sum += p_tile[threadIdx.y][t] * v_tile[t][threadIdx.x];
            }
        }
        __syncthreads();
    }

    if (row < n && col < d) out[row * d + col] = sum;
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

    std::vector<float> h_hidden(n * d);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < d; ++j) {
            h_hidden[i * d + j] = std::sinf(i * 0.13f + j * 0.07f) + 0.01f * j;
        }
    }

    float *d_hidden = nullptr, *d_scores = nullptr, *d_probs = nullptr, *d_out = nullptr;
    CHECK_CUDA(cudaMalloc(&d_hidden, n * d * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_scores, n * n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_probs, n * n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_out, n * d * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_hidden, h_hidden.data(), n * d * sizeof(float),
                          cudaMemcpyHostToDevice));

    dim3 block_mm(TILE_N, TILE_M);
    dim3 grid_qk((n + TILE_N - 1) / TILE_N, (n + TILE_M - 1) / TILE_M);
    dim3 grid_pv((d + TILE_N - 1) / TILE_N, (n + TILE_M - 1) / TILE_M);
    dim3 block_softmax(256);
    dim3 grid_softmax(n);
    const float scale = 1.0f / std::sqrt(static_cast<float>(d));

    for (int i = 0; i < warmup_iters; ++i) {
        qk_matmul_tiled_kernel<<<grid_qk, block_mm>>>(d_hidden, d_hidden, d_scores, n, d, scale);
        softmax_rows_kernel<<<grid_softmax, block_softmax>>>(d_scores, d_probs, n);
        pv_matmul_tiled_kernel<<<grid_pv, block_mm>>>(d_probs, d_hidden, d_out, n, d);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < bench_iters; ++i) {
        qk_matmul_tiled_kernel<<<grid_qk, block_mm>>>(d_hidden, d_hidden, d_scores, n, d, scale);
        softmax_rows_kernel<<<grid_softmax, block_softmax>>>(d_scores, d_probs, n);
        pv_matmul_tiled_kernel<<<grid_pv, block_mm>>>(d_probs, d_hidden, d_out, n, d);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, stop));
    float avg_ms = total_ms / bench_iters;

    std::vector<float> h_out(n * d);
    CHECK_CUDA(cudaMemcpy(h_out.data(), d_out, n * d * sizeof(float), cudaMemcpyDeviceToHost));

    if (print_output) {
        print_tensor(h_out, n, d, "FP32_OPT output tensor");
    }
    std::cout << "N=" << n << "\n";
    std::cout << "D=" << d << "\n";
    std::cout << "FP32_OPT_TOTAL_MS=" << total_ms << "\n";
    std::cout << "FP32_OPT_AVG_MS=" << avg_ms << "\n";
    std::cout << "FP32_OPT_ITERS=" << bench_iters << "\n";

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_hidden));
    CHECK_CUDA(cudaFree(d_scores));
    CHECK_CUDA(cudaFree(d_probs));
    CHECK_CUDA(cudaFree(d_out));
    return 0;
}

# Flash Attention FP32 vs BF16 对比报告（中文）

## 1. 报告概述

本报告对同一套 Flash Attention Demo 在 `FP32` 与 `BF16` 两种精度下的运行效率进行对比分析。  
测试硬件为 RTX 3060（Ampere 架构），`BF16` 路径使用 `cuBLAS + Tensor Core GEMM`。

- 时间预算（time budget）：`60s`
- Warmup iterations：`20`
- 测试配置数量：`10`
- 最优 BF16 优势：`17.2275x`（`N=1024, D=128, ITERS=30`）

## 2. 可视化结果

![Benchmark Plot](report/benchmark_plot.svg)

图中柱状对比的是每个配置下的平均单步耗时（`avg ms`）。可以看到随着规模增大，`BF16` 的优势逐步扩大。

## 3. 详细数据

| N | D | ITERS | FP32 avg (ms) | BF16 avg (ms) | Speedup (FP32/BF16) |
|---:|---:|---:|---:|---:|---:|
| 256 | 64 | 400 | 0.076931 | 0.020262 | 3.7967x |
| 256 | 128 | 300 | 0.138185 | 0.022354 | 6.1817x |
| 384 | 64 | 250 | 0.161456 | 0.030163 | 5.3528x |
| 384 | 128 | 200 | 0.295849 | 0.029174 | 10.1409x |
| 512 | 64 | 160 | 0.258054 | 0.036691 | 7.0331x |
| 512 | 128 | 120 | 0.512222 | 0.039757 | 12.8839x |
| 768 | 64 | 80 | 0.548506 | 0.065216 | 8.4106x |
| 768 | 128 | 60 | 1.033470 | 0.090573 | 11.4104x |
| 1024 | 64 | 40 | 0.956646 | 0.093747 | 10.2045x |
| 1024 | 128 | 30 | 1.827600 | 0.106086 | 17.2275x |

## 4. 结论分析

在本实现中，`BF16` 路径在所有已测配置上都优于 `FP32`，且在较大 `N/D` 下优势显著。  
主要原因是 `BF16` 主计算采用 `Tensor Core GEMM`，在矩阵规模提升后可以更充分发挥 Ampere 架构吞吐能力。

需要说明的是：当前 `FP32` 路径是自定义 CUDA kernel baseline，而非 `cuBLAS FP32` baseline，因此本报告反映的是“当前两条实现路径”的效率差异。

## 5. 复现实验

在项目根目录执行：

```bash
MAX_SECONDS=60 WARMUP=20 python3 generate_benchmark_report.py
```

将生成：

- `benchmark_report.md`（英文报告）
- `benchmark_report_zh.md`（本中文报告）
- `report/benchmark_results.csv`
- `report/benchmark_plot.svg`

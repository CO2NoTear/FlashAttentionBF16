# Flash Attention FP32 vs BF16 Report

## Summary

- Time budget: 60s
- Warmup iterations: 20
- Tested configurations: 10
- Best BF16 advantage: **17.2275x** at `N=1024, D=128, ITERS=30`

## Plot

![Benchmark Plot](report/benchmark_plot.svg)

## Detailed Results

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

## Notes

- In this implementation, BF16 path uses cuBLAS Tensor Core GEMM and shows clear advantages at larger matrix sizes.
- FP32 path remains a custom kernel baseline, so this report compares current implementation paths.

Generated files: `report/benchmark_results.csv`, `report/benchmark_plot.svg`.
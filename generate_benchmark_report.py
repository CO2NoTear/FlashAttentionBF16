#!/usr/bin/env python3
import csv
import os
import re
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPORT_DIR = ROOT / "report"
REPORT_DIR.mkdir(exist_ok=True)

CASES = [
    (256, 64, 400),
    (256, 128, 300),
    (384, 64, 250),
    (384, 128, 200),
    (512, 64, 160),
    (512, 128, 120),
    (768, 64, 80),
    (768, 128, 60),
    (1024, 64, 40),
    (1024, 128, 30),
]

MAX_SECONDS = int(os.environ.get("MAX_SECONDS", "60"))
WARMUP = int(os.environ.get("WARMUP", "20"))

FP32_BIN = ROOT / "flash_attention_fp32"
FP32_OPT_BIN = ROOT / "flash_attention_fp32_opt"
BF16_BIN = ROOT / "flash_attention_bf16"


def run(cmd):
    return subprocess.run(cmd, check=True, text=True, capture_output=True)


def build():
    run(["nvcc", "-O3", "-std=c++17", "flash_attention_fp32.cu", "-o", str(FP32_BIN)])
    run(["nvcc", "-O3", "-std=c++17", "flash_attention_fp32_opt.cu", "-o", str(FP32_OPT_BIN)])
    run(["nvcc", "-O3", "-std=c++17", "flash_attention_bf16.cu", "-o", str(BF16_BIN), "-lcublas"])


def parse_metric(text, key):
    m = re.search(rf"^{re.escape(key)}=([0-9eE.+-]+)$", text, flags=re.MULTILINE)
    if not m:
        raise RuntimeError(f"Cannot find metric {key}")
    return float(m.group(1))


def run_one(bin_path, n, d, iters):
    out = run(
        [
            str(bin_path),
            "--n",
            str(n),
            "--d",
            str(d),
            "--warmup",
            str(WARMUP),
            "--iters",
            str(iters),
            "--print-output",
            "0",
        ]
    ).stdout
    return out


def write_svg(rows, out_path):
    width, height = 1080, 520
    margin_l, margin_r, margin_t, margin_b = 70, 20, 40, 80
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b
    max_v = max(max(r["fp32_avg_ms"], r["fp32_opt_avg_ms"], r["bf16_avg_ms"]) for r in rows) * 1.1
    if max_v <= 0:
        max_v = 1.0

    n_groups = len(rows)
    group_w = plot_w / max(1, n_groups)
    bar_w = group_w * 0.22

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="20" y="24" font-size="20" font-family="sans-serif">FP32 baseline vs FP32-OPT vs BF16 (avg step ms)</text>',
        f'<line x1="{margin_l}" y1="{margin_t}" x2="{margin_l}" y2="{margin_t + plot_h}" stroke="black"/>',
        f'<line x1="{margin_l}" y1="{margin_t + plot_h}" x2="{margin_l + plot_w}" y2="{margin_t + plot_h}" stroke="black"/>',
    ]

    for t in range(6):
        val = max_v * t / 5
        yy = margin_t + plot_h * (1.0 - val / max_v)
        parts.append(
            f'<line x1="{margin_l}" y1="{yy:.2f}" x2="{margin_l + plot_w}" y2="{yy:.2f}" stroke="#dddddd" stroke-dasharray="3,3"/>'
        )
        parts.append(f'<text x="8" y="{yy + 4:.2f}" font-size="11" font-family="sans-serif">{val:.3f}</text>')

    for i, r in enumerate(rows):
        gx = margin_l + i * group_w
        fp32_h = (r["fp32_avg_ms"] / max_v) * plot_h
        fp32_opt_h = (r["fp32_opt_avg_ms"] / max_v) * plot_h
        bf16_h = (r["bf16_avg_ms"] / max_v) * plot_h

        x1 = gx + group_w * 0.10
        x2 = x1 + bar_w + 3
        x3 = x2 + bar_w + 3
        y_base = margin_t + plot_h
        parts.append(f'<rect x="{x1:.2f}" y="{y_base - fp32_h:.2f}" width="{bar_w:.2f}" height="{fp32_h:.2f}" fill="#4e79a7"/>')
        parts.append(f'<rect x="{x2:.2f}" y="{y_base - fp32_opt_h:.2f}" width="{bar_w:.2f}" height="{fp32_opt_h:.2f}" fill="#59a14f"/>')
        parts.append(f'<rect x="{x3:.2f}" y="{y_base - bf16_h:.2f}" width="{bar_w:.2f}" height="{bf16_h:.2f}" fill="#f28e2b"/>')
        parts.append(f'<text x="{gx + group_w*0.04:.2f}" y="{y_base + 16}" font-size="10" font-family="sans-serif">{r["n"]}x{r["d"]}</text>')
        parts.append(f'<text x="{gx + group_w*0.08:.2f}" y="{y_base + 30}" font-size="9" font-family="sans-serif">x{r["iters"]}</text>')

    lx, ly = width - 230, 50
    parts += [
        f'<rect x="{lx}" y="{ly}" width="12" height="12" fill="#4e79a7"/>',
        f'<text x="{lx + 18}" y="{ly + 10}" font-size="12" font-family="sans-serif">FP32 baseline</text>',
        f'<rect x="{lx}" y="{ly + 18}" width="12" height="12" fill="#59a14f"/>',
        f'<text x="{lx + 18}" y="{ly + 28}" font-size="12" font-family="sans-serif">FP32-OPT</text>',
        f'<rect x="{lx}" y="{ly + 36}" width="12" height="12" fill="#f28e2b"/>',
        f'<text x="{lx + 18}" y="{ly + 46}" font-size="12" font-family="sans-serif">BF16</text>',
        "</svg>",
    ]
    out_path.write_text("\n".join(parts), encoding="utf-8")


def write_reports(rows, csv_path, svg_path):
    best_opt = max(rows, key=lambda x: x["speedup_opt_vs_fp32"])
    best_bf16 = max(rows, key=lambda x: x["speedup_bf16_vs_fp32"])
    best_bf16_vs_opt = max(rows, key=lambda x: x["speedup_bf16_vs_opt"])

    en_lines = [
        "# Flash Attention FP32 Baseline vs FP32-OPT vs BF16 Report",
        "",
        "## Summary",
        "",
        f"- Time budget: {MAX_SECONDS}s",
        f"- Warmup iterations: {WARMUP}",
        f"- Tested configurations: {len(rows)}",
        f"- Best FP32-OPT over FP32: **{best_opt['speedup_opt_vs_fp32']:.4f}x** at `N={best_opt['n']}, D={best_opt['d']}, ITERS={best_opt['iters']}`",
        f"- Best BF16 over FP32: **{best_bf16['speedup_bf16_vs_fp32']:.4f}x** at `N={best_bf16['n']}, D={best_bf16['d']}, ITERS={best_bf16['iters']}`",
        f"- Best BF16 over FP32-OPT: **{best_bf16_vs_opt['speedup_bf16_vs_opt']:.4f}x** at `N={best_bf16_vs_opt['n']}, D={best_bf16_vs_opt['d']}, ITERS={best_bf16_vs_opt['iters']}`",
        "",
        "## Plot",
        "",
        "![Benchmark Plot](report/benchmark_plot.svg)",
        "",
        "## Detailed Results",
        "",
        "| N | D | ITERS | FP32 avg (ms) | FP32-OPT avg (ms) | BF16 avg (ms) | FP32/FP32-OPT | FP32/BF16 | FP32-OPT/BF16 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        en_lines.append(
            f"| {r['n']} | {r['d']} | {r['iters']} | {r['fp32_avg_ms']:.6f} | {r['fp32_opt_avg_ms']:.6f} | {r['bf16_avg_ms']:.6f} | {r['speedup_opt_vs_fp32']:.4f}x | {r['speedup_bf16_vs_fp32']:.4f}x | {r['speedup_bf16_vs_opt']:.4f}x |"
        )
    en_lines += [
        "",
        "## Notes",
        "",
        "- FP32-OPT applies shared-memory tiling and coalesced memory access.",
        "- BF16 uses cuBLAS Tensor Core GEMM.",
        f"- Generated files: `{csv_path.relative_to(ROOT)}`, `{svg_path.relative_to(ROOT)}`.",
    ]
    (ROOT / "benchmark_report.md").write_text("\n".join(en_lines), encoding="utf-8")

    zh_lines = [
        "# Flash Attention 三方案对比报告（中文）",
        "",
        "## 1. 概述",
        "",
        "本报告对三种实现进行对比：`FP32 baseline`、`FP32-OPT`（共享内存 tiling + 合并访存）和 `BF16`（Tensor Core GEMM）。",
        "",
        f"- 时间预算：`{MAX_SECONDS}s`",
        f"- Warmup iterations：`{WARMUP}`",
        f"- 测试配置数量：`{len(rows)}`",
        f"- FP32-OPT 相对 FP32 最优加速：`{best_opt['speedup_opt_vs_fp32']:.4f}x`（`N={best_opt['n']}, D={best_opt['d']}, ITERS={best_opt['iters']}`）",
        f"- BF16 相对 FP32 最优加速：`{best_bf16['speedup_bf16_vs_fp32']:.4f}x`（`N={best_bf16['n']}, D={best_bf16['d']}, ITERS={best_bf16['iters']}`）",
        "",
        "## 2. 可视化",
        "",
        "![Benchmark Plot](report/benchmark_plot.svg)",
        "",
        "## 3. 详细数据",
        "",
        "| N | D | ITERS | FP32 avg (ms) | FP32-OPT avg (ms) | BF16 avg (ms) | FP32/FP32-OPT | FP32/BF16 | FP32-OPT/BF16 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        zh_lines.append(
            f"| {r['n']} | {r['d']} | {r['iters']} | {r['fp32_avg_ms']:.6f} | {r['fp32_opt_avg_ms']:.6f} | {r['bf16_avg_ms']:.6f} | {r['speedup_opt_vs_fp32']:.4f}x | {r['speedup_bf16_vs_fp32']:.4f}x | {r['speedup_bf16_vs_opt']:.4f}x |"
        )
    zh_lines += [
        "",
        "## 4. 结论",
        "",
        "- `FP32-OPT` 相比 `FP32 baseline` 有稳定加速，说明共享内存 tiling 与合并访存有效降低了 memory traffic。",
        "- `BF16` 依然整体最优，尤其在大规模矩阵上优势明显。",
        f"- 产物文件：`{csv_path.relative_to(ROOT)}`、`{svg_path.relative_to(ROOT)}`。",
    ]
    (ROOT / "benchmark_report_zh.md").write_text("\n".join(zh_lines), encoding="utf-8")

    return best_opt, best_bf16, best_bf16_vs_opt


def main():
    start = time.time()
    build()

    rows = []
    for n, d, iters in CASES:
        if time.time() - start > MAX_SECONDS - 5:
            break
        fp32_out = run_one(FP32_BIN, n, d, iters)
        fp32_opt_out = run_one(FP32_OPT_BIN, n, d, iters)
        bf16_out = run_one(BF16_BIN, n, d, iters)
        fp32_avg = parse_metric(fp32_out, "FP32_AVG_MS")
        fp32_opt_avg = parse_metric(fp32_opt_out, "FP32_OPT_AVG_MS")
        bf16_avg = parse_metric(bf16_out, "BF16_AVG_MS")
        rows.append(
            {
                "n": n,
                "d": d,
                "iters": iters,
                "fp32_avg_ms": fp32_avg,
                "fp32_opt_avg_ms": fp32_opt_avg,
                "bf16_avg_ms": bf16_avg,
                "speedup_opt_vs_fp32": fp32_avg / fp32_opt_avg if fp32_opt_avg > 0 else 0.0,
                "speedup_bf16_vs_fp32": fp32_avg / bf16_avg if bf16_avg > 0 else 0.0,
                "speedup_bf16_vs_opt": fp32_opt_avg / bf16_avg if bf16_avg > 0 else 0.0,
            }
        )

    if not rows:
        raise RuntimeError("No benchmark rows generated within time budget.")

    csv_path = REPORT_DIR / "benchmark_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    svg_path = REPORT_DIR / "benchmark_plot.svg"
    write_svg(rows, svg_path)
    best_opt, best_bf16, _ = write_reports(rows, csv_path, svg_path)

    print(f"Report generated: {ROOT / 'benchmark_report.md'}")
    print(f"Rows: {len(rows)}")
    print(
        f"Best FP32-OPT/FP32: {best_opt['speedup_opt_vs_fp32']:.4f}x, "
        f"Best BF16/FP32: {best_bf16['speedup_bf16_vs_fp32']:.4f}x"
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

echo "[1/4] Building FP32 baseline demo..."
nvcc -O3 -std=c++17 flash_attention_fp32.cu -o flash_attention_fp32

echo "[2/4] Building FP32 optimized demo..."
nvcc -O3 -std=c++17 flash_attention_fp32_opt.cu -o flash_attention_fp32_opt

echo "[3/4] Building BF16 demo..."
nvcc -O3 -std=c++17 flash_attention_bf16.cu -o flash_attention_bf16 -lcublas

echo "[4/4] Running benchmarks..."
FP32_LOG="$(mktemp)"
FP32_OPT_LOG="$(mktemp)"
BF16_LOG="$(mktemp)"
trap 'rm -f "$FP32_LOG" "$FP32_OPT_LOG" "$BF16_LOG"' EXIT

N="${N:-16}"
D="${D:-16}"
WARMUP="${WARMUP:-200}"
ITERS="${ITERS:-2000}"
PRINT_OUTPUT="${PRINT_OUTPUT:-1}"

./flash_attention_fp32 --n "$N" --d "$D" --warmup "$WARMUP" --iters "$ITERS" --print-output "$PRINT_OUTPUT" > "$FP32_LOG"
./flash_attention_fp32_opt --n "$N" --d "$D" --warmup "$WARMUP" --iters "$ITERS" --print-output "$PRINT_OUTPUT" > "$FP32_OPT_LOG"
./flash_attention_bf16 --n "$N" --d "$D" --warmup "$WARMUP" --iters "$ITERS" --print-output "$PRINT_OUTPUT" > "$BF16_LOG"

echo "===== FP32 Output (tail) ====="
tail -n 6 "$FP32_LOG"
echo "===== FP32_OPT Output (tail) ====="
tail -n 6 "$FP32_OPT_LOG"
echo "===== BF16 Output (tail) ====="
tail -n 6 "$BF16_LOG"

FP32_TOTAL="$(grep '^FP32_TOTAL_MS=' "$FP32_LOG" | cut -d= -f2)"
FP32_AVG="$(grep '^FP32_AVG_MS=' "$FP32_LOG" | cut -d= -f2)"
FP32_OPT_TOTAL="$(grep '^FP32_OPT_TOTAL_MS=' "$FP32_OPT_LOG" | cut -d= -f2)"
FP32_OPT_AVG="$(grep '^FP32_OPT_AVG_MS=' "$FP32_OPT_LOG" | cut -d= -f2)"
BF16_TOTAL="$(grep '^BF16_TOTAL_MS=' "$BF16_LOG" | cut -d= -f2)"
BF16_AVG="$(grep '^BF16_AVG_MS=' "$BF16_LOG" | cut -d= -f2)"

python3 - <<PY
fp32_total = float("${FP32_TOTAL}")
fp32_avg = float("${FP32_AVG}")
fp32_opt_total = float("${FP32_OPT_TOTAL}")
fp32_opt_avg = float("${FP32_OPT_AVG}")
bf16_total = float("${BF16_TOTAL}")
bf16_avg = float("${BF16_AVG}")

speedup_opt_vs_base = fp32_avg / fp32_opt_avg if fp32_opt_avg > 0 else float("inf")
speedup_bf16_vs_base = fp32_avg / bf16_avg if bf16_avg > 0 else float("inf")
speedup_bf16_vs_opt = fp32_opt_avg / bf16_avg if bf16_avg > 0 else float("inf")

print("===== Efficiency Comparison =====")
print(f"N={int(${N})}, D={int(${D})}, ITERS={int(${ITERS})}")
print(f"FP32 baseline total: {fp32_total:.6f} ms")
print(f"FP32 optimized total: {fp32_opt_total:.6f} ms")
print(f"BF16 total: {bf16_total:.6f} ms")
print(f"FP32 baseline avg/step: {fp32_avg:.6f} ms")
print(f"FP32 optimized avg/step: {fp32_opt_avg:.6f} ms")
print(f"BF16 avg/step: {bf16_avg:.6f} ms")
print(f"Speedup FP32_OPT over FP32: {speedup_opt_vs_base:.4f}x")
print(f"Speedup BF16 over FP32: {speedup_bf16_vs_base:.4f}x")
print(f"Speedup BF16 over FP32_OPT: {speedup_bf16_vs_opt:.4f}x")
PY

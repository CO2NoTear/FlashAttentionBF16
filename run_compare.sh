#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

echo "[1/3] Building FP32 demo..."
nvcc -O3 -std=c++17 flash_attention_fp32.cu -o flash_attention_fp32

echo "[2/3] Building BF16 demo..."
nvcc -O3 -std=c++17 flash_attention_bf16.cu -o flash_attention_bf16 -lcublas

echo "[3/3] Running benchmarks..."
FP32_LOG="$(mktemp)"
BF16_LOG="$(mktemp)"
trap 'rm -f "$FP32_LOG" "$BF16_LOG"' EXIT

N="${N:-16}"
D="${D:-16}"
WARMUP="${WARMUP:-200}"
ITERS="${ITERS:-2000}"
PRINT_OUTPUT="${PRINT_OUTPUT:-1}"

./flash_attention_fp32 --n "$N" --d "$D" --warmup "$WARMUP" --iters "$ITERS" --print-output "$PRINT_OUTPUT" > "$FP32_LOG"
./flash_attention_bf16 --n "$N" --d "$D" --warmup "$WARMUP" --iters "$ITERS" --print-output "$PRINT_OUTPUT" > "$BF16_LOG"

echo "===== FP32 Output (tail) ====="
tail -n 6 "$FP32_LOG"
echo "===== BF16 Output (tail) ====="
tail -n 6 "$BF16_LOG"

FP32_TOTAL="$(grep '^FP32_TOTAL_MS=' "$FP32_LOG" | cut -d= -f2)"
FP32_AVG="$(grep '^FP32_AVG_MS=' "$FP32_LOG" | cut -d= -f2)"
BF16_TOTAL="$(grep '^BF16_TOTAL_MS=' "$BF16_LOG" | cut -d= -f2)"
BF16_AVG="$(grep '^BF16_AVG_MS=' "$BF16_LOG" | cut -d= -f2)"

python3 - <<PY
fp32_total = float("${FP32_TOTAL}")
fp32_avg = float("${FP32_AVG}")
bf16_total = float("${BF16_TOTAL}")
bf16_avg = float("${BF16_AVG}")

speedup_total = fp32_total / bf16_total if bf16_total > 0 else float("inf")
speedup_avg = fp32_avg / bf16_avg if bf16_avg > 0 else float("inf")

print("===== Efficiency Comparison =====")
print(f"N={int(${N})}, D={int(${D})}, ITERS={int(${ITERS})}")
print(f"FP32 total: {fp32_total:.6f} ms")
print(f"BF16 total: {bf16_total:.6f} ms")
print(f"FP32 avg/step: {fp32_avg:.6f} ms")
print(f"BF16 avg/step: {bf16_avg:.6f} ms")
print(f"Total speedup (FP32/BF16): {speedup_total:.4f}x")
print(f"Avg-step speedup (FP32/BF16): {speedup_avg:.4f}x")
PY

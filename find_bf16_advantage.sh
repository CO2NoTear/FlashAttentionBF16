#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

MAX_SECONDS="${MAX_SECONDS:-60}"
WARMUP="${WARMUP:-20}"
PRINT_OUTPUT=0

echo "[1/3] Building FP32/BF16 demos..."
nvcc -O3 -std=c++17 flash_attention_fp32.cu -o flash_attention_fp32
nvcc -O3 -std=c++17 flash_attention_bf16.cu -o flash_attention_bf16 -lcublas

start_ts="$(date +%s)"

declare -a CASES=(
  "256 64 400"
  "256 128 300"
  "384 64 250"
  "384 128 200"
  "512 64 160"
  "512 128 120"
  "768 64 80"
  "768 128 60"
  "1024 64 40"
  "1024 128 30"
)

best_speedup=0
best_cfg=""
best_fp32=0
best_bf16=0

echo "[2/3] Searching BF16-favorable configs (time budget ${MAX_SECONDS}s)..."
printf "%-12s %-12s %-14s %-14s %-10s\n" "N" "D" "FP32_AVG(ms)" "BF16_AVG(ms)" "SPEEDUP"

for cfg in "${CASES[@]}"; do
  read -r n d iters <<< "${cfg}"
  now="$(date +%s)"
  elapsed="$((now - start_ts))"
  if (( elapsed >= MAX_SECONDS - 5 )); then
    echo "Time budget nearly reached (${elapsed}s), stopping scan."
    break
  fi

  fp32_log="$(mktemp)"
  bf16_log="$(mktemp)"

  ./flash_attention_fp32 --n "$n" --d "$d" --warmup "$WARMUP" --iters "$iters" --print-output "$PRINT_OUTPUT" > "$fp32_log"
  ./flash_attention_bf16 --n "$n" --d "$d" --warmup "$WARMUP" --iters "$iters" --print-output "$PRINT_OUTPUT" > "$bf16_log"

  fp32_avg="$(grep '^FP32_AVG_MS=' "$fp32_log" | cut -d= -f2)"
  bf16_avg="$(grep '^BF16_AVG_MS=' "$bf16_log" | cut -d= -f2)"
  speedup="$(python3 - <<PY
fp32 = float("${fp32_avg}")
bf16 = float("${bf16_avg}")
print(f"{(fp32 / bf16) if bf16 > 0 else 0.0:.4f}")
PY
)"
  printf "%-12s %-12s %-14.6f %-14.6f %-10s\n" "$n" "$d" "$fp32_avg" "$bf16_avg" "$speedup"

  better="$(python3 - <<PY
cur = float("${speedup}")
best = float("${best_speedup}")
print(1 if cur > best else 0)
PY
)"
  if [[ "$better" == "1" ]]; then
    best_speedup="$speedup"
    best_cfg="N=${n}, D=${d}, ITERS=${iters}"
    best_fp32="$fp32_avg"
    best_bf16="$bf16_avg"
  fi

  rm -f "$fp32_log" "$bf16_log"
done

echo "[3/3] Summary"
if python3 - <<PY
print(1 if float("${best_speedup}") > 1.0 else 0)
PY
then
  :
fi

is_advantage="$(python3 - <<PY
print(1 if float("${best_speedup}") > 1.0 else 0)
PY
)"

if [[ "${is_advantage}" == "1" ]]; then
  echo "Found BF16 advantage config: ${best_cfg}"
  echo "FP32 avg: ${best_fp32} ms"
  echo "BF16 avg: ${best_bf16} ms"
  echo "Speedup (FP32/BF16): ${best_speedup}x"
else
  echo "No BF16 advantage found in current scan set under ${MAX_SECONDS}s."
  echo "Best observed config: ${best_cfg}"
  echo "Best speedup (FP32/BF16): ${best_speedup}x"
fi

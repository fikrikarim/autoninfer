#!/bin/bash
# NInfer research-GPU health probe.
#
# Usage: tools/gpu_health.sh <physical-gpu-index>
#
# The verdict is based on *executed, host-timed, high-occupancy* work, because
# nvidia-smi counters and low-occupancy kernels are known to mislead on this
# box (2026-08-18: a wedged GPU reported 100% SM busy, healthy clocks, and no
# throttle flags; a single-warp clock64() spin ran at ~2.9 MHz whether the card
# was wedged or fully healthy, and was useless as an indicator).
#
# The probe compiles a small CUDA kernel on the fly (nvcc, ~5 s) and measures
# two workloads on the target GPU:
#
#   bw_triad - 20 x (c = a + 0.5*b) over three 256 MiB buffers, 172k blocks.
#              Healthy 5090: ~1.4-1.7 TB/s. Wedged state (observed): ~52 GiB/s.
#              Pass: >= 500 GiB/s (10x above the observed wedged value).
#   fma      - all-SM FP32 FMA (multiProcessorCount*16 blocks x 256 threads,
#              500k dependent-FMA iterations), ~1392 GFLOP.
#              Healthy 5090: ~100-120 TFLOP/s. Pass: >= 20 TFLOP/s.
#
# It also reports the nvidia-smi wedge signature (SM pinned at 100% with no
# compute processes and near-zero memory) as informational context.
set -euo pipefail

idx="${1:-}"
if [[ -z "$idx" ]]; then
  echo "usage: $(basename "$0") <physical-gpu-index>" >&2
  exit 2
fi

# 1. nvidia-smi must see the GPU
if ! nvidia-smi --query-gpu=name --format=csv,noheader,nounits -i "$idx" >/dev/null 2>&1; then
  echo "GPU $idx: NOT VISIBLE to nvidia-smi"
  exit 1
fi

# Informational wedge signature (never sufficient on its own; the executed-work verdict rules).
util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$idx" | tr -d ' ')
mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$idx" | tr -d ' ')
signature=""
if [[ "$util" -ge 100 && "$mem" -lt 100 ]]; then
  signature=" [nvidia-smi shows the wedge signature: SM=${util}%, ${mem} MiB used]"
fi

# 2. The executed-work verdict. (nvcc needs a .cu-suffixed source file.)
src="${TMPDIR:-/tmp}/gpu_health_$$.cu"
bin="${TMPDIR:-/tmp}/gpu_health_$$/probe"
mkdir -p "$(dirname "$bin")"
trap 'rm -f "$src" "$bin"; rmdir "$(dirname "$bin")" 2>/dev/null || true' EXIT
cat > "$src" <<'CU'
#include <cstdio>
#include <cuda_runtime.h>
#include <chrono>
__global__ void triad(float* a, float* b, float* c, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i] * 0.5f;
}
// all-SM FP32 FMA; two dependent fmaf per iter per thread; host-timed.
__global__ void fma_full(float* x, int iters) {
  float a = x[threadIdx.x & 1023] + 1.0f;
  float b = 1.0f + a * 1e-7f;
  for (int i = 0; i < iters; i++) { a = fmaf(a, b, 1e-9f); b = fmaf(b, a, 1e-9f); }
  if (a == 12345.6789f) x[0] = a;   // sink, no DCE
}
int main() {
  cudaDeviceProp p;
  cudaGetDeviceProperties(&p, 0);
  size_t n = 256ull * 1024 * 1024 / 4;
  float *a, *b, *c;
  cudaMalloc(&a, n * 4);
  cudaMalloc(&b, n * 4);
  cudaMalloc(&c, n * 4);
  cudaMemset(a, 1, n * 4);
  cudaMemset(b, 2, n * 4);
  triad<<<(n + 255) / 256, 256>>>(a, b, c, n);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("CUDA_ERROR %s\n", cudaGetErrorString(err));
    return 1;
  }
  auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < 20; i++) triad<<<(n + 255) / 256, 256>>>(a, b, c, n);
  cudaDeviceSynchronize();
  auto t1 = std::chrono::high_resolution_clock::now();
  double bytes = 20.0 * n * 4 * 3;
  double sec = std::chrono::duration<double>(t1 - t0).count();
  printf("bw_triad: %.1f GiB/s\n", bytes / sec / 1073741824.0);
  float* x;
  cudaMalloc(&x, 1024 * 4);
  int blocks = p.multiProcessorCount * 16, threads = 256, iters = 500000;
  fma_full<<<blocks, threads>>>(x, 1000);
  cudaDeviceSynchronize();
  t0 = std::chrono::high_resolution_clock::now();
  fma_full<<<blocks, threads>>>(x, iters);
  cudaDeviceSynchronize();
  t1 = std::chrono::high_resolution_clock::now();
  sec = std::chrono::duration<double>(t1 - t0).count();
  double flop = (double)blocks * threads * iters * 2.0 * 2.0;
  printf("fma_tflops: %.2f\n", flop / sec / 1e12);
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA_ERROR %s\n", cudaGetErrorString(err));
    return 1;
  }
  return 0;
}
CU
nvcc -O2 -arch=sm_120a "$src" -o "$bin" >/dev/null 2>&1 || { echo "GPU $idx: nvcc failed to build the probe"; exit 2; }

out=$(CUDA_VISIBLE_DEVICES="$idx" "$bin")
bw=$(printf '%s' "$out" | sed -n 's/^bw_triad: \([0-9.]*\) GiB\/s$/\1/p')
fma=$(printf '%s' "$out" | sed -n 's/^fma_tflops: \([0-9.]*\)$/\1/p')
if [[ -z "$bw" || -z "$fma" ]]; then
  echo "GPU $idx: probe failed to run: $out"
  exit 2
fi
bw_pass=$(awk -v v="$bw" 'BEGIN { print (v >= 500.0) ? 1 : 0 }')
fma_pass=$(awk -v v="$fma" 'BEGIN { print (v >= 20.0) ? 1 : 0 }')

echo "GPU $idx: triad=${bw} GiB/s (pass >=500), fma=${fma} TFLOP/s (pass >=20)${signature}"
if [[ "$bw_pass" == 1 && "$fma_pass" == 1 ]]; then
  echo "GPU $idx: HEALTHY"
  exit 0
fi
echo "GPU $idx: DEGRADED - executed work far below rated speed; check for a wedged GPU (restart if it persists)"
exit 1
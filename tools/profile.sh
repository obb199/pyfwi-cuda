#!/usr/bin/env bash
# Profiling and memory-checking recipes for the CUDA work (roadmap step 0.8).
#
# Usage:   tools/profile.sh <tool> <python script> [script arguments...]
#
# Tools:
#   nsys       Nsight Systems timeline (CUDA API, kernels, copies, NVTX ranges): where the GPU is
#              idle and how many launches and copies there are. Writes <out>.nsys-rep and prints
#              the kernel and copy summaries.
#   ncu        Nsight Compute: per-kernel metrics (duration, achieved memory throughput as a
#              percentage of the peak). Needs access to the GPU performance counters, which the
#              driver restricts to root by default (RmProfilingAdminOnly: 1); see the roadmap,
#              "Pending actions for the project owner".
#   memcheck   compute-sanitizer: out-of-bounds and misaligned accesses of every CUDA kernel.
#   racecheck  compute-sanitizer: shared-memory data races.
#   initcheck  compute-sanitizer: reads of uninitialised device memory.
#   synccheck  compute-sanitizer: invalid uses of synchronisation primitives.
#
# The compute-sanitizer tools exit with code 99 when they find an error, so they can be used
# in CI. OpenCL kernels (PyFWI 0.1.10) are not visible to these tools: profile them with
# tools/opencl_profile.py, and check their memory accesses with test/oob_guard.py.
#
# Variables: OUT   output prefix for nsys/ncu reports (default: profile-<tool>)
#            PY    Python interpreter (default: ../envs/dev/bin/python next to the repository)
#
# Used by: developers, and test/test_profiling.py (self-tests of the tools on tiny CuPy kernels).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-$(dirname "$REPO")/envs/dev/bin/python}"
tool="${1:-}"
if [[ -z "$tool" || $# -lt 2 ]]; then
  echo "usage: $0 {nsys|ncu|memcheck|racecheck|initcheck|synccheck} <python script> [args...]" >&2
  exit 2
fi
shift
OUT="${OUT:-profile-$tool}"

case "$tool" in
  nsys)
    nsys profile --force-overwrite true --trace cuda,nvtx --output "$OUT" "$PY" "$@"
    nsys stats --report cuda_gpu_kern_sum,cuda_gpu_mem_time_sum "$OUT.nsys-rep"
    ;;
  ncu)
    ncu --force-overwrite --export "$OUT" \
        --metrics gpu__time_duration.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed \
        "$PY" "$@"
    ;;
  memcheck|racecheck|initcheck|synccheck)
    compute-sanitizer --tool "$tool" --error-exitcode 99 "$PY" "$@"
    ;;
  *)
    echo "unknown tool: $tool" >&2
    exit 2
    ;;
esac

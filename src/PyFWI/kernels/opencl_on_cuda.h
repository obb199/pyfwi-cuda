// OpenCL C on CUDA: the few OpenCL constructs of PyFWI's kernels, defined for CUDA C++ (NVRTC).
//
// PyFWI's kernels (elastic.cl, elastic_crosswell.cl, elastic_surface.cl) are OpenCL C. The CUDA
// backend (PyFWI/backends/cuda.py, roadmap step 2.3) compiles the same files with NVRTC after this
// header, so that both backends run one source:
//
// * __kernel becomes an exported CUDA kernel, and __global (the OpenCL global address space)
//   disappears: every pointer of a CUDA kernel already points to global memory;
// * get_global_id(0) is the row (z, slow index of the arrays) and get_global_id(1) the column
//   (x, contiguous in memory), so that the column follows threadIdx.x: neighbouring threads read
//   neighbouring addresses. The backend launches blocks of 32 columns x 8 rows.
//
// The grid of blocks covers the OpenCL global size (Nz rows, Nx columns) rounded up to whole
// blocks; the backend adds, at the start of every kernel, a return for the work items beyond it, so
// that exactly the work items of the OpenCL launch run.
#define __kernel extern "C" __global__
#define __global
#define __local __shared__

__device__ __forceinline__ int get_global_id(int d) {
    return d == 0 ? (int)(blockIdx.y * blockDim.y + threadIdx.y)
                  : (int)(blockIdx.x * blockDim.x + threadIdx.x);
}

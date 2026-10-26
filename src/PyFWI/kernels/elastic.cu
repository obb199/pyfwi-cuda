// Native CUDA kernels of PyFWI for the fixed numeric mode (roadmap Phase 3, from step 3.2).
//
// The CUDA backend runs these kernels instead of the OpenCL files (elastic.cl...) when the numeric
// mode is "fixed" (PyFWI/backends/cuda.py, PYFWI_CUDA_KERNELS). They compute what the fixed branches
// of the OpenCL files compute, written for CUDA so that the optimisations of Phase 3 can change them
// one at a time, each checked against the OpenCL files compiled for CUDA (the "parity" kernels, OTM
// tests of test/test_native_kernels.py).
//
// The source is compiled after the #define block of WavePreparation.kernel_caller, the same as for
// the OpenCL files: the grid size (Nz rows, Nx columns, absorbing layer included), the time step
// and grid spacing (dt, dx, dz), the half order sdo (2 for order 4, 4 for order 8) and the
// coefficients c1..c4 of the staggered first derivative, npml, and the acquisition constants. The
// block also defines center, left, right2... for the OpenCL files: those names are not used here.
//
// Every kernel has the name and the arguments of its OpenCL counterpart, so that the propagation
// engine (PyFWI/engine/propagation.py) launches either set the same way: a grid covering Nz x Nx
// work items, the row i from the y index of the thread and the column j, contiguous in memory,
// from its x index (coalesced accesses).
//
// Step 3.2: the order of the finite differences is a template parameter of the derivatives below,
// so that order 4 contains no term of the zero coefficients c3 and c4 (they read up to 4 cells
// away); the expressions keep the order of the operations of elastic.cl.
// Step 3.3: the derivatives multiply by 1 / dx and 1 / dz instead of dividing. The kernels being
// limited by memory bandwidth, lambda + 2 mu is not stored: the kernels that use it also need
// lambda, so a precomputed field would not save a read.

// ------------------------------------------------------------------ storage of the fields (step 4.3)
//
// The fields and adjoint fields (and their checkpoints and stored bands) are arrays of `real`. In the
// default float32 storage `real` is float, and every kernel below is exactly the float32 code of Phase 3.
// In the FP16 mode (PYFWI_FP16 in the #define block: PropagationEngine with fp16) a `real` is a half-
// precision number, 2 bytes, read as a float and written from a float: the arithmetic stays float32, the
// fields take half the memory and half the bandwidth. The engine scales the fields so that their values
// suit half precision (see PropagationEngine, step 4.3).
#ifdef PYFWI_FP16
#include <cuda_fp16.h>
struct real {
    __half value;
    __device__ __forceinline__ operator float() const { return __half2float(value); }
    __device__ __forceinline__ real &operator=(float x) {
        value = __float2half(x);
        return *this;
    }
    __device__ __forceinline__ real &operator+=(float x) { return *this = float(*this) + x; }
    __device__ __forceinline__ real &operator-=(float x) { return *this = float(*this) - x; }
};
#else
typedef float real;
#endif

// ------------------------------------------------------------------ indices and derivatives

// The row and column of the work item running this thread.
__device__ __forceinline__ int row_of_thread() { return blockIdx.y * blockDim.y + threadIdx.y; }
__device__ __forceinline__ int column_of_thread() { return blockIdx.x * blockDim.x + threadIdx.x; }

// Step 3.9: several shots at once. The kernels marked "batched" (all but advance and sum_slots) run on
// a grid whose z index is the slot of a shot in the batch (PropagationEngine.batch): the fields,
// adjoint fields and gradients of slot b start b * (Nz + 1) * Nx values into their buffers (slot 0
// with a single shot, the grid then having one z index), its seismograms and residual b * rows * Nr
// values; the model and the absorbing layer are the same for every shot, one plane. Between two
// slots, one row that no kernel writes, holding the value of the margins: the stencils of the last
// rows read one row past the field (row Nz, test_known_bugs), which must not be the first row of the
// next shot (a source or receivers on row 0 without absorbing layer), so that each slot computes
// exactly what a single shot computes.
__device__ __forceinline__ long long slot_values() { return (long long)blockIdx.z * (Nz + 1) * Nx; }

// Staggered first derivatives of order 2*SDO of the quantity f (a callable: f(k) is its value at
// the flat index k), at the flat index k, along the axis whose neighbours are `step` apart (1 for
// x, the columns; Nx for z, the rows). Not divided by the grid spacing.
//
//   forward  (half a cell after the point):  sum_m c_m (f(k + m step) - f(k - (m - 1) step))
//   backward (half a cell before the point): sum_m c_m (f(k + (m - 1) step) - f(k - m step))
//
// with m = 1..SDO. The sums run in the order of elastic.cl, so that the compiler contracts them into
// fused multiply-adds the same way.
template <int SDO, typename F>
__device__ __forceinline__ float forward_difference(F f, int k, int step) {
    float sum = c1 * (f(k + step) - f(k)) + c2 * (f(k + 2 * step) - f(k - step));
    if (SDO == 4) {
        sum = sum + c3 * (f(k + 3 * step) - f(k - 2 * step)) + c4 * (f(k + 4 * step) - f(k - 3 * step));
    }
    return sum;
}

template <int SDO, typename F>
__device__ __forceinline__ float backward_difference(F f, int k, int step) {
    float sum = c1 * (f(k) - f(k - step)) + c2 * (f(k + step) - f(k - 2 * step));
    if (SDO == 4) {
        sum = sum + c3 * (f(k + 2 * step) - f(k - 3 * step)) + c4 * (f(k + 3 * step) - f(k - 4 * step));
    }
    return sum;
}

// The inverse grid spacings, computed once by the compiler (step 3.3): the derivatives multiply by
// them instead of dividing by dx and dz, an IEEE division being several instructions.
constexpr float inv_dx = 1.0f / dx;
constexpr float inv_dz = 1.0f / dz;

// The damping of the absorbing layer at a cell, dxpml + dzpml (step 3.10). Inside the rectangle of rows
// z0..z1 and columns x0..x1 where both profiles are zero (PropagationEngine.set_pml finds it: most of
// the grid), the profiles are not read, which saves two reads per cell; the value there is 0, as read.
__device__ __forceinline__ float damping(const float *dxpml, const float *dzpml, int i, int j, int k,
                                         int z0, int z1, int x0, int x1) {
    return (i >= z0 && i <= z1 && j >= x0 && j <= x1) ? 0.0f : dxpml[k] + dzpml[k];
}

// A field as a callable for the derivatives: field(p)(k) is p[k], as a float.
struct Field {
    const real *p;
    __device__ __forceinline__ float operator()(int k) const { return p[k]; }
};
__device__ __forceinline__ Field field(const real *p) { return Field{p}; }

// ------------------------------------------------------------------ zeroing

extern "C" __global__ void MakeAllZero(real *vx, real *vz, real *taux, real *tauz, real *tauxz) {
    const int i = row_of_thread(), j = column_of_thread();
    if (i >= Nz || j >= Nx) return;
    const long long k = slot_values() + i * Nx + j;  // batched
    vx[k] = 0;
    vz[k] = 0;
    taux[k] = 0;
    tauz[k] = 0;
    tauxz[k] = 0;
}

extern "C" __global__ void MakeGradZero(float *Gmu, float *Glam, float *Grho,
                                        float *Gmu_precond, float *Glam_precond, float *Grho_precond) {
    const int i = row_of_thread(), j = column_of_thread();
    if (i >= Nz || j >= Nx) return;
    const long long k = slot_values() + i * Nx + j;  // batched
    Gmu[k] = 0;
    Glam[k] = 0;
    Grho[k] = 0;
    Gmu_precond[k] = 0;
    Glam_precond[k] = 0;
    Grho_precond[k] = 0;
}

// ------------------------------------------------------------------ forward field
//
// One time step: the velocities from the stresses (update_velx, update_velz), then the stresses
// from the new velocities (update_tauz, update_tauxz). coeff = 1 steps forward; coeff = -1 undoes a
// step (the gradient rebuilds the forward field backwards). The cells updated are rows and columns
// sdo to N - sdo; rho_b holds 1 / density, dxpml and dzpml the damping of the absorbing layer.

extern "C" __global__ void update_velx(int coeff, real *vx, const real *taux, const real *tauxz,
                                       const float *rho_b, const float *dxpml, const float *dzpml) {
    const long long f = slot_values();  // batched
    vx += f, taux += f, tauxz += f;
    const int i = row_of_thread() + sdo, j = column_of_thread() + sdo;
    if (i > Nz - sdo || j > Nx - sdo) return;
    const int k = i * Nx + j;
    const float DxmTx = backward_difference<sdo>(field(taux), k, 1) * inv_dx;
    const float DzmTxz = backward_difference<sdo>(field(tauxz), k, Nx) * inv_dz;
    if (coeff == 1) {
        vx[k] += coeff * dt * (rho_b[k] * (DxmTx + DzmTxz));
        vx[k] += coeff * dt * (-(dxpml[k] + dzpml[k]) * vx[k]);
    } else if (coeff == -1) {
        vx[k] += coeff * dt * (+(dxpml[k] + dzpml[k]) * vx[k]);
        vx[k] += coeff * dt * (rho_b[k] * (DxmTx + DzmTxz));
    }
}

extern "C" __global__ void update_velz(int coeff, real *vz, const real *tauz, const real *tauxz,
                                       const float *rho_b, const float *dxpml, const float *dzpml) {
    const long long f = slot_values();  // batched
    vz += f, tauz += f, tauxz += f;
    const int i = row_of_thread() + sdo, j = column_of_thread() + sdo;
    if (i > Nz - sdo || j > Nx - sdo) return;
    const int k = i * Nx + j;
    const float DxpTxz = forward_difference<sdo>(field(tauxz), k, 1) * inv_dx;
    const float DzpTz = forward_difference<sdo>(field(tauz), k, Nx) * inv_dz;
    if (coeff == 1) {
        vz[k] += coeff * dt * (rho_b[k] * (DzpTz + DxpTxz));
        vz[k] += coeff * dt * (-(dzpml[k] + dxpml[k]) * vz[k]);
    } else if (coeff == -1) {
        vz[k] += coeff * dt * (+(dzpml[k] + dxpml[k]) * vz[k]);
        vz[k] += coeff * dt * (rho_b[k] * (DzpTz + DxpTxz));
    }
}

extern "C" __global__ void update_tauz(int coeff, const real *vx, const real *vz, real *taux, real *tauz,
                                       const float *lambda, const float *mu, const float *dxpml,
                                       const float *dzpml) {
    const long long f = slot_values();  // batched
    vx += f, vz += f, taux += f, tauz += f;
    const int i = row_of_thread() + sdo, j = column_of_thread() + sdo;
    if (i > Nz - sdo || j > Nx - sdo) return;
    const int k = i * Nx + j;
    const float DxpVx = forward_difference<sdo>(field(vx), k, 1) * inv_dx;
    const float DzmVz = backward_difference<sdo>(field(vz), k, Nx) * inv_dz;
    if (coeff == 1) {
        taux[k] += coeff * dt * ((lambda[k] + 2 * mu[k]) * DxpVx + lambda[k] * DzmVz);
        tauz[k] += coeff * dt * ((lambda[k] + 2 * mu[k]) * DzmVz + lambda[k] * DxpVx);
        taux[k] += coeff * dt * (-(dxpml[k] + dzpml[k]) * taux[k]);
        tauz[k] += coeff * dt * (-(dxpml[k] + dzpml[k]) * tauz[k]);
    } else if (coeff == -1) {
        taux[k] += coeff * dt * (+(dxpml[k] + dzpml[k]) * taux[k]);
        tauz[k] += coeff * dt * (+(dxpml[k] + dzpml[k]) * tauz[k]);
        taux[k] += coeff * dt * ((lambda[k] + 2 * mu[k]) * DxpVx + lambda[k] * DzmVz);
        tauz[k] += coeff * dt * ((lambda[k] + 2 * mu[k]) * DzmVz + lambda[k] * DxpVx);
    }
}

extern "C" __global__ void update_tauxz(int coeff, const real *vx, const real *vz, real *tauxz,
                                        const float *mu, const float *dxpml, const float *dzpml) {
    const long long f = slot_values();  // batched
    vx += f, vz += f, tauxz += f;
    const int i = row_of_thread() + sdo, j = column_of_thread() + sdo;
    if (i > Nz - sdo || j > Nx - sdo) return;
    const int k = i * Nx + j;
    const float DxmVz = backward_difference<sdo>(field(vz), k, 1) * inv_dx;
    const float DzpVx = forward_difference<sdo>(field(vx), k, Nx) * inv_dz;
    if (coeff == 1) {
        tauxz[k] += coeff * dt * (mu[k] * (DxmVz + DzpVx));
        tauxz[k] += coeff * dt * (-(dxpml[k] + dzpml[k]) * tauxz[k]);
    } else if (coeff == -1) {
        tauxz[k] += coeff * dt * (+(dxpml[k] + dzpml[k]) * tauxz[k]);
        tauxz[k] += coeff * dt * (mu[k] * (DxmVz + DzpVx));
    }
}

// ------------------------------------------------------------------ adjoint field
//
// The transpose of the forward step (PropagationEngine._adjoint_updates runs Adj_update_v, then
// Adj_update_tau): each adjoint field damped by dxpml + dzpml in the absorbing layer, as its forward
// field.

extern "C" __global__ void Adj_update_tau(const real *Avx, const real *Avz, real *Ataux, real *Atauz,
                                          real *Atauxz, const float *rho_b, const float *dxpml,
                                          const float *dzpml, int z0, int z1, int x0, int x1) {
    const long long f = slot_values();  // batched
    Avx += f, Avz += f, Ataux += f, Atauz += f, Atauxz += f;
    const int i = row_of_thread() + sdo, j = column_of_thread() + sdo;
    if (i > Nz - sdo || j > Nx - sdo) return;
    const int k = i * Nx + j;
    const float d = damping(dxpml, dzpml, i, j, k, z0, z1, x0, x1);
    const auto avx_rho = [=](int n) { return Avx[n] * rho_b[n]; };
    const auto avz_rho = [=](int n) { return Avz[n] * rho_b[n]; };
    const float DxpAVx = forward_difference<sdo>(avx_rho, k, 1) * inv_dx;
    const float DzmAVz = backward_difference<sdo>(avz_rho, k, Nx) * inv_dz;
    Ataux[k] += -dt * (d * Ataux[k] + DxpAVx);
    Atauz[k] += -dt * (d * Atauz[k] + DzmAVz);
    const float DzpAvx = forward_difference<sdo>(avx_rho, k, Nx) * inv_dz;
    const float DxmAvz = backward_difference<sdo>(avz_rho, k, 1) * inv_dx;
    Atauxz[k] += -dt * ((d * Atauxz[k]) + (DzpAvx + DxmAvz));
}

extern "C" __global__ void Adj_update_v(real *Avx, real *Avz, const real *Ataux, const real *Atauz,
                                        const real *Atauxz, const float *lam, const float *mu,
                                        const float *dxpml, const float *dzpml) {
    const long long f = slot_values();  // batched
    Avx += f, Avz += f, Ataux += f, Atauz += f, Atauxz += f;
    const int i = row_of_thread() + sdo, j = column_of_thread() + sdo;
    if (i > Nz - sdo || j > Nx - sdo) return;
    const int k = i * Nx + j;
    const auto p_ataux = [=](int n) { return (lam[n] + 2 * mu[n]) * Ataux[n]; };  // lambda + 2 mu
    const auto p_atauz = [=](int n) { return (lam[n] + 2 * mu[n]) * Atauz[n]; };
    const auto l_ataux = [=](int n) { return lam[n] * Ataux[n]; };
    const auto l_atauz = [=](int n) { return lam[n] * Atauz[n]; };
    const auto m_atauxz = [=](int n) { return mu[n] * Atauxz[n]; };
    const float DxmAtaux = backward_difference<sdo>(p_ataux, k, 1) * inv_dx;
    const float DxmAtauz = backward_difference<sdo>(l_atauz, k, 1) * inv_dx;
    const float DzmAtauxz = backward_difference<sdo>(m_atauxz, k, Nx) * inv_dz;
    const float DzpAtauz = forward_difference<sdo>(p_atauz, k, Nx) * inv_dz;
    const float DzpAtaux = forward_difference<sdo>(l_ataux, k, Nx) * inv_dz;
    const float DxpAtauxz = forward_difference<sdo>(m_atauxz, k, 1) * inv_dx;
    Avx[k] += -dt * ((dxpml[k] + dzpml[k]) * Avx[k] + (DxmAtaux + DxmAtauz + DzmAtauxz));
    Avz[k] += -dt * ((dxpml[k] + dzpml[k]) * Avz[k] + (DzpAtaux + DzpAtauz + DxpAtauxz));
}

// ------------------------------------------------------------------ gradient
//
// Accumulates, at every step, the correlation of the forward field with the adjoint field (the
// gradient with respect to mu, lambda and 1 / density) and the energy of the forward field (the
// preconditioners), outside the absorbing layer.

extern "C" __global__ void Grad(const real *vx, const real *vz, const real *taux, const real *tauz,
                                const real *tauxz, const real *Avx, const real *Avz, const real *Ataux,
                                const real *Atauz, const real *Atauxz, const float *lambda, const float *mu,
                                const float *rho_b, float *Gmu, float *Gmu_precond, float *Glam,
                                float *Glam_precond, float *Grho, float *Grho_precond) {
    const long long f = slot_values();  // batched
    vx += f, vz += f, taux += f, tauz += f, tauxz += f, Avx += f, Avz += f, Ataux += f, Atauz += f, Atauxz += f;
    Gmu += f, Gmu_precond += f, Glam += f, Glam_precond += f, Grho += f, Grho_precond += f;
    const int i = row_of_thread() + npml + sdo, j = column_of_thread() + npml + sdo;
    if (i > Nz - sdo - npml || j > Nx - sdo - npml) return;
    const int k = i * Nx + j;
    const float DzpVx = forward_difference<sdo>(field(vx), k, Nx) * inv_dz;
    const float DxmVz = backward_difference<sdo>(field(vz), k, 1) * inv_dx;
    const float DxpVx = forward_difference<sdo>(field(vx), k, 1) * inv_dx;
    const float DzmVz = backward_difference<sdo>(field(vz), k, Nx) * inv_dz;
    const float DzpPz = forward_difference<sdo>(field(tauz), k, Nx) * inv_dz;
    const float DxmPx = backward_difference<sdo>(field(taux), k, 1) * inv_dx;
    const float DxpPxz = forward_difference<sdo>(field(tauxz), k, 1) * inv_dx;
    const float DzmPxz = backward_difference<sdo>(field(tauxz), k, Nx) * inv_dz;
    Gmu[k] += 2 * dt * (Ataux[k] * DxpVx + Atauz[k] * DzmVz) + dt * Atauxz[k] * (DzpVx + DxmVz);
    Gmu_precond[k] += 4 * dt * dt * (taux[k] * taux[k] * DxpVx * DxpVx + tauz[k] * tauz[k] * DzmVz * DzmVz)
                      + dt * dt * tauxz[k] * tauxz[k] * (DzpVx * DzpVx + DxmVz * DxmVz);
    Glam[k] += dt * (Ataux[k] + Atauz[k]) * (DxpVx + DzmVz);
    Glam_precond[k] += dt * dt * (taux[k] * taux[k] + tauz[k] * tauz[k]) * (DxpVx * DxpVx + DzmVz * DzmVz);
    Grho[k] += -(dt * (rho_b[k] * rho_b[k])) * (Avx[k] * (DxmPx + DzmPxz) + Avz[k] * (DzpPz + DxpPxz));
    Grho_precond[k] += dt * dt * (vx[k] * vx[k] * (DxmPx * DxmPx + DzmPxz * DzmPxz)
                                  + vz[k] * vz[k] * (DzpPz * DzpPz + DxpPxz * DxpPxz));
}

// ------------------------------------------------------------------ fused kernels (step 3.5)
//
// What the time loops run: the two velocity updates in one kernel, the two stress updates in one
// kernel, and the gradient with the adjoint velocity update. Each thread does, for its cell, exactly
// what the separate kernels above do, in the same order: the halves of each pair are independent
// (update_velx and update_velz read different stresses, update_tauz and update_tauxz write different
// stresses), and the gradient reads the adjoint velocities of its own cell before the adjoint update
// changes them (no thread reads the adjoint velocities of another cell there). Fewer launches, and
// every field read once per kernel instead of once per half (a forward step moves 29 % less data).

extern "C" __global__ void update_v(int coeff, real *vx, real *vz, const real *taux, const real *tauz,
                                    const real *tauxz, const float *rho_b, const float *dxpml,
                                    const float *dzpml, int z0, int z1, int x0, int x1) {
    const long long f = slot_values();  // batched
    vx += f, vz += f, taux += f, tauz += f, tauxz += f;
    const int i = row_of_thread() + sdo, j = column_of_thread() + sdo;
    if (i > Nz - sdo || j > Nx - sdo) return;
    const int k = i * Nx + j;
    const float d = damping(dxpml, dzpml, i, j, k, z0, z1, x0, x1);
    const float DxmTx = backward_difference<sdo>(field(taux), k, 1) * inv_dx;
    const float DzmTxz = backward_difference<sdo>(field(tauxz), k, Nx) * inv_dz;
    const float DxpTxz = forward_difference<sdo>(field(tauxz), k, 1) * inv_dx;
    const float DzpTz = forward_difference<sdo>(field(tauz), k, Nx) * inv_dz;
    if (coeff == 1) {
        vx[k] += coeff * dt * (rho_b[k] * (DxmTx + DzmTxz));
        vx[k] += coeff * dt * (-d * vx[k]);
        vz[k] += coeff * dt * (rho_b[k] * (DzpTz + DxpTxz));
        vz[k] += coeff * dt * (-d * vz[k]);
    } else if (coeff == -1) {
        vx[k] += coeff * dt * (+d * vx[k]);
        vx[k] += coeff * dt * (rho_b[k] * (DxmTx + DzmTxz));
        vz[k] += coeff * dt * (+d * vz[k]);
        vz[k] += coeff * dt * (rho_b[k] * (DzpTz + DxpTxz));
    }
}

extern "C" __global__ void update_tau(int coeff, const real *vx, const real *vz, real *taux, real *tauz,
                                      real *tauxz, const float *lambda, const float *mu, const float *dxpml,
                                      const float *dzpml, int z0, int z1, int x0, int x1) {
    const long long f = slot_values();  // batched
    vx += f, vz += f, taux += f, tauz += f, tauxz += f;
    const int i = row_of_thread() + sdo, j = column_of_thread() + sdo;
    if (i > Nz - sdo || j > Nx - sdo) return;
    const int k = i * Nx + j;
    const float d = damping(dxpml, dzpml, i, j, k, z0, z1, x0, x1);
    const float DxpVx = forward_difference<sdo>(field(vx), k, 1) * inv_dx;
    const float DzmVz = backward_difference<sdo>(field(vz), k, Nx) * inv_dz;
    const float DxmVz = backward_difference<sdo>(field(vz), k, 1) * inv_dx;
    const float DzpVx = forward_difference<sdo>(field(vx), k, Nx) * inv_dz;
    if (coeff == 1) {
        taux[k] += coeff * dt * ((lambda[k] + 2 * mu[k]) * DxpVx + lambda[k] * DzmVz);
        tauz[k] += coeff * dt * ((lambda[k] + 2 * mu[k]) * DzmVz + lambda[k] * DxpVx);
        taux[k] += coeff * dt * (-d * taux[k]);
        tauz[k] += coeff * dt * (-d * tauz[k]);
        tauxz[k] += coeff * dt * (mu[k] * (DxmVz + DzpVx));
        tauxz[k] += coeff * dt * (-d * tauxz[k]);
    } else if (coeff == -1) {
        taux[k] += coeff * dt * (+d * taux[k]);
        tauz[k] += coeff * dt * (+d * tauz[k]);
        taux[k] += coeff * dt * ((lambda[k] + 2 * mu[k]) * DxpVx + lambda[k] * DzmVz);
        tauz[k] += coeff * dt * ((lambda[k] + 2 * mu[k]) * DzmVz + lambda[k] * DxpVx);
        tauxz[k] += coeff * dt * (+d * tauxz[k]);
        tauxz[k] += coeff * dt * (mu[k] * (DxmVz + DzpVx));
    }
}

// Grad on the cells outside the absorbing layer, then Adj_update_v on every updated cell.
extern "C" __global__ void grad_adjoint_v(const real *vx, const real *vz, const real *taux, const real *tauz,
                                          const real *tauxz, real *Avx, real *Avz, const real *Ataux,
                                          const real *Atauz, const real *Atauxz, const float *lambda,
                                          const float *mu, const float *rho_b, const float *dxpml,
                                          const float *dzpml, float *Gmu, float *Gmu_precond, float *Glam,
                                          float *Glam_precond, float *Grho, float *Grho_precond, int z0,
                                          int z1, int x0, int x1) {
    const long long f = slot_values();  // batched
    vx += f, vz += f, taux += f, tauz += f, tauxz += f, Avx += f, Avz += f, Ataux += f, Atauz += f, Atauxz += f;
    Gmu += f, Gmu_precond += f, Glam += f, Glam_precond += f, Grho += f, Grho_precond += f;
    const int i = row_of_thread() + sdo, j = column_of_thread() + sdo;
    if (i > Nz - sdo || j > Nx - sdo) return;
    const int k = i * Nx + j;
    if (i >= npml + sdo && j >= npml + sdo && i <= Nz - sdo - npml && j <= Nx - sdo - npml) {
        const float DzpVx = forward_difference<sdo>(field(vx), k, Nx) * inv_dz;
        const float DxmVz = backward_difference<sdo>(field(vz), k, 1) * inv_dx;
        const float DxpVx = forward_difference<sdo>(field(vx), k, 1) * inv_dx;
        const float DzmVz = backward_difference<sdo>(field(vz), k, Nx) * inv_dz;
        const float DzpPz = forward_difference<sdo>(field(tauz), k, Nx) * inv_dz;
        const float DxmPx = backward_difference<sdo>(field(taux), k, 1) * inv_dx;
        const float DxpPxz = forward_difference<sdo>(field(tauxz), k, 1) * inv_dx;
        const float DzmPxz = backward_difference<sdo>(field(tauxz), k, Nx) * inv_dz;
        Gmu[k] += 2 * dt * (Ataux[k] * DxpVx + Atauz[k] * DzmVz) + dt * Atauxz[k] * (DzpVx + DxmVz);
        Gmu_precond[k] += 4 * dt * dt * (taux[k] * taux[k] * DxpVx * DxpVx + tauz[k] * tauz[k] * DzmVz * DzmVz)
                          + dt * dt * tauxz[k] * tauxz[k] * (DzpVx * DzpVx + DxmVz * DxmVz);
        Glam[k] += dt * (Ataux[k] + Atauz[k]) * (DxpVx + DzmVz);
        Glam_precond[k] += dt * dt * (taux[k] * taux[k] + tauz[k] * tauz[k]) * (DxpVx * DxpVx + DzmVz * DzmVz);
        Grho[k] += -(dt * (rho_b[k] * rho_b[k])) * (Avx[k] * (DxmPx + DzmPxz) + Avz[k] * (DzpPz + DxpPxz));
        Grho_precond[k] += dt * dt * (vx[k] * vx[k] * (DxmPx * DxmPx + DzmPxz * DzmPxz)
                                      + vz[k] * vz[k] * (DzpPz * DzpPz + DxpPxz * DxpPxz));
    }
    const float d = damping(dxpml, dzpml, i, j, k, z0, z1, x0, x1);
    const auto p_ataux = [=](int n) { return (lambda[n] + 2 * mu[n]) * Ataux[n]; };
    const auto p_atauz = [=](int n) { return (lambda[n] + 2 * mu[n]) * Atauz[n]; };
    const auto l_ataux = [=](int n) { return lambda[n] * Ataux[n]; };
    const auto l_atauz = [=](int n) { return lambda[n] * Atauz[n]; };
    const auto m_atauxz = [=](int n) { return mu[n] * Atauxz[n]; };
    const float DxmAtaux = backward_difference<sdo>(p_ataux, k, 1) * inv_dx;
    const float DxmAtauz = backward_difference<sdo>(l_atauz, k, 1) * inv_dx;
    const float DzmAtauxz = backward_difference<sdo>(m_atauxz, k, Nx) * inv_dz;
    const float DzpAtauz = forward_difference<sdo>(p_atauz, k, Nx) * inv_dz;
    const float DzpAtaux = forward_difference<sdo>(l_ataux, k, Nx) * inv_dz;
    const float DxpAtauxz = forward_difference<sdo>(m_atauxz, k, 1) * inv_dx;
    Avx[k] += -dt * (d * Avx[k] + (DxmAtaux + DxmAtauz + DzmAtauxz));
    Avz[k] += -dt * (d * Avz[k] + (DzpAtaux + DzpAtauz + DxpAtauxz));
}

// ------------------------------------------------------------------ source and receivers
//
// Step 3.4: one thread per receiver instead of the whole grid. The cell of every receiver comes
// from a table made on the host (PyFWI.engine.propagation.receiver_table, the loops of the OpenCL
// kernels); the engine launches these kernels on a grid of 1 row and Nr columns (per slot of a batch).
//
// Step 3.6: the time step is on the device. *step is the internal step t (advance changes it); the
// source of every step comes from source_steps (4 values per step: vx, vz, taux, tauz), the residual
// from the residual of the shot, and the recorded sample of a step is t / dt_scale when t is a
// multiple of dt_scale and that sample exists. Every launch of a step then has the same arguments,
// so that the steps can be recorded once into a CUDA graph and replayed (step 3.7), and the residual
// is read in place, without copies.

// The recorded sample of internal step t, or -1 when t is not a recorded step (or past the last row).
__device__ __forceinline__ int recorded_sample(int t, int dt_scale, int rows) {
    const int sample = t / dt_scale;
    return (t % dt_scale == 0 && sample < rows) ? sample : -1;
}

// Add the source of step *step at its cell (thread 0) and, at a recorded step, record every receiver
// in the row of that sample (thread r for receiver r): what injSrc of the OpenCL files does, in one
// launch.
extern "C" __global__ void inject_and_record(real *vx, real *vz, real *taux, real *tauz, const real *tauxz,
                                             float *seismogram_vxi, float *seismogram_vzi,
                                             float *seismogram_tauxi, float *seismogram_tauzi,
                                             float *seismogram_tauxzi, const int *cells, const int *source_cell,
                                             const float *source_steps, const int *step, int dt_scale, int rows) {
    // batched: source_cell holds the cell of the source of every slot
    const long long f = slot_values(), g = (long long)blockIdx.z * rows * Nr;
    vx += f, vz += f, taux += f, tauz += f, tauxz += f;
    seismogram_vxi += g, seismogram_vzi += g, seismogram_tauxi += g, seismogram_tauzi += g, seismogram_tauxzi += g;
    const int r = column_of_thread();
    if (row_of_thread() != 0) return;
    const int t = *step;
    if (r == 0) {
        const int cell = source_cell[blockIdx.z];
        vx[cell] += source_steps[4 * t];
        vz[cell] += source_steps[4 * t + 1];
        taux[cell] += source_steps[4 * t + 2];
        tauz[cell] += source_steps[4 * t + 3];
    }
    const int row = recorded_sample(t, dt_scale, rows);
    if (row < 0 || r >= Nr || cells[r] < 0) return;
    const int cell = cells[r];
    seismogram_vxi[row * Nr + r] = vx[cell];
    seismogram_vzi[row * Nr + r] = vz[cell];
    seismogram_tauxi[row * Nr + r] = taux[cell];
    seismogram_tauzi[row * Nr + r] = tauz[cell];
    seismogram_tauxzi[row * Nr + r] = tauxz[cell];
}

// At a recorded step, add the residual of every receiver (residual of the shot: rows x Nr values per
// component) to the adjoint fields at its cell (Adj_injSrc of the OpenCL files). The first receiver of
// a cell adds those of the receivers that share it, in their order (first, following:
// PyFWI.engine.propagation.receiver_table), so that no two threads write one cell: deterministic
// without atomic operations.
extern "C" __global__ void inject_residual(real *Avx, real *Avz, real *Ataux, real *Atauz, real *Atauxz,
                                           const float *res_vx, const float *res_vz, const float *res_taux,
                                           const float *res_tauz, const float *res_tauxz, const int *cells,
                                           const int *first, const int *following, const int *step,
                                           int dt_scale, int rows) {
    // batched: the residual of each slot
    const long long f = slot_values(), g = (long long)blockIdx.z * rows * Nr;
    Avx += f, Avz += f, Ataux += f, Atauz += f, Atauxz += f;
    res_vx += g, res_vz += g, res_taux += g, res_tauz += g, res_tauxz += g;
    const int r = column_of_thread();
    if (row_of_thread() != 0 || r >= Nr || cells[r] < 0 || !first[r]) return;
    const int sample = recorded_sample(*step, dt_scale, rows);
    if (sample < 0) return;
    const int base = sample * Nr;
    float sum_vx = res_vx[base + r], sum_vz = res_vz[base + r], sum_taux = res_taux[base + r],
          sum_tauz = res_tauz[base + r], sum_tauxz = res_tauxz[base + r];
    for (int q = following[r]; q >= 0; q = following[q]) {
        sum_vx += res_vx[base + q];
        sum_vz += res_vz[base + q];
        sum_taux += res_taux[base + q];
        sum_tauz += res_tauz[base + q];
        sum_tauxz += res_tauxz[base + q];
    }
    const int cell = cells[r];
    Avx[cell] += sum_vx;
    Avz[cell] += sum_vz;
    Ataux[cell] += sum_taux;
    Atauz[cell] += sum_tauz;
    Atauxz[cell] += sum_tauxz;
}

// Move the time step on the device: *step += delta (one thread).
extern "C" __global__ void advance(int *step, int delta) {
    if (row_of_thread() == 0 && column_of_thread() == 0) *step += delta;
}

// Step 3.9: add the gradients of the slots 1 to slots - 1 to slot 0, in that order (a fixed order:
// the same bits at every run, without atomic operations). Launched on the grid of one shot.
extern "C" __global__ void sum_slots(float *Gmu, float *Gmu_precond, float *Glam, float *Glam_precond, float *Grho,
                                     float *Grho_precond, int slots) {
    const int i = row_of_thread(), j = column_of_thread();
    if (i >= Nz || j >= Nx) return;
    const long long k = i * Nx + j, stride = (long long)(Nz + 1) * Nx;  // see slot_values
    for (int b = 1; b < slots; b++) {
        Gmu[k] += Gmu[b * stride + k];
        Gmu_precond[k] += Gmu_precond[b * stride + k];
        Glam[k] += Glam[b * stride + k];
        Glam_precond[k] += Glam_precond[b * stride + k];
        Grho[k] += Grho[b * stride + k];
        Grho_precond[k] += Grho_precond[b * stride + k];
    }
}

// Step 3.9: copy the forward field of every slot launched into its checkpoint on the device (restore =
// 0), or back (restore = 1), in one launch instead of one copy per component and shot. The checkpoints
// of the shots kept on the device are one buffer of (5 components, shots, checkpoints, Nz * Nx)
// values (PropagationEngine._device_checkpoints): slot b's checkpoint of component c starts
// c * component_pitch + offset + b * shot_pitch values into it, offset being that of the first slot.
extern "C" __global__ void copy_checkpoint(real *vx, real *vz, real *taux, real *tauz, real *tauxz,
                                           real *checkpoints, long long offset, long long shot_pitch,
                                           long long component_pitch, int restore) {
    const int i = row_of_thread(), j = column_of_thread();
    if (i >= Nz || j >= Nx) return;
    const long long k = i * Nx + j, f = slot_values() + k;  // batched
    real *stored = checkpoints + offset + blockIdx.z * shot_pitch + k;
    real *fields[5] = {vx, vz, taux, tauz, tauxz};
    for (int c = 0; c < 5; c++) {
        if (restore) {
            fields[c][f] = stored[c * component_pitch];
        } else {
            stored[c * component_pitch] = fields[c][f];
        }
    }
}

// ------------------------------------------------------------------ boundary saving (step 4.2)
//
// Instead of checkpoints of the whole field, the forward modelling stores at every step only the fields of
// a band of cells: those of the undamped rectangle (no absorbing layer: PropagationEngine.undamped) within
// sdo cells of its edge (`band`, n_band flat indices), right after the source injection. The gradient then
// rebuilds the forward field backwards from the last field: it undoes the stresses, writes back the stored
// stresses of the band, undoes the velocities, writes back the stored velocities of the band, and removes
// the source. Inside the rectangle there is no damping, so undoing a step is exact but for float32
// rounding, and the cells off the band read only cells of the rectangle (the stencils reach sdo cells). The
// store of a shot holds nt steps of 5 x n_band values (vx, vz, taux, tauz, tauxz); band_shot[slot] is the
// shot of every slot of a batch. Launched on a grid of 1 row and n_band columns per slot.

extern "C" __global__ void save_band(const real *vx, const real *vz, const real *taux, const real *tauz,
                                     const real *tauxz, real *store, const int *band, const int *band_shot,
                                     const int *step, int n_band, long long shot_values) {
    const int b = column_of_thread();
    if (row_of_thread() != 0 || b >= n_band) return;
    const long long cell = slot_values() + band[b];  // batched
    real *stored = store + band_shot[blockIdx.z] * shot_values + (long long)(*step) * 5 * n_band;
    stored[b] = vx[cell];
    stored[n_band + b] = vz[cell];
    stored[2 * n_band + b] = taux[cell];
    stored[3 * n_band + b] = tauz[cell];
    stored[4 * n_band + b] = tauxz[cell];
}

// Write back the band stored at step *step + delta: the stresses when `stresses` is 1, else the velocities.
extern "C" __global__ void restore_band(real *vx, real *vz, real *taux, real *tauz, real *tauxz,
                                        const real *store, const int *band, const int *band_shot, const int *step,
                                        int delta, int stresses, int n_band, long long shot_values) {
    const int b = column_of_thread();
    if (row_of_thread() != 0 || b >= n_band) return;
    const long long cell = slot_values() + band[b];  // batched
    const real *stored = store + band_shot[blockIdx.z] * shot_values + (long long)(*step + delta) * 5 * n_band;
    if (stresses) {
        taux[cell] = stored[2 * n_band + b];
        tauz[cell] = stored[3 * n_band + b];
        tauxz[cell] = stored[4 * n_band + b];
    } else {
        vx[cell] = stored[b];
        vz[cell] = stored[n_band + b];
    }
}

// Remove the source of step *step + delta from its cell: the inverse of the injection of inject_and_record.
extern "C" __global__ void remove_source(real *vx, real *vz, real *taux, real *tauz, const int *source_cell,
                                         const float *source_steps, const int *step, int delta) {
    if (row_of_thread() != 0 || column_of_thread() != 0) return;
    const long long cell = slot_values() + source_cell[blockIdx.z];  // batched
    const int t = *step + delta;
    vx[cell] -= source_steps[4 * t];
    vz[cell] -= source_steps[4 * t + 1];
    taux[cell] -= source_steps[4 * t + 2];
    tauz[cell] -= source_steps[4 * t + 3];
}

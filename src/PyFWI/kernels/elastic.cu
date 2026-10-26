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

// ------------------------------------------------------------------ indices and derivatives

// The row and column of the work item running this thread.
__device__ __forceinline__ int row_of_thread() { return blockIdx.y * blockDim.y + threadIdx.y; }
__device__ __forceinline__ int column_of_thread() { return blockIdx.x * blockDim.x + threadIdx.x; }

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

// A field as a callable for the derivatives: field(p)(k) is p[k].
struct Field {
    const float *p;
    __device__ __forceinline__ float operator()(int k) const { return p[k]; }
};
__device__ __forceinline__ Field field(const float *p) { return Field{p}; }

// ------------------------------------------------------------------ zeroing

extern "C" __global__ void MakeAllZero(float *vx, float *vz, float *taux, float *tauz, float *tauxz) {
    const int i = row_of_thread(), j = column_of_thread();
    if (i >= Nz || j >= Nx) return;
    const int k = i * Nx + j;
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
    const int k = i * Nx + j;
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

extern "C" __global__ void update_velx(int coeff, float *vx, const float *taux, const float *tauxz,
                                       const float *rho_b, const float *dxpml, const float *dzpml) {
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

extern "C" __global__ void update_velz(int coeff, float *vz, const float *tauz, const float *tauxz,
                                       const float *rho_b, const float *dxpml, const float *dzpml) {
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

extern "C" __global__ void update_tauz(int coeff, const float *vx, const float *vz, float *taux, float *tauz,
                                       const float *lambda, const float *mu, const float *dxpml,
                                       const float *dzpml) {
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

extern "C" __global__ void update_tauxz(int coeff, const float *vx, const float *vz, float *tauxz,
                                        const float *mu, const float *dxpml, const float *dzpml) {
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

extern "C" __global__ void Adj_update_tau(const float *Avx, const float *Avz, float *Ataux, float *Atauz,
                                          float *Atauxz, const float *rho_b, const float *dxpml,
                                          const float *dzpml) {
    const int i = row_of_thread() + sdo, j = column_of_thread() + sdo;
    if (i > Nz - sdo || j > Nx - sdo) return;
    const int k = i * Nx + j;
    const auto avx_rho = [=](int n) { return Avx[n] * rho_b[n]; };
    const auto avz_rho = [=](int n) { return Avz[n] * rho_b[n]; };
    const float DxpAVx = forward_difference<sdo>(avx_rho, k, 1) * inv_dx;
    const float DzmAVz = backward_difference<sdo>(avz_rho, k, Nx) * inv_dz;
    Ataux[k] += -dt * ((dxpml[k] + dzpml[k]) * Ataux[k] + DxpAVx);
    Atauz[k] += -dt * ((dxpml[k] + dzpml[k]) * Atauz[k] + DzmAVz);
    const float DzpAvx = forward_difference<sdo>(avx_rho, k, Nx) * inv_dz;
    const float DxmAvz = backward_difference<sdo>(avz_rho, k, 1) * inv_dx;
    Atauxz[k] += -dt * (((dxpml[k] + dzpml[k]) * Atauxz[k]) + (DzpAvx + DxmAvz));
}

extern "C" __global__ void Adj_update_v(float *Avx, float *Avz, const float *Ataux, const float *Atauz,
                                        const float *Atauxz, const float *lam, const float *mu,
                                        const float *dxpml, const float *dzpml) {
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

extern "C" __global__ void Grad(const float *vx, const float *vz, const float *taux, const float *tauz,
                                const float *tauxz, const float *Avx, const float *Avz, const float *Ataux,
                                const float *Atauz, const float *Atauxz, const float *lambda, const float *mu,
                                const float *rho_b, float *Gmu, float *Gmu_precond, float *Glam,
                                float *Glam_precond, float *Grho, float *Grho_precond) {
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

extern "C" __global__ void update_v(int coeff, float *vx, float *vz, const float *taux, const float *tauz,
                                    const float *tauxz, const float *rho_b, const float *dxpml,
                                    const float *dzpml) {
    const int i = row_of_thread() + sdo, j = column_of_thread() + sdo;
    if (i > Nz - sdo || j > Nx - sdo) return;
    const int k = i * Nx + j;
    const float DxmTx = backward_difference<sdo>(field(taux), k, 1) * inv_dx;
    const float DzmTxz = backward_difference<sdo>(field(tauxz), k, Nx) * inv_dz;
    const float DxpTxz = forward_difference<sdo>(field(tauxz), k, 1) * inv_dx;
    const float DzpTz = forward_difference<sdo>(field(tauz), k, Nx) * inv_dz;
    if (coeff == 1) {
        vx[k] += coeff * dt * (rho_b[k] * (DxmTx + DzmTxz));
        vx[k] += coeff * dt * (-(dxpml[k] + dzpml[k]) * vx[k]);
        vz[k] += coeff * dt * (rho_b[k] * (DzpTz + DxpTxz));
        vz[k] += coeff * dt * (-(dzpml[k] + dxpml[k]) * vz[k]);
    } else if (coeff == -1) {
        vx[k] += coeff * dt * (+(dxpml[k] + dzpml[k]) * vx[k]);
        vx[k] += coeff * dt * (rho_b[k] * (DxmTx + DzmTxz));
        vz[k] += coeff * dt * (+(dzpml[k] + dxpml[k]) * vz[k]);
        vz[k] += coeff * dt * (rho_b[k] * (DzpTz + DxpTxz));
    }
}

extern "C" __global__ void update_tau(int coeff, const float *vx, const float *vz, float *taux, float *tauz,
                                      float *tauxz, const float *lambda, const float *mu, const float *dxpml,
                                      const float *dzpml) {
    const int i = row_of_thread() + sdo, j = column_of_thread() + sdo;
    if (i > Nz - sdo || j > Nx - sdo) return;
    const int k = i * Nx + j;
    const float DxpVx = forward_difference<sdo>(field(vx), k, 1) * inv_dx;
    const float DzmVz = backward_difference<sdo>(field(vz), k, Nx) * inv_dz;
    const float DxmVz = backward_difference<sdo>(field(vz), k, 1) * inv_dx;
    const float DzpVx = forward_difference<sdo>(field(vx), k, Nx) * inv_dz;
    if (coeff == 1) {
        taux[k] += coeff * dt * ((lambda[k] + 2 * mu[k]) * DxpVx + lambda[k] * DzmVz);
        tauz[k] += coeff * dt * ((lambda[k] + 2 * mu[k]) * DzmVz + lambda[k] * DxpVx);
        taux[k] += coeff * dt * (-(dxpml[k] + dzpml[k]) * taux[k]);
        tauz[k] += coeff * dt * (-(dxpml[k] + dzpml[k]) * tauz[k]);
        tauxz[k] += coeff * dt * (mu[k] * (DxmVz + DzpVx));
        tauxz[k] += coeff * dt * (-(dxpml[k] + dzpml[k]) * tauxz[k]);
    } else if (coeff == -1) {
        taux[k] += coeff * dt * (+(dxpml[k] + dzpml[k]) * taux[k]);
        tauz[k] += coeff * dt * (+(dxpml[k] + dzpml[k]) * tauz[k]);
        taux[k] += coeff * dt * ((lambda[k] + 2 * mu[k]) * DxpVx + lambda[k] * DzmVz);
        tauz[k] += coeff * dt * ((lambda[k] + 2 * mu[k]) * DzmVz + lambda[k] * DxpVx);
        tauxz[k] += coeff * dt * (+(dxpml[k] + dzpml[k]) * tauxz[k]);
        tauxz[k] += coeff * dt * (mu[k] * (DxmVz + DzpVx));
    }
}

// Grad on the cells outside the absorbing layer, then Adj_update_v on every updated cell.
extern "C" __global__ void grad_adjoint_v(const float *vx, const float *vz, const float *taux, const float *tauz,
                                          const float *tauxz, float *Avx, float *Avz, const float *Ataux,
                                          const float *Atauz, const float *Atauxz, const float *lambda,
                                          const float *mu, const float *rho_b, const float *dxpml,
                                          const float *dzpml, float *Gmu, float *Gmu_precond, float *Glam,
                                          float *Glam_precond, float *Grho, float *Grho_precond) {
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
    Avx[k] += -dt * ((dxpml[k] + dzpml[k]) * Avx[k] + (DxmAtaux + DxmAtauz + DzmAtauxz));
    Avz[k] += -dt * ((dxpml[k] + dzpml[k]) * Avz[k] + (DzpAtaux + DzpAtauz + DxpAtauxz));
}

// ------------------------------------------------------------------ source and receivers
//
// Step 3.4: one thread per receiver instead of the whole grid. The cell of every receiver comes
// from a table made on the host (PyFWI.engine.propagation.receiver_table, the loops of the OpenCL
// kernels); the engine launches these kernels on a grid of 1 row and Nr columns.

// Add the source samples at its cell (thread 0) and, when row >= 0, record every receiver in that row
// of the seismograms (thread r for receiver r): what injSrc of the OpenCL files does, in one launch.
extern "C" __global__ void inject_and_record(float *vx, float *vz, float *taux, float *tauz, const float *tauxz,
                                             float *seismogram_vxi, float *seismogram_vzi,
                                             float *seismogram_tauxi, float *seismogram_tauzi,
                                             float *seismogram_tauxzi, const int *cells, int source_cell,
                                             float vsrcx, float vsrcz, float tsrcx, float tsrcz, int row) {
    const int r = column_of_thread();
    if (row_of_thread() != 0) return;
    if (r == 0) {
        vx[source_cell] += vsrcx;
        vz[source_cell] += vsrcz;
        taux[source_cell] += tsrcx;
        tauz[source_cell] += tsrcz;
    }
    if (row < 0 || r >= Nr || cells[r] < 0) return;
    const int cell = cells[r];
    seismogram_vxi[row * Nr + r] = vx[cell];
    seismogram_vzi[row * Nr + r] = vz[cell];
    seismogram_tauxi[row * Nr + r] = taux[cell];
    seismogram_tauzi[row * Nr + r] = tauz[cell];
    seismogram_tauxzi[row * Nr + r] = tauxz[cell];
}

// Add the residual of every receiver to the adjoint fields at its cell (Adj_injSrc of the OpenCL
// files). The first receiver of a cell adds those of the receivers that share it, in their order
// (first, following: PyFWI.engine.propagation.receiver_table), so that no two threads write one
// cell: deterministic without atomic operations.
extern "C" __global__ void inject_residual(float *Avx, float *Avz, float *Ataux, float *Atauz, float *Atauxz,
                                           const float *res_vx, const float *res_vz, const float *res_taux,
                                           const float *res_tauz, const float *res_tauxz, const int *cells,
                                           const int *first, const int *following) {
    const int r = column_of_thread();
    if (row_of_thread() != 0 || r >= Nr || cells[r] < 0 || !first[r]) return;
    float sum_vx = res_vx[r], sum_vz = res_vz[r], sum_taux = res_taux[r], sum_tauz = res_tauz[r],
          sum_tauxz = res_tauxz[r];
    for (int q = following[r]; q >= 0; q = following[q]) {
        sum_vx += res_vx[q];
        sum_vz += res_vz[q];
        sum_taux += res_taux[q];
        sum_tauz += res_tauz[q];
        sum_tauxz += res_tauxz[q];
    }
    const int cell = cells[r];
    Avx[cell] += sum_vx;
    Avz[cell] += sum_vz;
    Ataux[cell] += sum_taux;
    Atauz[cell] += sum_tauz;
    Atauxz[cell] += sum_tauxz;
}

"""Verificação do ambiente do fork PyFWI-CUDA.

Roda nos dois ambientes (envs/ref-0.1.10 e envs/dev) e reporta:
  1. versões de Python e das dependências;
  2. dispositivos OpenCL;
  3. CUDA (só se o CuPy estiver instalado): kernel NVRTC, CUDA Graph, cuFFT
     contra NumPy e cupyx contra SciPy;
  4. smoke test da PyFWI: forward + gradiente em cada dispositivo OpenCL,
     com checagem de valores finitos e de determinismo bit a bit;
  5. guarda de precisão: instruções FP64 no PTX dos kernels da PyFWI
     compilados como CUDA pelo NVRTC.

Uso:
  ../envs/dev/bin/python tools/check_env.py [--skip-pyfwi] [--repeats 2]

Sai com código 1 se alguma checagem falhar. Problemas conhecidos da 0.1.10
aparecem como AVISO e não contam como falha.
"""
import argparse
import importlib
import os
import platform
import re
import sys
import time
import warnings

import numpy as np

FAILURES = []

# Cabeçalho de portabilidade OpenCL -> CUDA proposto no plano (seção "Arquitetura-alvo").
CUDA_PORT_HEADER = r"""
#define __kernel extern "C" __global__
#define __global
#define __local __shared__
__device__ __forceinline__ int get_global_id(int d) {
    return d == 0 ? (int)(blockIdx.y * blockDim.y + threadIdx.y)
                  : (int)(blockIdx.x * blockDim.x + threadIdx.x);
}
"""

# Na 0.1.10 os estênceis leem fora dos buffers (linha Nz e i - 3 com i = 2).
# Na NVIDIA a leitura calha de ser finita; no POCL vira NaN e varia entre execuções.
POCL_OOB_ISSUE = "bug conhecido da 0.1.10: leitura fora do buffer (corrigido no modo fixed)"


def check(name, ok, detail="", known_issue=None):
    """Registra uma checagem; com known_issue, a falha vira aviso."""
    if ok:
        tag = "OK   "
    elif known_issue:
        tag = "AVISO"
        detail = f"{detail}; {known_issue}" if detail else known_issue
    else:
        tag = "FALHA"
        FAILURES.append(name)
    print(f"  [{tag}] {name}{(': ' + detail) if detail else ''}")
    return ok


def section(title):
    print(f"\n== {title} ==")


def versions():
    section("Versões")
    print(f"  Python {platform.python_version()} ({sys.executable})")
    for mod in ["numpy", "scipy", "matplotlib", "pyopencl", "torch", "cupy", "cuda.core", "PyFWI"]:
        try:
            m = importlib.import_module(mod)
            v = getattr(m, "__version__", None) or getattr(m, "VERSION_TEXT", "?")
            where = os.path.dirname(getattr(m, "__file__", "") or "")
            print(f"  {mod:<11} {v:<14} {where if mod == 'PyFWI' else ''}")
        except Exception as e:  # noqa: BLE001
            print(f"  {mod:<11} ausente ({type(e).__name__})")
    try:
        from scipy.optimize.optimize import MemoizeJac  # noqa: F401
        print("  scipy.optimize.optimize.MemoizeJac: disponível")
    except ImportError:
        print("  scipy.optimize.optimize.MemoizeJac: AUSENTE (fwi.py/tl_fwi.py/optimization.py da 0.1.10 não importam)")


def opencl_devices():
    section("Dispositivos OpenCL")
    import pyopencl as cl
    found = []
    for i, p in enumerate(cl.get_platforms()):
        for j, d in enumerate(p.get_devices()):
            fp64 = "fp64" if d.double_fp_config else "sem fp64"
            print(f"  {i}:{j}  {p.name} | {d.name} | {d.version} | {fp64}")
            found.append((i, j, p.name, d.name))
    check("ao menos um dispositivo OpenCL", bool(found))
    check("OpenCL NVIDIA presente", any("NVIDIA" in f[2] for f in found))
    check("POCL (CPU) presente", any("Portable" in f[2] for f in found))
    return found


def cuda_checks():
    section("CUDA (CuPy)")
    try:
        import cupy as cp
        from cupy.cuda import nvrtc, runtime
    except ImportError:
        print("  CuPy ausente: pulando (esperado no ambiente de referência)")
        return False
    p = runtime.getDeviceProperties(0)
    drv, rt = runtime.driverGetVersion(), runtime.runtimeGetVersion()
    print(f"  {p['name'].decode()} sm_{p['major']}{p['minor']}, {p['totalGlobalMem'] / 2**30:.1f} GB, "
          f"PCI {runtime.deviceGetPCIBusId(0)}")
    print(f"  driver CUDA {drv // 1000}.{drv % 1000 // 10}, runtime {rt // 1000}.{rt % 1000 // 10}, "
          f"NVRTC {'.'.join(map(str, nvrtc.getVersion()))}")
    check("driver suporta CUDA 13 (série 580+)", drv >= 13000)
    check("GPU RTX-alvo (sm_75 ou mais nova)", p["major"] * 10 + p["minor"] >= 75)

    k = cp.RawKernel(r'''extern "C" __global__ void axpy(float a, const float* x, float* y, int n){
      int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) y[i] = a * x[i] + y[i]; }''', "axpy")
    n = 1 << 20
    x = cp.random.default_rng(0).standard_normal(n, dtype=cp.float32)
    y = cp.ones(n, dtype=cp.float32)
    grid = ((n + 255) // 256,)
    k(grid, (256,), (np.float32(2), x, y, np.int32(n)))
    check("RawKernel (NVRTC -> cubin) executa", bool(cp.allclose(y, 2 * x + 1)))

    s = cp.cuda.Stream(non_blocking=True)
    with s:
        s.begin_capture()
        for _ in range(10):
            k(grid, (256,), (np.float32(1), x, y, np.int32(n)))
        g = s.end_capture()
    g.launch(stream=s)
    s.synchronize()
    check("CUDA Graph: captura e lançamento", bool(cp.allclose(y, 12 * x + 1, rtol=1e-5, atol=1e-5)))

    xh = cp.asnumpy(x).astype(np.float64)
    err = np.linalg.norm(cp.asnumpy(cp.fft.rfft(x)) - np.fft.rfft(xh)) / np.linalg.norm(np.fft.rfft(xh))
    check("cuFFT (float32) contra NumPy (float64)", err < 1e-5, f"erro relativo L2 {err:.1e}")

    from cupyx.scipy.ndimage import gaussian_filter as gf_gpu
    from scipy.ndimage import gaussian_filter as gf_cpu
    img = np.random.default_rng(1).standard_normal((128, 96)).astype(np.float32)
    err = np.abs(cp.asnumpy(gf_gpu(cp.asarray(img), 2)) - gf_cpu(img, 2)).max()
    check("cupyx gaussian_filter contra SciPy", err < 1e-5, f"erro máx. {err:.1e}")
    return True


def count_fp64(ptx):
    """Conta instruções FP64 por kernel (entrada .entry) no PTX."""
    counts, current = {}, None
    for line in ptx.splitlines():
        m = re.match(r"\s*\.visible \.entry (\w+)\(", line)
        if m:
            current = m.group(1)
            counts[current] = 0
        elif current and re.search(r"^\s*[a-z]+(\.[a-z0-9]+)*\.f64\b", line):
            counts[current] += 1
    return counts


def compile_ptx(source, arch="sm_86"):
    """Compila uma fonte OpenCL da PyFWI como CUDA (NVRTC) e devolve o PTX."""
    from cuda.core import Device, Program, ProgramOptions
    Device(0).set_current()
    with warnings.catch_warnings():
        # O PTX de um NVRTC mais novo que o driver não carrega, mas aqui só é lido.
        warnings.simplefilter("ignore", RuntimeWarning)
        ptx = Program(CUDA_PORT_HEADER + source, "c++", ProgramOptions(arch=arch)).compile("ptx")
    return ptx.code.decode() if isinstance(ptx.code, bytes) else ptx.code


def ptx_guard(sources):
    try:
        import cuda.core  # noqa: F401
    except ImportError:
        print("  cuda.core ausente: pulando a guarda de precisão")
        return
    for name, src in sources.items():
        try:
            counts = count_fp64(compile_ptx(src))
            check(f"NVRTC compila {name} com o cabeçalho de portabilidade", bool(counts),
                  ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
        except Exception as e:  # noqa: BLE001
            check(f"NVRTC compila {name} com o cabeçalho de portabilidade", False, str(e)[:300])


def test_model(nz=100, nx=100, smooth=0):
    """Modelo elástico com uma inclusão circular (estilo Louboutin), sem depender
    de PyFWI.model_dataset, que não importa com NumPy 2 na 0.1.10."""
    from scipy.ndimage import gaussian_filter
    zz, xx = np.mgrid[:nz, :nx]
    vp = np.where((zz - nz // 2) ** 2 + (xx - nx // 2) ** 2 <= (nz // 7) ** 2, 3000.0, 2500.0)
    if smooth:
        vp = gaussian_filter(vp, smooth * 10)
    vs = vp / np.sqrt(3.0)              # razão de Poisson 0,25
    rho = 310.0 * vp ** 0.25            # Gardner, kg/m³
    return {k: v.astype(np.float32) for k, v in {"vp": vp, "vs": vs, "rho": rho}.items()}


def pyfwi_case(platform_id, sdo, components=0):
    import PyFWI.acquisition as acq
    import PyFWI.fwi_tools as tools
    import PyFWI.processing as process
    import PyFWI.wave_propagation as wave

    model, m0 = test_model(), test_model(smooth=1)
    shape = model["vp"].shape
    inpa = {"ns": 4, "sdo": sdo, "fdom": 15, "dh": 7, "dt": 0.004, "acq_type": 1, "t": 0.6,
            "npml": 20, "pmlR": 1e-5, "pml_dir": 2, "rec_dis": 7}
    # Na 0.1.10, inpa['device'] falha em plataformas com um só dispositivo
    # (create_some_context acusa escolha "left over"); PYOPENCL_CTX evita isso.
    os.environ["PYOPENCL_CTX"] = f"{platform_id}"
    src_loc, rec_loc = acq.surface_seismic(inpa["ns"], inpa["rec_dis"], 7 * shape[1], 7, sdo)
    src_loc[:, 1] -= 5 * inpa["dh"]
    src = acq.Source(src_loc, inpa["dh"], inpa["dt"])
    src.Ricker(inpa["fdom"])

    t0 = time.perf_counter()
    W = wave.WavePropagator(inpa, src, rec_loc, shape, components=components, chpr=20)
    d_obs = W.forward_modeling(model, False)
    t_build_fwd = time.perf_counter() - t0
    d_obs = process.prepare_residual(d_obs, 1)

    L = wave.WavePropagator(inpa, src, rec_loc, shape, components=components, chpr=20)
    t0 = time.perf_counter()
    d_est = process.prepare_residual(L.forward_modeling(m0, False), 1)
    t_fwd = time.perf_counter() - t0
    _, adj = tools.cost_seismic(d_est, d_obs, fun=tools.CostFunction("l2"))
    t0 = time.perf_counter()
    grad = L.gradient(adj, False)
    t_grad = time.perf_counter() - t0
    out = {f"d_{k}": np.asarray(v) for k, v in d_obs.items()}
    out.update({f"g_{k}": np.asarray(v) for k, v in grad.items()})
    return out, (t_build_fwd, t_fwd, t_grad), L


def outputs_ok(out):
    """Tudo finito, ao menos um sismograma não nulo e todos os gradientes não nulos.
    Com components=0 só a pressão (taux, tauz) é gravada; os outros ficam em zero."""
    finite = all(np.isfinite(v).all() for v in out.values())
    data = any(np.abs(v).max() > 0 for k, v in out.items() if k.startswith("d_"))
    grads = all(np.abs(v).max() > 0 for k, v in out.items() if k.startswith("g_"))
    return finite and data and grads


def pyfwi_smoke(devices, repeats):
    section("Smoke test PyFWI: forward + gradiente (100x100, 4 tiros, chpr=20)")
    results, sources = {}, None
    for (pi, di, pname, dname) in devices:
        known = POCL_OOB_ISSUE if "Portable" in pname else None
        for sdo in (4, 8):  # inpa['sdo'] é a ordem espacial
            label = f"{pi}:{di} {dname.split('-')[0][:28]} ordem {sdo}"
            try:
                runs = []
                for _ in range(repeats):
                    out, times, L = pyfwi_case(pi, sdo)
                    runs.append(out)
                if sources is None and "NVIDIA" in pname and sdo == 8:
                    s, s_cw, s_surf = L.kernel_caller()
                    sources = {"elastic.cl": s, "elastic_crosswell.cl": s_cw, "elastic_surface.cl": s_surf}
                check(f"{label}: saídas finitas e não nulas", outputs_ok(runs[0]), known_issue=known)
                same = all(np.array_equal(r[k], runs[0][k]) for r in runs[1:] for k in runs[0])
                check(f"{label}: determinismo bit a bit em {repeats} execuções", same, known_issue=known)
                print(f"          tempos: construção+forward {times[0]:.2f}s | forward {times[1]:.2f}s | "
                      f"gradiente {times[2]:.2f}s (último run)")
                results[(pi, sdo)] = runs[0]
            except Exception as e:  # noqa: BLE001
                check(f"{label}: execução", False, f"{type(e).__name__}: {str(e)[:300]}")
    return sources


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-pyfwi", action="store_true", help="não roda o smoke test da PyFWI")
    ap.add_argument("--repeats", type=int, default=2, help="execuções por caso para checar determinismo")
    args = ap.parse_args()

    versions()
    devices = opencl_devices()
    has_cuda = cuda_checks()
    sources = None
    if not args.skip_pyfwi:
        sources = pyfwi_smoke(devices, args.repeats)
    if has_cuda and sources:
        section("Guarda de precisão: FP64 no PTX dos kernels (fonte atual, ordem 8)")
        ptx_guard(sources)

    section("Resultado")
    if FAILURES:
        print(f"  {len(FAILURES)} falha(s):")
        for f in FAILURES:
            print(f"   - {f}")
        sys.exit(1)
    print("  Todas as checagens passaram.")


if __name__ == "__main__":
    main()

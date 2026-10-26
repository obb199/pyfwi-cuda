"""Environment checker for the PyFWI-CUDA fork.

What this script is for
-----------------------
The migration of PyFWI from OpenCL to CUDA uses two Python environments
(see ``tools/setup_envs.sh`` and ``docs/cuda-migration/execution-roadmap.md``):

* ``envs/ref-0.1.10`` -- the frozen *reference* environment, with PyFWI 0.1.10
  exactly as published on PyPI. It produces the reference ("golden") results.
* ``envs/dev`` -- the *development* environment, with this repository installed
  in editable mode plus CuPy / CUDA 13;
* ``envs/cuda12`` -- the same with CuPy / CUDA 12 (roadmap step 2.6), to check the
  CUDA backend with a CUDA 12 runtime and NVRTC 12.

This script answers one question: "is this environment able to run PyFWI and the
CUDA tool chain correctly?" It runs five groups of checks and prints one line
per check:

1. **Versions** of Python and of the key packages.
2. **OpenCL devices** visible to PyOpenCL (NVIDIA GPU and/or POCL on the CPU).
3. **CUDA** (only if CuPy is installed): a kernel compiled by NVRTC, a CUDA
   Graph, cuFFT against NumPy and ``cupyx.scipy.ndimage`` against SciPy.
4. **PyFWI smoke test**: forward modelling + gradient on every OpenCL device,
   checking that the outputs are finite and bit-for-bit deterministic.
5. **Precision guard**: the PyFWI OpenCL kernels are compiled as CUDA with the
   portability header and the FP64 instructions in the PTX are counted.

Output format
-------------
Each check prints ``[OK   ]``, ``[FAIL ]``, ``[WARN ]`` (a *known* bug of
PyFWI 0.1.10, not a problem of the environment) or ``[INFO ]`` (something
skipped on purpose). The script exits with code 1 if any check failed.

Usage
-----
::

    ../envs/dev/bin/python tools/check_env.py              # full check
    ../envs/dev/bin/python tools/check_env.py --skip-pyfwi # versions/devices/CUDA only
    ../envs/dev/bin/python tools/check_env.py --no-gpu     # machine without NVIDIA (CI)

Used by
-------
* Developers, after creating or updating an environment.
* ``test/test_check_env.py``, which imports the helper functions below and also
  runs this file as a command-line program.
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

#: Names of the checks that failed so far, in order. ``check()`` appends to it
#: and ``main()`` reads it at the end to decide the exit code. Tests clear it
#: before and after use because it is module-level state.
FAILURES = []

#: Portability header that lets the OpenCL C kernels of PyFWI compile as CUDA
#: C++ with NVRTC (proposed in the "Target architecture" section of the
#: analysis document). It maps the OpenCL qualifiers to CUDA ones and
#: re-implements ``get_global_id``:
#:
#: * dimension 0 (OpenCL) -> row index ``i`` (depth, z) -> ``blockIdx.y/threadIdx.y``;
#: * dimension 1 (OpenCL) -> column index ``j`` (x)    -> ``blockIdx.x/threadIdx.x``.
#:
#: Putting ``j`` -- the direction that is contiguous in memory -- on
#: ``threadIdx.x`` makes neighbouring threads of a warp read neighbouring
#: addresses (coalesced access).
#:
#: The header is the one of the package, ``src/PyFWI/kernels/opencl_on_cuda.h``, which the CUDA
#: backend uses too (step 2.3); it is read from the repository, so that this tool also works in the
#: reference environment. Used by ``compile_ptx()`` (this module) and the GPU tests.
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "PyFWI", "kernels",
                       "opencl_on_cuda.h")) as _header:
    CUDA_PORT_HEADER = _header.read()

#: Explanation attached to the POCL warnings. In PyFWI 0.1.10 the finite
#: difference stencils read outside the field buffers (row ``Nz``, past the end,
#: and ``i - 3`` with ``i = 2``, before the start). On the NVIDIA driver those
#: reads happen to return finite values; on POCL they return garbage that turns
#: into NaN and changes from run to run. Used by ``pyfwi_smoke()``.
POCL_OOB_ISSUE = "known bug of 0.1.10: out-of-bounds buffer reads (fixed in the 'fixed' numerics mode)"


def check(name, ok, detail="", known_issue=None):
    """Record and print the result of one check.

    Parameters
    ----------
    name : str
        Short human-readable description, e.g. ``"POCL (CPU) present"``.
    ok : bool
        Whether the check passed.
    detail : str, optional
        Extra information printed after the name (numbers, error message).
    known_issue : str or None, optional
        If given and the check failed, the failure is a known bug of
        PyFWI 0.1.10 rather than an environment problem: it is printed as
        ``[WARN ]`` with this explanation and is **not** added to
        ``FAILURES``.

    Returns
    -------
    bool
        ``ok``, unchanged, so callers can write ``if check(...):``.

    Used by
    -------
    Every check in this module, and ``test_check_known_issue_is_warning_not_failure``.
    """
    if ok:
        tag = "OK   "
    elif known_issue:
        tag = "WARN "
        detail = f"{detail}; {known_issue}" if detail else known_issue
    else:
        tag = "FAIL "
        FAILURES.append(name)
    print(f"  [{tag}] {name}{(': ' + detail) if detail else ''}")
    return ok


def section(title):
    """Print a section header such as ``== Versions ==``. Used by every check group."""
    print(f"\n== {title} ==")


def versions():
    """Print the versions of Python and of the packages that matter for PyFWI.

    It also reports whether ``scipy.optimize.optimize.MemoizeJac`` exists:
    PyFWI 0.1.10 imports it in ``fwi.py``, ``tl_fwi.py`` and
    ``optimization.py``, and recent SciPy versions no longer export it.

    Used by ``main()``. Prints only; never fails.
    """
    section("Versions")
    print(f"  Python {platform.python_version()} ({sys.executable})")
    for mod in ["numpy", "scipy", "matplotlib", "pyopencl", "torch", "cupy", "cuda.core", "PyFWI"]:
        try:
            m = importlib.import_module(mod)
            # PyOpenCL exposes its version as VERSION_TEXT; the others as __version__.
            v = getattr(m, "__version__", None) or getattr(m, "VERSION_TEXT", "?")
            # For PyFWI, also show where it was imported from (PyPI copy or this repo).
            where = os.path.dirname(getattr(m, "__file__", "") or "")
            print(f"  {mod:<11} {v:<14} {where if mod == 'PyFWI' else ''}")
        except Exception as e:  # noqa: BLE001 - any import problem is just reported
            print(f"  {mod:<11} missing ({type(e).__name__})")
    try:
        from scipy.optimize.optimize import MemoizeJac  # noqa: F401
        print("  scipy.optimize.optimize.MemoizeJac: available")
    except ImportError:
        print("  scipy.optimize.optimize.MemoizeJac: MISSING (fwi.py/tl_fwi.py/optimization.py of 0.1.10 cannot import)")


def opencl_devices(require_gpu=True):
    """List the OpenCL platforms/devices and check that the expected ones exist.

    Parameters
    ----------
    require_gpu : bool, optional
        If True (default), a missing NVIDIA OpenCL platform is a failure.
        If False (``--no-gpu``), it is only reported as ``[INFO ]``.

    Returns
    -------
    list of tuple
        One tuple ``(platform_index, device_index, platform_name, device_name)``
        per device. ``platform_index`` is what ``PYOPENCL_CTX`` expects.

    Used by
    -------
    ``main()``; the returned list feeds ``pyfwi_smoke()``.
    """
    section("OpenCL devices")
    import pyopencl as cl
    found = []
    for i, p in enumerate(cl.get_platforms()):
        for j, d in enumerate(p.get_devices()):
            fp64 = "fp64" if d.double_fp_config else "no fp64"
            print(f"  {i}:{j}  {p.name} | {d.name} | {d.version} | {fp64}")
            found.append((i, j, p.name, d.name))
    check("at least one OpenCL device", bool(found))
    has_nvidia = any("NVIDIA" in f[2] for f in found)
    if require_gpu or has_nvidia:
        check("NVIDIA OpenCL present", has_nvidia)
    else:
        print("  [INFO ] NVIDIA OpenCL missing (--no-gpu)")
    check("POCL (CPU) present", any("Portable" in f[2] for f in found))
    return found


def cuda_checks(require_gpu=True):
    """Exercise the CUDA tool chain through CuPy on GPU 0.

    The checks are, in order: driver supports CUDA 13; the GPU is an RTX-class
    architecture (sm_75 or newer); a raw kernel compiled by NVRTC runs; a CUDA
    Graph can be captured and replayed; cuFFT matches NumPy; and
    ``cupyx.scipy.ndimage.gaussian_filter`` matches SciPy.

    Parameters
    ----------
    require_gpu : bool, optional
        If False (``--no-gpu``), a machine without a visible NVIDIA GPU is
        reported as ``[INFO ]`` instead of failing.

    Returns
    -------
    bool
        True if the CUDA checks ran (CuPy installed and a GPU visible).
        ``main()`` uses it to decide whether to run ``ptx_guard()``.

    Used by
    -------
    ``main()``.
    """
    section("CUDA (CuPy)")
    try:
        import cupy as cp
        from cupy.cuda import nvrtc, runtime
    except ImportError:
        print("  CuPy missing: skipping (expected in the reference environment)")
        return False
    try:
        n_gpus = runtime.getDeviceCount()
    except runtime.CUDARuntimeError:
        n_gpus = 0
    if not n_gpus:
        if require_gpu:
            check("NVIDIA GPU visible to CUDA", False)
        else:
            print("  [INFO ] no NVIDIA GPU visible (--no-gpu)")
        return False

    # Device properties: name, compute capability (major/minor) and memory.
    p = runtime.getDeviceProperties(0)
    # Versions are encoded as 1000*major + 10*minor (13000 -> 13.0).
    drv, rt = runtime.driverGetVersion(), runtime.runtimeGetVersion()
    print(f"  {p['name'].decode()} sm_{p['major']}{p['minor']}, {p['totalGlobalMem'] / 2**30:.1f} GB, "
          f"PCI {runtime.deviceGetPCIBusId(0)}")
    print(f"  CUDA driver {drv // 1000}.{drv % 1000 // 10}, runtime {rt // 1000}.{rt % 1000 // 10}, "
          f"NVRTC {'.'.join(map(str, nvrtc.getVersion()))}")
    check("driver supports CUDA 13 (series 580+)", drv >= 13000)
    check("RTX-class GPU (sm_75 or newer)", p["major"] * 10 + p["minor"] >= 75)

    # y <- a*x + y on 2**20 elements; one thread per element, 256 threads per block.
    k = cp.RawKernel(r'''extern "C" __global__ void axpy(float a, const float* x, float* y, int n){
      int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) y[i] = a * x[i] + y[i]; }''', "axpy")
    n = 1 << 20
    x = cp.random.default_rng(0).standard_normal(n, dtype=cp.float32)
    y = cp.ones(n, dtype=cp.float32)
    grid = ((n + 255) // 256,)  # number of blocks, rounded up
    k(grid, (256,), (np.float32(2), x, y, np.int32(n)))
    check("RawKernel (NVRTC -> cubin) runs", bool(cp.allclose(y, 2 * x + 1)))

    # Capture 10 launches (each adds x once) into a graph and replay it once:
    # y goes from 2x + 1 to 12x + 1.
    s = cp.cuda.Stream(non_blocking=True)
    with s:
        s.begin_capture()
        for _ in range(10):
            k(grid, (256,), (np.float32(1), x, y, np.int32(n)))
        g = s.end_capture()
    g.launch(stream=s)
    s.synchronize()
    check("CUDA Graph: capture and launch", bool(cp.allclose(y, 12 * x + 1, rtol=1e-5, atol=1e-5)))

    # cuFFT in float32 against NumPy in float64; the error is the relative L2 norm.
    xh = cp.asnumpy(x).astype(np.float64)
    err = np.linalg.norm(cp.asnumpy(cp.fft.rfft(x)) - np.fft.rfft(xh)) / np.linalg.norm(np.fft.rfft(xh))
    check("cuFFT (float32) vs NumPy (float64)", err < 1e-5, f"relative L2 error {err:.1e}")

    from cupyx.scipy.ndimage import gaussian_filter as gf_gpu
    from scipy.ndimage import gaussian_filter as gf_cpu
    img = np.random.default_rng(1).standard_normal((128, 96)).astype(np.float32)
    err = np.abs(cp.asnumpy(gf_gpu(cp.asarray(img), 2)) - gf_cpu(img, 2)).max()
    check("cupyx gaussian_filter vs SciPy", err < 1e-5, f"max error {err:.1e}")
    return True


def count_fp64(ptx):
    """Count the FP64 (double precision) instructions of each kernel in a PTX listing.

    PTX is the assembly-like text that NVRTC produces. Each kernel starts with
    a ``.visible .entry <name>(`` line. An instruction is FP64 if its type
    suffix is ``.f64`` (``mul.f64``, ``cvt.f64.f32``, ``ld.global.f64`` ...).
    Register declarations such as ``.reg .f64 %fd<4>;`` start with a dot and
    are **not** counted.

    Why it matters: on GeForce RTX cards FP64 runs at 1/64 of the FP32 rate,
    so a hot kernel with FP64 instructions (caused by literals without the
    ``f`` suffix) is much slower than necessary.

    Parameters
    ----------
    ptx : str
        PTX source text.

    Returns
    -------
    dict[str, int]
        Kernel name -> number of FP64 instructions. Empty if there is no kernel.

    Used by
    -------
    ``ptx_guard()`` and the GPU tests in ``test/test_check_env.py``.
    """
    counts, current = {}, None  # current: name of the kernel being read
    for line in ptx.splitlines():
        m = re.match(r"\s*\.visible \.entry (\w+)\(", line)
        if m:
            current = m.group(1)
            counts[current] = 0
        elif current and re.search(r"^\s*[a-z]+(\.[a-z0-9]+)*\.f64\b", line):
            counts[current] += 1
    return counts


def cuda_include_dir():
    """The folder of the CUDA headers installed by pip (``nvidia/cu*/include``, with ``cuda_fp16.h``), or None.

    CuPy finds them by itself; NVRTC called through ``cuda.core`` (``compile_ptx``) needs the path to compile
    the half-precision storage of the native kernels (step 4.3).
    """
    import glob
    import site
    for root in site.getsitepackages():
        found = sorted(glob.glob(os.path.join(root, "nvidia", "cu*", "include", "cuda_fp16.h")))
        if found:
            return os.path.dirname(found[-1])
    return None


def compile_ptx(source, arch="sm_86", include_path=None):
    """Compile one PyFWI OpenCL source as CUDA with NVRTC and return the PTX text.

    Parameters
    ----------
    source : str
        OpenCL C source, i.e. the ``#define`` block generated by
        ``WavePreparation.kernel_caller()`` followed by a ``.cl`` file.
        ``CUDA_PORT_HEADER`` is prepended automatically.
    arch : str, optional
        Target GPU architecture. ``sm_86`` is Ampere (RTX 30 series), the GPU
        of the development machine.
    include_path : str, optional
        Folder of headers that the source includes (``cuda_include_dir()`` for ``cuda_fp16.h``).

    Returns
    -------
    str
        The PTX listing.

    Notes
    -----
    The NVRTC installed by pip (13.4) is newer than the driver of the
    development machine (CUDA 13.0), so this PTX could not be *loaded* on the
    GPU; that is fine because it is only *read* here. CUDA's warning about it
    is silenced. Kernels that run are compiled to cubin by CuPy instead.

    Used by
    -------
    ``ptx_guard()`` and the GPU tests in ``test/test_check_env.py``.
    """
    from cuda.core import Device, Program, ProgramOptions
    Device(0).set_current()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ptx = Program(CUDA_PORT_HEADER + source, "c++", ProgramOptions(arch=arch, include_path=include_path)).compile("ptx")
    return ptx.code.decode() if isinstance(ptx.code, bytes) else ptx.code


def ptx_guard(sources):
    """Compile each PyFWI kernel file as CUDA and print its FP64 counts.

    Parameters
    ----------
    sources : dict[str, str]
        File name (``"elastic.cl"`` ...) -> generated OpenCL source, as returned
        by ``pyfwi_smoke()``.

    Used by
    -------
    ``main()``, when CUDA is available and the smoke test ran on NVIDIA.
    """
    try:
        import cuda.core  # noqa: F401
    except ImportError:
        print("  cuda.core missing: skipping the precision guard")
        return
    for name, src in sources.items():
        try:
            counts = count_fp64(compile_ptx(src))
            check(f"NVRTC compiles {name} with the portability header", bool(counts),
                  ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
        except Exception as e:  # noqa: BLE001 - report the compiler message
            check(f"NVRTC compiles {name} with the portability header", False, str(e)[:300])


def inclusion_model(nz=100, nx=100, smooth=0):
    """Build a small elastic model with a circular high-velocity inclusion.

    It mimics the "louboutin" model of PyFWI without importing
    ``PyFWI.model_dataset``, which fails to import with NumPy 2 in 0.1.10.

    Parameters
    ----------
    nz, nx : int, optional
        Number of grid points in depth (z) and in x.
    smooth : int, optional
        0 for the true model. ``smooth > 0`` applies a Gaussian filter of
        standard deviation ``10 * smooth`` grid points to vp, producing a
        smooth starting model for the gradient.

    Returns
    -------
    dict[str, numpy.ndarray]
        ``{"vp", "vs", "rho"}``, each ``float32`` with shape ``(nz, nx)``:

        * ``vp`` -- P-wave velocity in m/s: 2500 in the background and 3000
          inside a circle of radius ``nz // 7`` centred in the model;
        * ``vs`` -- S-wave velocity in m/s, ``vp / sqrt(3)`` (Poisson's ratio 0.25);
        * ``rho`` -- density in kg/m^3 from Gardner's relation ``310 * vp**0.25``.

    Used by
    -------
    ``pyfwi_case()`` and ``test_inclusion_model_is_elastic_and_float32``.
    """
    from scipy.ndimage import gaussian_filter
    zz, xx = np.mgrid[:nz, :nx]  # row (depth) and column indices of every grid point
    vp = np.where((zz - nz // 2) ** 2 + (xx - nx // 2) ** 2 <= (nz // 7) ** 2, 3000.0, 2500.0)
    if smooth:
        vp = gaussian_filter(vp, smooth * 10)
    vs = vp / np.sqrt(3.0)
    rho = 310.0 * vp ** 0.25
    return {k: v.astype(np.float32) for k, v in {"vp": vp, "vs": vs, "rho": rho}.items()}


def pyfwi_case(platform_id, sdo, components=0):
    """Run one forward modelling + one gradient with the installed PyFWI.

    The setup follows the gradient example of the PyFWI documentation: 4
    surface sources, receivers every grid point, a 100x100 model with a
    20-point PML, 15 Hz Ricker wavelet and 0.6 s of recording.

    Parameters
    ----------
    platform_id : int
        OpenCL platform index (first element of the tuples returned by
        ``opencl_devices()``). It is passed through ``PYOPENCL_CTX`` because
        ``inpa['device']`` fails on platforms with a single device in 0.1.10.
    sdo : int
        Spatial order of the finite differences: 4 or 8. (Inside PyFWI the
        attribute ``self.sdo`` stores half of it.)
    components : int, optional
        Which seismograms are recorded (PyFWI convention): 0 pressure,
        1 taux, 2 vx/vz, 3 stresses, 4 everything.

    Returns
    -------
    out : dict[str, numpy.ndarray]
        ``"d_<component>"``: observed seismograms (true model), each with shape
        ``(nt, nr * ns)``; ``"g_vp"``, ``"g_vs"``, ``"g_rho"``: gradient of the
        L2 misfit with respect to vp, vs and rho, shape ``(nz, nx)``.
    times : tuple of float
        Wall-clock seconds for (propagator construction + first forward,
        second forward, gradient).
    L : PyFWI.wave_propagation.WavePropagator
        The propagator used for the gradient; ``pyfwi_smoke()`` reads its
        generated kernel sources.

    Used by
    -------
    ``pyfwi_smoke()`` and ``test_pyfwi_smoke_on_nvidia_opencl_is_finite_and_deterministic``.
    """
    import PyFWI.acquisition as acq
    import PyFWI.fwi_tools as tools
    import PyFWI.processing as process
    import PyFWI.wave_propagation as wave

    model, m0 = inclusion_model(), inclusion_model(smooth=1)  # true model and starting model
    shape = model["vp"].shape
    # inpa: the PyFWI "input parameters" dictionary.
    inpa = {"ns": 4,             # number of sources (shots)
            "sdo": sdo,          # spatial order of the finite differences (4 or 8)
            "fdom": 15,          # dominant frequency of the Ricker wavelet, Hz
            "dh": 7,             # grid spacing, m
            "dt": 0.004,         # sampling interval of the recorded data, s
            "acq_type": 1,       # acquisition: 0 crosswell, 1 surface, 2 both
            "t": 0.6,            # recording length, s
            "npml": 20,          # thickness of the absorbing boundary, grid points
            "pmlR": 1e-5,        # target reflection coefficient of the boundary
            "pml_dir": 2,        # boundary on all sides
            "rec_dis": 7}        # receiver spacing, m
    os.environ["PYOPENCL_CTX"] = f"{platform_id}"
    src_loc, rec_loc = acq.surface_seismic(inpa["ns"], inpa["rec_dis"], 7 * shape[1], 7, sdo)
    src_loc[:, 1] -= 5 * inpa["dh"]  # move the sources 5 grid points up, as in the docs example
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
    _, adj = tools.cost_seismic(d_est, d_obs, fun=tools.CostFunction("l2"))  # adj: adjoint source
    t0 = time.perf_counter()
    grad = L.gradient(adj, False)
    t_grad = time.perf_counter() - t0
    out = {f"d_{k}": np.asarray(v) for k, v in d_obs.items()}
    out.update({f"g_{k}": np.asarray(v) for k, v in grad.items()})
    return out, (t_build_fwd, t_fwd, t_grad), L


def outputs_ok(out):
    """Tell whether the outputs of ``pyfwi_case()`` look physically sane.

    Sane means: every value is finite, at least one seismogram is non-zero and
    every gradient is non-zero. Some seismograms are legitimately all zero:
    with ``components=0`` only the pressure (taux, tauz) is recorded and
    ``prepare_residual`` fills the other components with zeros.

    Parameters
    ----------
    out : dict[str, numpy.ndarray]
        The first value returned by ``pyfwi_case()`` (keys ``d_*`` and ``g_*``).

    Returns
    -------
    bool

    Used by
    -------
    ``pyfwi_smoke()`` and the tests.
    """
    finite = all(np.isfinite(v).all() for v in out.values())
    data = any(np.abs(v).max() > 0 for k, v in out.items() if k.startswith("d_"))
    grads = all(np.abs(v).max() > 0 for k, v in out.items() if k.startswith("g_"))
    return finite and data and grads


def pyfwi_smoke(devices, repeats):
    """Run ``pyfwi_case()`` on every OpenCL device, in orders 4 and 8.

    For each (device, order) pair it checks that the outputs are sane
    (``outputs_ok``) and identical bit for bit across ``repeats`` runs. On POCL
    both checks are expected to fail in 0.1.10 and are reported as warnings
    (see ``POCL_OOB_ISSUE``).

    Parameters
    ----------
    devices : list of tuple
        As returned by ``opencl_devices()``.
    repeats : int
        Number of runs per case used for the determinism check (>= 2 to be meaningful).

    Returns
    -------
    dict[str, str] or None
        Generated OpenCL sources of the NVIDIA order-8 run, keyed by file name,
        for ``ptx_guard()``; None if there was no NVIDIA run.

    Used by
    -------
    ``main()``.
    """
    section("PyFWI smoke test: forward + gradient (100x100, 4 shots, chpr=20)")
    sources = None
    for (pi, di, pname, dname) in devices:
        known = POCL_OOB_ISSUE if "Portable" in pname else None
        for sdo in (4, 8):
            label = f"{pi}:{di} {dname.split('-')[0][:28]} order {sdo}"
            try:
                runs = []
                for _ in range(repeats):
                    out, times, L = pyfwi_case(pi, sdo)
                    runs.append(out)
                if sources is None and "NVIDIA" in pname and sdo == 8:
                    s, s_cw, s_surf = L.kernel_caller()
                    sources = {"elastic.cl": s, "elastic_crosswell.cl": s_cw, "elastic_surface.cl": s_surf}
                check(f"{label}: outputs finite and non-zero", outputs_ok(runs[0]), known_issue=known)
                same = all(np.array_equal(r[k], runs[0][k]) for r in runs[1:] for k in runs[0])
                check(f"{label}: bit-for-bit deterministic over {repeats} runs", same, known_issue=known)
                print(f"          times: build+forward {times[0]:.2f}s | forward {times[1]:.2f}s | "
                      f"gradient {times[2]:.2f}s (last run)")
            except Exception as e:  # noqa: BLE001 - report and keep checking other devices
                check(f"{label}: execution", False, f"{type(e).__name__}: {str(e)[:300]}")
    return sources


def main():
    """Command-line entry point: parse the options, run all check groups, set the exit code."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-pyfwi", action="store_true", help="do not run the PyFWI smoke test")
    ap.add_argument("--repeats", type=int, default=2, help="runs per case for the determinism check")
    ap.add_argument("--no-gpu", action="store_true",
                    help="machine without NVIDIA (e.g. CI): a missing GPU is not a failure")
    args = ap.parse_args()

    versions()
    devices = opencl_devices(require_gpu=not args.no_gpu)
    has_cuda = cuda_checks(require_gpu=not args.no_gpu)
    sources = None
    if not args.skip_pyfwi:
        sources = pyfwi_smoke(devices, args.repeats)
    if has_cuda and sources:
        section("Precision guard: FP64 in the PTX of the kernels (current source, order 8)")
        ptx_guard(sources)

    section("Result")
    if FAILURES:
        print(f"  {len(FAILURES)} failure(s):")
        for f in FAILURES:
            print(f"   - {f}")
        sys.exit(1)
    print("  All checks passed.")


if __name__ == "__main__":
    main()

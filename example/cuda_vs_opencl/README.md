# CUDA against OpenCL: see the difference yourself

Five notebooks that run the same seismic calculations on the two backends of PyFWI-CUDA, on your GPU, and show
the difference: speed, results, a complete inversion, where the time goes, and memory. They are the hands-on
side of [CUDA against OpenCL on the project's graphics card](../../docs/cuda-migration/cuda-vs-opencl.md),
which measures the same things more carefully and explains them for readers who know neither GPUs nor the code.

| Notebook | The question | What you see | Time on an RTX 3060 |
| --- | --- | --- | --- |
| [1_speed.ipynb](1_speed.ipynb) | How much faster is CUDA? | The backend chosen by one entry of `inpa`; forward modelling and gradient timed on four engines and three model sizes; how many times faster CUDA is, like for like and by default | about 4 minutes |
| [2_same_results.ipynb](2_same_results.ipynb) | Do both compute the same thing? | Seismograms and gradients of the four engines: the backends differ by rounding, the numeric modes by real corrections | under a minute |
| [3_inversion.ipynb](3_inversion.ipynb) | And for a complete inversion? | The same FWI on both backends: the time, the misfit per evaluation and per second, the models found | about a minute and a half |
| [4_time_per_step.ipynb](4_time_per_step.ipynb) | Where does the gain come from? | The time of one time step against the size of the model, several sources at once, the commands OpenCL sends | under a minute |
| [5_memory.ipynb](5_memory.ipynb) | What does it cost in memory? | Where the checkpoints live on each backend; boundary saving and FP16, the options only CUDA has, with their memory, time and accuracy | about a minute |

An *engine* is a backend (`opencl` or `cuda`) and a numeric mode (`legacy`, the arithmetic of PyFWI 0.1.10, or
`fixed`, its bugs corrected). OpenCL starts in `legacy` and CUDA in `fixed`, but each runs either mode, and a
notebook can run all four engines side by side: each propagator reads `inpa["backend"]` and `inpa["numerics"]`.

## Running them

1. **A machine with an NVIDIA GPU and PyFWI-CUDA installed with a CUDA extra** (`pip install ".[cuda13]"`, or
   `".[cuda12]"` for an older driver), which brings CuPy; PyOpenCL comes with PyFWI. The development
   environment of the repository has everything: `tools/setup_envs.sh dev`.
2. **A Jupyter kernel in that environment**: `pip install jupyterlab` there and run `jupyter lab` from this folder,
   or, in VS Code, `pip install ipykernel` there and choose that environment's Python as the kernel.
3. **Run the cells in order.** The first code cell of every notebook prints the four engines and the GPU each one
   runs on, or stops with the reason CUDA cannot run.

The notebooks import `comparison.py`, the code they share (the model, the acquisition, how a propagator is put on
an engine, how a run is timed): Jupyter runs a notebook from its own folder, so it finds it. Each notebook has a
*Settings* cell near the top: change the model sizes, the number of repetitions or the engines there.

Your numbers will not be exactly those of the final document: another GPU, another driver, the clock of the GPU
(which rises under load: each time is the median of several runs after an untimed one), and other programs
drawing on the GPU, such as the desktop, all change them a little. The picture should stay the same.

The test suite runs every notebook in a reduced size on a machine with CUDA (`test/test_notebooks.py`).

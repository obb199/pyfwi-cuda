# Self-hosted GPU runner

Roadmap step 0.9. GitHub's own machines have no NVIDIA GPU, so the tests that need the RTX
(NVIDIA OpenCL, CUDA, `compute-sanitizer`), the slow characterisation and the benchmarks run on a
*self-hosted runner*: a small program from GitHub that runs on the RTX machine, waits for jobs of
this repository and executes them. The workflow is ready in
[`.github/workflows/gpu.yml`](../../.github/workflows/gpu.yml); it stays manual-only until the runner
exists.

## Security first

A runner executes whatever code a workflow asks for, on your machine, with your user's rights.

- Register it **only for this private repository**, never for a public one: on a public repository
  anyone could open a pull request whose workflow runs on your computer.
- Run it under your normal user (not root). It needs no sudo.
- If the repository is ever made public, remove the runner first (step 5.4 of the roadmap).

## What the machine needs

- The NVIDIA driver (series 580 or newer) and the RTX visible to `nvidia-smi`.
- [uv](https://docs.astral.sh/uv/) on the `PATH` (`pipx install uv`): the jobs re-create the
  environments with `tools/setup_envs.sh`.
- Optional: Nsight Systems, Nsight Compute and `compute-sanitizer` on the `PATH` (they come with the
  CUDA Toolkit). Tests that need a missing tool skip themselves.

## Registering the runner

1. On GitHub, open the repository, then **Settings → Actions → Runners → New self-hosted runner**,
   and choose **Linux** and **x64**. GitHub shows a download command and a `config.sh` command with a
   one-time **token**; the token expires after one hour.
2. In a terminal on the RTX machine, in a folder of your choice (for example `~/actions-runner`), run
   the download and extraction commands shown by GitHub.
3. Configure it with the extra label `gpu`, which the workflow asks for:

   ```bash
   ./config.sh --url https://github.com/obb199/pyfwi-cuda --token <TOKEN> --labels gpu --name rtx3060
   ```

4. Start it: `./run.sh` in a terminal that stays open, or, to keep it running after a reboot, as a
   user service (see GitHub's documentation on `svc.sh`; that step needs sudo once).
5. Back in **Settings → Actions → Runners**, the runner must show as *Idle* with the labels
   `self-hosted`, `Linux`, `X64` and `gpu`.

## Turning the GPU workflow on

1. Run it once by hand: **Actions → GPU → Run workflow**. It must finish green.
2. Then uncomment the `push` and `schedule` triggers at the top of `.github/workflows/gpu.yml`, so
   that every push to `main` runs the GPU suite and the benchmark runs every night.
3. Check a failing case once (for example by pushing a branch with a deliberately broken test) to see
   the job fail, as the roadmap asks.

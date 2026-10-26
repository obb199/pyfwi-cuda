#!/usr/bin/env bash
# Re-create the Python environments of the project from the pinned versions in requirements/.
#
# Environments (created next to the repository, in $ENVS_DIR):
#
#   ref-0.1.10  Python 3.10 + PyFWI 0.1.10 from PyPI, with the package versions of the
#               original requirements.txt (SciPy 1.7.3 still exports MemoizeJac).
#               It produces the reference ("golden") results of Phase 0 and must never
#               receive code from this fork.
#   dev         Python 3.12 + this repository in editable mode + CuPy/CUDA 13 + PyOpenCL.
#               Used for development and for running the tests.
#   cuda12      The same with CuPy/CUDA 12 instead (the [cuda12] extra, roadmap step 2.6), to run
#               the parity tests with a CUDA 12 runtime and NVRTC 12; same versions of the rest.
#
# Usage:     tools/setup_envs.sh [ref|dev|cuda12|all]        (default: all)
#
# Variables: ENVS_DIR  where the environments go (default: <folder above the repo>/envs)
#            UV        path of the uv executable (default: uv on PATH, else ~/.local/bin/uv)
#            FORCE=1   re-create an environment that already exists (it is skipped otherwise)
#
# Requires uv (install with: pipx install uv). Does not need sudo.
# Used by: developers and CI; documented in docs/cuda-migration/execution-roadmap.md.
# Tested by: test/test_check_env.py::test_setup_envs_script_is_valid_bash.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # absolute path of the repository
ENVS_DIR="${ENVS_DIR:-$(dirname "$REPO")/envs}"
UV="${UV:-$(command -v uv || echo "$HOME/.local/bin/uv")}"
TORCH_CPU="https://download.pytorch.org/whl/cpu"          # index with the CPU-only PyTorch wheels

# new_venv <python-version> <dir>
#   Create an empty virtual environment. Returns 1 (and does nothing) when <dir> already
#   exists and FORCE is not 1, so callers can skip the installation step.
new_venv() {
  if [[ -d "$2" && "${FORCE:-0}" != 1 ]]; then
    echo "$2 already exists; use FORCE=1 to re-create it." >&2
    return 1
  fi
  "$UV" venv --clear --python "$1" "$2"
}

# pip_install <env-dir> <uv pip install arguments...>
#   Install packages into the given environment without activating it.
pip_install() {
  local env="$1"; shift
  VIRTUAL_ENV="$env" "$UV" pip install "$@"
}

# make_ref: the frozen reference environment.
make_ref() {
  local env="$ENVS_DIR/ref-0.1.10"
  new_venv 3.10 "$env" || return 0
  pip_install "$env" -r "$REPO/requirements/ref-0.1.10.txt" \
    --extra-index-url "$TORCH_CPU" --index-strategy unsafe-best-match
  # The published 0.1.10 without dependencies: every version comes from the pinned file.
  pip_install "$env" --no-deps "PyFWI==0.1.10"
}

# make_dev: the development environment, with this repository installed in editable mode.
make_dev() {
  local env="$ENVS_DIR/dev"
  new_venv 3.12 "$env" || return 0
  pip_install "$env" -r "$REPO/requirements/dev.txt" \
    --extra-index-url "$TORCH_CPU" --index-strategy unsafe-best-match
  pip_install "$env" --no-deps --no-build-isolation -e "$REPO"
}

# make_cuda12: dev with CuPy for CUDA 12 (requirements/cuda12.txt), the repository in editable mode.
make_cuda12() {
  local env="$ENVS_DIR/cuda12"
  new_venv 3.12 "$env" || return 0
  pip_install "$env" -r "$REPO/requirements/cuda12.txt"
  pip_install "$env" --no-deps --no-build-isolation -e "$REPO"
}

case "${1:-all}" in
  ref) make_ref ;;
  dev) make_dev ;;
  cuda12) make_cuda12 ;;
  all) make_ref; make_dev; make_cuda12 ;;
  *) echo "usage: $0 [ref|dev|cuda12|all]" >&2; exit 2 ;;
esac
echo "Environments in $ENVS_DIR. Check them with: \$ENV/bin/python $REPO/tools/check_env.py"

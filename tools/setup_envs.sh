#!/usr/bin/env bash
# Recria os ambientes do projeto a partir das versões travadas em requirements/.
#
#   ref-0.1.10  Python 3.10 + PyFWI 0.1.10 do PyPI, com as versões do requirements.txt
#               original (SciPy 1.7.3 ainda exporta MemoizeJac). Gera a referência da Fase 0.
#   dev         Python 3.12 + este repositório em modo editável + CuPy/CUDA 13 + PyOpenCL.
#
# Uso:   tools/setup_envs.sh [ref|dev|all]      (padrão: all)
# Vars:  ENVS_DIR (padrão: <pasta acima do repo>/envs), UV (caminho do uv), FORCE=1 recria.
# Requer: uv (pipx install uv). Não usa sudo.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENVS_DIR="${ENVS_DIR:-$(dirname "$REPO")/envs}"
UV="${UV:-$(command -v uv || echo "$HOME/.local/bin/uv")}"
TORCH_CPU="https://download.pytorch.org/whl/cpu"

new_venv() {  # new_venv <python> <dir>; devolve 1 se já existe e FORCE != 1
  if [[ -d "$2" && "${FORCE:-0}" != 1 ]]; then
    echo "$2 já existe; use FORCE=1 para recriar." >&2
    return 1
  fi
  "$UV" venv --clear --python "$1" "$2"
}

pip_install() {  # pip_install <env> <args...>
  local env="$1"; shift
  VIRTUAL_ENV="$env" "$UV" pip install "$@"
}

make_ref() {
  local env="$ENVS_DIR/ref-0.1.10"
  new_venv 3.10 "$env" || return 0
  pip_install "$env" -r "$REPO/requirements/ref-0.1.10.txt" \
    --extra-index-url "$TORCH_CPU" --index-strategy unsafe-best-match
  # A 0.1.10 publicada, sem dependências: as versões vêm todas do arquivo travado.
  pip_install "$env" --no-deps "PyFWI==0.1.10"
}

make_dev() {
  local env="$ENVS_DIR/dev"
  new_venv 3.12 "$env" || return 0
  pip_install "$env" -r "$REPO/requirements/dev.txt" \
    --extra-index-url "$TORCH_CPU" --index-strategy unsafe-best-match
  pip_install "$env" --no-deps --no-build-isolation -e "$REPO"
}

case "${1:-all}" in
  ref) make_ref ;;
  dev) make_dev ;;
  all) make_ref; make_dev ;;
  *) echo "uso: $0 [ref|dev|all]" >&2; exit 2 ;;
esac
echo "Ambientes em $ENVS_DIR. Verifique com: \$ENV/bin/python $REPO/tools/check_env.py"

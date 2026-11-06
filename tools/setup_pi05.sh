#!/usr/bin/env bash
# Prepare the π0.5 model environment: a pinned official OpenPI checkout with
# the Evo-DexBench overlay and its own uv-managed Python 3.11 environment.
set -euo pipefail

export PYTHONNOUSERSITE=1
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
example="$repo_root/examples/policies/pi05"
upstream_root="$repo_root/.third_party/openpi"
upstream_url="https://github.com/Physical-Intelligence/openpi.git"
pip_index_url=""
python_request="3.11"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --upstream-root) upstream_root="$2"; shift 2 ;;
    --upstream-url) upstream_url="$2"; shift 2 ;;
    --pip-index-url) pip_index_url="$2"; shift 2 ;;
    --python) python_request="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: tools/setup_pi05.sh [--upstream-root DIR] [--upstream-url URL] [--pip-index-url URL] [--python VERSION_OR_PATH]"
      exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

command -v git >/dev/null || { echo "git is required" >&2; exit 1; }
command -v uv >/dev/null || {
  echo "uv is required: https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 1
}
expected_commit="$(tail -n 1 "$example/pi05_policy/UPSTREAM_COMMIT" | tr -d '[:space:]')"

# The overlay adds four modules and edits three upstream files.
overlay_status=" M src/openpi/models/tokenizer.py
 M src/openpi/training/config.py
 M src/openpi/training/data_loader.py
?? src/openpi/policies/evodex_policy.py
?? src/openpi/training/evodex_config.py
?? src/openpi/training/evodex_lerobot.py
?? src/openpi/training/evodex_lerobot_cpu.py"

if [[ ! -d "$upstream_root/.git" ]]; then
  if [[ -e "$upstream_root" ]]; then
    echo "Refusing to replace an existing non-Git path: $upstream_root" >&2
    exit 1
  fi
  mkdir -p "$(dirname "$upstream_root")"
  # The model code needs no Git LFS objects or example submodules.
  GIT_LFS_SKIP_SMUDGE=1 git -c http.version=HTTP/1.1 clone --no-tags "$upstream_url" "$upstream_root"
fi
if [[ "$(git -C "$upstream_root" rev-parse HEAD)" != "$expected_commit" ]]; then
  if [[ -n "$(git -C "$upstream_root" status --porcelain)" ]]; then
    echo "Refusing to replace local changes: $upstream_root" >&2
    exit 1
  fi
  GIT_LFS_SKIP_SMUDGE=1 git -C "$upstream_root" fetch origin "$expected_commit"
  GIT_LFS_SKIP_SMUDGE=1 git -C "$upstream_root" checkout --detach "$expected_commit"
fi
status="$(git -C "$upstream_root" status --porcelain --untracked-files=all -- src)"
if [[ -z "$status" ]]; then
  python3 "$example/install_overlay.py" "$upstream_root"
elif [[ "$status" != "$overlay_status" ]]; then
  echo "The OpenPI checkout has changes other than the Evo-DexBench overlay: $upstream_root" >&2
  exit 1
fi
upstream_root="$(cd "$upstream_root" && pwd)"

index_args=()
if [[ -n "$pip_index_url" ]]; then index_args=(--index-url "$pip_index_url"); fi
cd "$upstream_root"
# Install the locked OpenPI environment, except its pre-v3 LeRobot revision.
GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen --python "$python_request" \
  --no-install-package lerobot "${index_args[@]}"
venv_python="$upstream_root/.venv/bin/python"
# Evo-DexBench datasets use LeRobot v3.0. Install only its dataset API, as in
# the other policy examples; video decoding uses PyAV.
uv pip install --python "$venv_python" --no-deps \
  lerobot==0.4.0 accelerate==1.10.1 pyserial==3.5 "${index_args[@]}"
# Keep exactly one OpenCV distribution (headless) in the server environment.
uv pip uninstall --python "$venv_python" opencv-python opencv-python-headless >/dev/null 2>&1 || true
uv pip install --python "$venv_python" --no-deps opencv-python-headless==4.11.0.86 "${index_args[@]}"

"$venv_python" "$example/validate_install.py" --openpi-root "$upstream_root"
echo "π0.5 is ready. Model environment interpreter: $venv_python"
echo "Use: export OPENPI_ROOT='$upstream_root' PI05_PYTHON='$venv_python'"

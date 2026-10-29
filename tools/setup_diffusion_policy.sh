#!/usr/bin/env bash
set -euo pipefail

export PYTHONNOUSERSITE=1
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
example="$repo_root/examples/policies/diffusion_policy"
name="evodex-dp"
upstream_root="$repo_root/.third_party/DiffusionPolicy-official"
upstream_url="https://github.com/real-stanford/diffusion_policy.git"
torch_index_url="https://download.pytorch.org/whl/cu124"
pip_index_url=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --name) name="$2"; shift 2 ;;
    --upstream-root) upstream_root="$2"; shift 2 ;;
    --upstream-url) upstream_url="$2"; shift 2 ;;
    --torch-index-url) torch_index_url="$2"; shift 2 ;;
    --pip-index-url) pip_index_url="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: tools/setup_diffusion_policy.sh [--name ENV] [--upstream-root DIR] [--upstream-url URL] [--torch-index-url URL] [--pip-index-url URL]"
      exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

# Conda may be a shell function in the user's interactive shell, not on PATH
# in this script. Activated environments expose the executable via CONDA_EXE.
if ! command -v conda >/dev/null && [[ -x "${CONDA_EXE:-}" ]]; then
  conda() { "$CONDA_EXE" "$@"; }
fi
command -v conda >/dev/null || { echo "conda is required" >&2; exit 1; }
command -v git >/dev/null || { echo "git is required" >&2; exit 1; }
expected_commit="$(tail -n 1 "$example/dp_policy/UPSTREAM_COMMIT" | tr -d '[:space:]')"
if [[ ! -d "$upstream_root/.git" ]]; then
  if [[ -e "$upstream_root" ]]; then
    echo "Refusing to replace an existing non-Git path: $upstream_root" >&2
    exit 1
  fi
  mkdir -p "$(dirname "$upstream_root")"
  git -c http.version=HTTP/1.1 clone --no-tags "$upstream_url" "$upstream_root"
fi
if [[ "$(git -C "$upstream_root" rev-parse HEAD)" != "$expected_commit" ]]; then
  if [[ -n "$(git -C "$upstream_root" status --porcelain)" ]]; then
    echo "Refusing to replace local changes: $upstream_root" >&2
    exit 1
  fi
  git -C "$upstream_root" fetch origin "$expected_commit"
  git -C "$upstream_root" checkout --detach "$expected_commit"
fi
if [[ -n "$(git -C "$upstream_root" status --porcelain)" ]]; then
  echo "The official Diffusion Policy checkout must be unmodified: $upstream_root" >&2
  exit 1
fi
upstream_root="$(cd "$upstream_root" && pwd)"

if ! conda env list | awk '{print $1}' | grep -Fxq "$name"; then
  conda create --name "$name" python=3.10 pip -y
fi
conda env config vars set --name "$name" PYTHONNOUSERSITE=1 DIFFUSION_POLICY_ROOT="$upstream_root"
pip_index_args=()
if [[ -n "$pip_index_url" ]]; then pip_index_args=(--index-url "$pip_index_url"); fi
conda run --no-capture-output --name "$name" python -m pip install \
  numpy==1.26.4 "${pip_index_args[@]}"
conda run --no-capture-output --name "$name" python -m pip install \
  torch==2.5.1 torchvision==0.20.1 --index-url "$torch_index_url"
conda run --no-capture-output --name "$name" python -m pip install \
  -r "$example/requirements.txt" "${pip_index_args[@]}"
# Use LeRobot's dataset API, not its robot/UI application dependency stack.
conda run --no-capture-output --name "$name" python -m pip install \
  --no-deps lerobot==0.4.0 "${pip_index_args[@]}"
conda run --no-capture-output --name "$name" python -m pip install \
  -e "$repo_root[policy-client,data]" -e "$example" "${pip_index_args[@]}"

env_prefix="$(conda run --name "$name" python -c 'import sys; print(sys.prefix)')"
mkdir -p "$env_prefix/etc/conda/activate.d" "$env_prefix/etc/conda/deactivate.d"
cp "$example/conda/activate.d/evodex_dp_cuda.sh" "$env_prefix/etc/conda/activate.d/"
cp "$example/conda/deactivate.d/evodex_dp_cuda.sh" "$env_prefix/etc/conda/deactivate.d/"
conda run --no-capture-output --name "$name" \
  python "$example/validate_install.py" --official-root "$upstream_root"
echo "DP-C and DP-T are ready in '$name'."
echo "Use: export DIFFUSION_POLICY_ROOT='$upstream_root'"

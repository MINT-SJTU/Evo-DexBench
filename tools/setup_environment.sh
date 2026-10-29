#!/usr/bin/env bash
set -euo pipefail

# Do not satisfy environment dependencies from ~/.local Python packages.
export PYTHONNOUSERSITE=1

profile="sim"
name="evodexbench"
with_kinedex_client="false"
with_dp_client="false"
torch_index_url="https://download.pytorch.org/whl/cu124"
pip_index_url=""
upstream_url=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile) profile="$2"; shift 2 ;;
    --name) name="$2"; shift 2 ;;
    --with-kinedex-client) with_kinedex_client="true"; shift ;;
    --with-dp-client) with_dp_client="true"; shift ;;
    --torch-index-url) torch_index_url="$2"; shift 2 ;;
    --pip-index-url) pip_index_url="$2"; shift 2 ;;
    --upstream-url) upstream_url="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: tools/setup_environment.sh [--profile sim|vr|policy-kinedex|policy-dp] [--name ENV] [--with-kinedex-client] [--with-dp-client] [--torch-index-url URL] [--pip-index-url URL] [--upstream-url URL]"
      exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

case "$profile" in
  sim) extras="maniskill,data,assets" ;;
  vr) extras="maniskill,data,assets,vr" ;;
  policy-kinedex|policy-dp)
    if [[ "$with_kinedex_client" == "true" || "$with_dp_client" == "true" ]]; then
      echo "Client flags belong to the sim or vr profile." >&2
      exit 2
    fi
    repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    model_args=(
      --name "$name"
      --torch-index-url "$torch_index_url"
    )
    if [[ -n "$pip_index_url" ]]; then
      model_args+=(--pip-index-url "$pip_index_url")
    fi
    if [[ -n "$upstream_url" ]]; then
      model_args+=(--upstream-url "$upstream_url")
    fi
    if [[ "$profile" == "policy-dp" ]]; then
      exec "$repo_root/tools/setup_diffusion_policy.sh" "${model_args[@]}"
    fi
    exec "$repo_root/tools/setup_kinedex.sh" "${model_args[@]}" ;;
  *) echo "Unknown profile: $profile" >&2; exit 2 ;;
esac

command -v conda >/dev/null || { echo "conda is required" >&2; exit 1; }
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if ! conda env list | awk '{print $1}' | grep -Fxq "$name"; then
  conda create --name "$name" python=3.10 pip -y
fi
conda env config vars set --name "$name" PYTHONNOUSERSITE=1
conda run --no-capture-output --name "$name" python -m pip install \
  torch==2.5.1 --index-url "$torch_index_url"

pip_index_args=()
if [[ -n "$pip_index_url" ]]; then
  pip_index_args=(--index-url "$pip_index_url")
fi
conda run --no-capture-output --name "$name" python -m pip install \
  -e "$repo_root[$extras]" "${pip_index_args[@]}"
if [[ "$with_kinedex_client" == "true" ]]; then
  conda run --no-capture-output --name "$name" python -m pip install \
    -e "$repo_root/examples/policies/kinedex" "${pip_index_args[@]}"
fi
if [[ "$with_dp_client" == "true" ]]; then
  conda run --no-capture-output --name "$name" python -m pip install \
    -e "$repo_root/examples/policies/diffusion_policy" "${pip_index_args[@]}"
fi

conda run --no-capture-output --name "$name" python -m pip check
echo "Environment '$name' is ready for profile '$profile'."

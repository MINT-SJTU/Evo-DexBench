#!/usr/bin/env bash
set -euo pipefail

export PYTHONNOUSERSITE=1

name="evodex-kinedex"
torch_index_url="https://download.pytorch.org/whl/cu124"
pip_index_url=""
lerobot_version="0.4.0"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
upstream_root="$repo_root/.third_party/KineDex-official"
upstream_url="https://github.com/DinoMini00/KineDex_code.git"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --name) name="$2"; shift 2 ;;
    --upstream-root) upstream_root="$2"; shift 2 ;;
    --upstream-url) upstream_url="$2"; shift 2 ;;
    --torch-index-url) torch_index_url="$2"; shift 2 ;;
    --pip-index-url) pip_index_url="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: tools/setup_kinedex.sh [--name ENV] [--upstream-root DIR] [--upstream-url URL] [--torch-index-url URL] [--pip-index-url URL]"
      exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

command -v conda >/dev/null || { echo "conda is required" >&2; exit 1; }
command -v git >/dev/null || { echo "git is required" >&2; exit 1; }
expected_commit="$(tr -d '[:space:]' < "$repo_root/examples/policies/kinedex/kinedex_policy/UPSTREAM_COMMIT")"

if [[ ! -d "$upstream_root/.git" ]]; then
  upstream_parent="$(dirname "$upstream_root")"
  mkdir -p "$upstream_parent"
  if [[ -e "$upstream_root" ]]; then
    if [[ -d "$upstream_root" ]] && [[ -z "$(find "$upstream_root" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
      rmdir "$upstream_root"
    else
      echo "Refusing to replace a non-Git KineDex path: $upstream_root" >&2
      exit 1
    fi
  fi

  clone_staging="$(mktemp -d "$upstream_parent/.kinedex-clone.XXXXXX")"
  cleanup_clone_staging() { rm -rf -- "$clone_staging"; }
  trap cleanup_clone_staging EXIT
  clone_checkout="$clone_staging/checkout"
  if ! git clone --no-tags "$upstream_url" "$clone_checkout"; then
    echo "Initial KineDex clone failed; retrying once with Git HTTP/1.1." >&2
    clone_checkout="$clone_staging/checkout-http1"
    if ! git -c http.version=HTTP/1.1 clone --no-tags "$upstream_url" "$clone_checkout"; then
      echo "Unable to clone KineDex from $upstream_url" >&2
      exit 1
    fi
  fi
  mv "$clone_checkout" "$upstream_root"
  cleanup_clone_staging
  trap - EXIT
fi

current_commit="$(git -C "$upstream_root" rev-parse HEAD)"
if [[ "$current_commit" != "$expected_commit" ]]; then
  if [[ -n "$(git -C "$upstream_root" status --porcelain)" ]]; then
    echo "KineDex checkout has local changes; refusing to replace them: $upstream_root" >&2
    exit 1
  fi
  if ! git -C "$upstream_root" fetch origin "$expected_commit"; then
    echo "Initial KineDex fetch failed; retrying once with Git HTTP/1.1." >&2
    git -c http.version=HTTP/1.1 -C "$upstream_root" \
      fetch origin "$expected_commit"
  fi
  git -C "$upstream_root" checkout --detach "$expected_commit"
fi
python3 "$repo_root/examples/policies/kinedex/prepare_upstream.py" "$upstream_root"

if ! conda env list | awk '{print $1}' | grep -Fxq "$name"; then
  conda create --name "$name" python=3.10 pip -y
fi
conda env config vars set --name "$name" PYTHONNOUSERSITE=1

pip_index_args=()
if [[ -n "$pip_index_url" ]]; then
  pip_index_args=(--index-url "$pip_index_url")
fi

# Install NumPy first so the CUDA wheel transaction cannot temporarily select
# NumPy 2. LeRobot's optional visualization stack requires NumPy 2, but the
# validated KineDex/robomimic stack intentionally remains on NumPy 1.26.
conda run --no-capture-output --name "$name" python -m pip install \
  numpy==1.26.4 "${pip_index_args[@]}"
if ! conda run --no-capture-output --name "$name" python -m pip install \
  torch==2.5.1 torchvision==0.20.1 --index-url "$torch_index_url"; then
  echo "PyTorch wheel index failed; retrying the pinned install once." >&2
  conda run --no-capture-output --name "$name" python -m pip install \
    torch==2.5.1 torchvision==0.20.1 --index-url "$torch_index_url"
fi

conda run --no-capture-output --name "$name" python -m pip install \
  -r "$repo_root/examples/policies/kinedex/requirements.txt" "${pip_index_args[@]}"
# LeRobot 0.4 declares dependencies for every supported robot, UI, decoder,
# and visualization backend. Installing it without those optional dependency
# groups preserves the validated Torch pair; validate_install.py imports the
# exact dataset API used by training so missing runtime dependencies fail here.
conda run --no-capture-output --name "$name" python -m pip install \
  --no-deps "lerobot==$lerobot_version" "${pip_index_args[@]}"
conda run --no-capture-output --name "$name" python -m pip install \
  -e "$repo_root[policy-client,data]" "${pip_index_args[@]}"
conda run --no-capture-output --name "$name" python -m pip install \
  -e "$repo_root/examples/policies/kinedex" "${pip_index_args[@]}"
conda run --no-capture-output --name "$name" env KINEDEX_ROOT="$upstream_root" \
  python "$repo_root/examples/policies/kinedex/validate_install.py" \
  --upstream-root "$upstream_root"

echo "KineDex is ready in conda environment '$name'."
echo "Use: export KINEDEX_ROOT='$upstream_root'"

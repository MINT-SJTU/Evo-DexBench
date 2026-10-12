#!/usr/bin/env bash
set -euo pipefail

install_packages="true"
icd_path="/etc/vulkan/icd.d/evodex_nvidia_egl_icd.json"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-install) install_packages="false"; shift ;;
    --icd-path) icd_path="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: sudo tools/setup_headless_vulkan.sh [--no-install] [--icd-path PATH]"
      exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

if [[ "$EUID" -ne 0 ]]; then
  echo "Run this host setup as root (for example, with sudo)." >&2
  exit 1
fi

if [[ "$install_packages" == "true" ]]; then
  command -v apt-get >/dev/null || {
    echo "Automatic package installation currently supports Debian/Ubuntu hosts only." >&2
    exit 1
  }
  apt-get update
  apt-get install -y libvulkan1 vulkan-tools libegl1 libsm6
fi

egl_library="$(ldconfig -p | awk '/libEGL_nvidia\.so\.0/{print $NF; exit}')"
if [[ -z "$egl_library" || ! -e "$egl_library" ]]; then
  echo "NVIDIA EGL library libEGL_nvidia.so.0 is unavailable." >&2
  echo "The host/container must expose the NVIDIA graphics driver capability." >&2
  exit 1
fi

install -d "$(dirname "$icd_path")"
printf '%s\n' \
  '{' \
  '  "file_format_version": "1.0.0",' \
  '  "ICD": {' \
  "    \"library_path\": \"$egl_library\"," \
  '    "api_version": "1.3.277"' \
  '  }' \
  '}' >"$icd_path"

summary="$(VK_ICD_FILENAMES="$icd_path" vulkaninfo --summary 2>&1)" || {
  printf '%s\n' "$summary" >&2
  exit 1
}
printf '%s\n' "$summary"
if ! grep -q 'deviceName.*NVIDIA' <<<"$summary"; then
  echo "Vulkan did not enumerate an NVIDIA device." >&2
  exit 1
fi

echo
echo "Headless NVIDIA Vulkan is ready. Run this in every new shell:"
printf 'export VK_ICD_FILENAMES=%q\n' "$icd_path"

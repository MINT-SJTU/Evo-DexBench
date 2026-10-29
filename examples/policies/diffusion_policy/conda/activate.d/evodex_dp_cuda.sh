#!/usr/bin/env sh
# PyPI CUDA wheels bundle nvJitLink. Prefer that copy over a host CUDA toolkit
# that may be older; this change is scoped to the activated conda environment.
export EVODEX_DP_OLD_LD_LIBRARY_PATH="${LD_LIBRARY_PATH-}"
for evodex_nvjitlink_dir in \
    "${CONDA_PREFIX}"/lib/python*/site-packages/nvidia/nvjitlink/lib
do
    if [ -d "${evodex_nvjitlink_dir}" ]; then
        export LD_LIBRARY_PATH="${evodex_nvjitlink_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
        break
    fi
done
unset evodex_nvjitlink_dir

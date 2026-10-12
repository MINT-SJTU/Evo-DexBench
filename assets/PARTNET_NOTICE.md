# PartNet-Mobility assets are not distributed

Original PartNet-Mobility data and locally generated collision/URDF overlays
are excluded from the public asset bundle. Recipients must accept the upstream
terms and use their own authorized Hugging Face account.

- Terms: https://sapien.ucsd.edu/about
- Official gated dataset: https://huggingface.co/datasets/sapien-sim/PartNetMobility

`dexbench assets download` retrieves only the 49 model archives used by the
benchmark from that official dataset, then generates the URDF overlays
locally. No PartNet file is uploaded to or served from the Evo-DexBench asset
repository.

The workbench overlay for model `25144` preserves the historical replacement of
the missing `textured_objs/new-24.obj` reference with the existing
`textured_objs/original-24.obj` in both visual and collision elements. Original
downloads are not modified; overlay generation verifies its referenced meshes.

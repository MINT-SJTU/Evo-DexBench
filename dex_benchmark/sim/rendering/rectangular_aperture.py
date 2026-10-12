from __future__ import annotations

from pathlib import Path
import shutil
import tempfile
import threading
from typing import Any

import numpy as np

from dex_benchmark.sim.scenes.layouts.spec import RectangularLightApertureSpec


_SHADER_BUILD_LOCK = threading.Lock()
_SHADER_CONFIG_CACHE: dict[
    RectangularLightApertureSpec,
    tuple[Any, Any],
] = {}
_SHADER_TEMP_DIRS: list[tempfile.TemporaryDirectory[str]] = []


def _glsl_float(value: float) -> str:
    formatted_value = f"{value:.9g}"
    if "." not in formatted_value and "e" not in formatted_value.lower():
        formatted_value += ".0"
    return formatted_value


def _glsl_vec3(value_xyz: tuple[float, float, float]) -> str:
    return "vec3(" + ", ".join(_glsl_float(value) for value in value_xyz) + ")"


def _aperture_glsl(aperture: RectangularLightApertureSpec) -> str:
    axis_u = np.asarray(aperture.axis_u_xyz, dtype=np.float64)
    axis_v = np.asarray(aperture.axis_v_xyz, dtype=np.float64)
    normal = np.cross(axis_u, axis_v)
    normal /= np.linalg.norm(normal)
    normal_xyz = tuple(float(component) for component in normal)
    return f"""
float rectangularLightAperture(vec3 worldPosition, vec3 lightDirection) {{
  const vec3 apertureCenter = {_glsl_vec3(aperture.center_xyz)};
  const vec3 apertureAxisU = {_glsl_vec3(aperture.axis_u_xyz)};
  const vec3 apertureAxisV = {_glsl_vec3(aperture.axis_v_xyz)};
  const vec3 apertureNormal = {_glsl_vec3(normal_xyz)};
  const vec2 apertureHalfSize = vec2(
      {_glsl_float(aperture.half_size_uv[0])},
      {_glsl_float(aperture.half_size_uv[1])});
  const float edgeSoftness = {_glsl_float(aperture.edge_softness_m)};

  float directionDotNormal = dot(lightDirection, apertureNormal);
  if (abs(directionDotNormal) < 1e-6) {{
    return 0.0;
  }}
  // Trace the receiver backward, opposite the light-travel direction, until
  // it intersects the finite window plane.
  float reverseDistance = dot(
      worldPosition - apertureCenter, apertureNormal) / directionDotNormal;
  if (reverseDistance < 0.0) {{
    return 0.0;
  }}
  vec3 planePoint = worldPosition - reverseDistance * lightDirection;
  vec3 apertureDelta = planePoint - apertureCenter;
  vec2 apertureCoord = vec2(
      dot(apertureDelta, apertureAxisU),
      dot(apertureDelta, apertureAxisV));
  vec2 edgeStart = apertureHalfSize - vec2(edgeSoftness);
  vec2 apertureMask = vec2(1.0) - smoothstep(
      edgeStart, apertureHalfSize, abs(apertureCoord));
  return apertureMask.x * apertureMask.y;
}}
"""


def _replace_once(source: str, old: str, new: str, *, label: str) -> str:
    occurrence_count = source.count(old)
    if occurrence_count != 1:
        raise RuntimeError(
            f"Cannot patch SAPIEN shader section {label!r}: expected exactly one "
            f"matching source block, found {occurrence_count}. The installed "
            "SAPIEN shader layout may be incompatible with this benchmark build."
        )
    return source.replace(old, new, 1)


def _patch_raster_lighting_shader(
    source: str,
    aperture: RectangularLightApertureSpec,
) -> str:
    world_to_camera_function = """vec4 world2camera(vec4 pos) {
  return cameraBuffer.viewMatrix * pos;
}
"""
    source = _replace_once(
        source,
        world_to_camera_function,
        world_to_camera_function + _aperture_glsl(aperture),
        label="rectangular aperture helper",
    )

    light_direction_block = """    vec3 lightDir = mat3(cameraBuffer.viewMatrix) * sceneBuffer.directionalLights[i].direction.xyz;

    vec4 ssPosition"""
    source = _replace_once(
        source,
        light_direction_block,
        """    vec3 worldLightDirection = sceneBuffer.directionalLights[i].direction.xyz;
    vec3 lightDir = mat3(cameraBuffer.viewMatrix) * worldLightDirection;
    vec3 worldPosition = vec3(
        cameraBuffer.viewMatrixInverse * vec4(csPosition.xyz, 1));

    vec4 ssPosition""",
        label="directional-light world position",
    )

    shadow_coordinate_block = """    shadowMapCoord.xy = shadowMapCoord.xy * 0.5 + 0.5;

    float resolution = textureSize(samplerDirectionalLightDepths[i], 0).x;"""
    source = _replace_once(
        source,
        shadow_coordinate_block,
        """    shadowMapCoord.xy = shadowMapCoord.xy * 0.5 + 0.5;
    // SAPIEN's stock raster shader has no receiver bias. A small slope-scaled
    // bias prevents planar surfaces from shadowing themselves at grazing angles.
    float receiverCosine = max(dot(normal, -lightDir), 0.0);
    shadowMapCoord.z -= max(0.00005, 0.0005 * (1.0 - receiverCosine));

    float resolution = textureSize(samplerDirectionalLightDepths[i], 0).x;""",
        label="directional-light receiver bias",
    )

    shadow_visibility_block = """    float visibility = ShadowMapPCF(
        samplerDirectionalLightDepths[i], shadowMapCoord.xyz, resolution, 1 / resolution, 1);

    color += visibility * computeDirectionalLight("""
    source = _replace_once(
        source,
        shadow_visibility_block,
        """    float visibility = ShadowMapPCF(
        samplerDirectionalLightDepths[i], shadowMapCoord.xyz, resolution, 1 / resolution, 1);
    visibility *= rectangularLightAperture(worldPosition, worldLightDirection);

    color += visibility * computeDirectionalLight(""",
        label="directional-light aperture visibility",
    )

    spot_shadow_coordinate_block = """    shadowMapCoord.xy = shadowMapCoord.xy * 0.5 + 0.5;

    float resolution = textureSize(samplerSpotLightDepths[i], 0).x;"""
    source = _replace_once(
        source,
        spot_shadow_coordinate_block,
        """    shadowMapCoord.xy = shadowMapCoord.xy * 0.5 + 0.5;
    // Wide spotlights expose the stock shader's lack of receiver bias on large
    // planar surfaces. Offset the receiver conservatively in shadow-map depth;
    // the slope term grows only when the light grazes the surface.
    vec3 spotDirectionToSource = normalize(l);
    float spotReceiverCosine = max(dot(normal, spotDirectionToSource), 0.0);
    shadowMapCoord.z -= max(0.00005, 0.0005 * (1.0 - spotReceiverCosine));

    float resolution = textureSize(samplerSpotLightDepths[i], 0).x;""",
        label="spot-light receiver bias",
    )
    return source


def _copy_and_patch_shader_pack(
    aperture: RectangularLightApertureSpec,
) -> tuple[Any, Any]:
    try:
        import sapien
        from mani_skill.render import (
            PREBUILT_SHADER_CONFIGS,
            SAPIEN_RENDER_SYSTEM,
            ShaderConfig,
        )
    except Exception as exc:  # pragma: no cover - optional simulator dependency.
        raise ImportError(
            "ManiSkill and SAPIEN are required to build the rectangular-aperture "
            "camera shader."
        ) from exc

    if SAPIEN_RENDER_SYSTEM != "3.0":
        raise RuntimeError(
            "The rectangular-aperture shader currently requires SAPIEN render "
            f"system 3.0, got {SAPIEN_RENDER_SYSTEM!r}."
        )
    source_shader_root = Path(sapien.__file__).resolve().parent / "vulkan_shader"
    required_source_dirs = [
        source_shader_root / "common",
        source_shader_root / "minimal",
        source_shader_root / "default",
    ]
    missing_source_dirs = [path for path in required_source_dirs if not path.is_dir()]
    if missing_source_dirs:
        raise FileNotFoundError(
            "Installed SAPIEN shader directories are incomplete: "
            + ", ".join(str(path) for path in missing_source_dirs)
        )

    temporary_shader_dir = tempfile.TemporaryDirectory(
        prefix="evodex_rectangular_aperture_"
    )
    _SHADER_TEMP_DIRS.append(temporary_shader_dir)
    target_shader_root = Path(temporary_shader_dir.name)
    for directory_name in ("common", "minimal", "default"):
        shutil.copytree(
            source_shader_root / directory_name,
            target_shader_root / directory_name,
        )

    shader_files = [
        target_shader_root / "minimal" / "gbuffer.frag",
        target_shader_root / "default" / "deferred.frag",
    ]
    for shader_file in shader_files:
        source = shader_file.read_text(encoding="utf-8")
        patched_source = _patch_raster_lighting_shader(source, aperture)
        shader_file.write_text(patched_source, encoding="utf-8")

    minimal_pack_path = str((target_shader_root / "minimal").resolve())
    default_pack_path = str((target_shader_root / "default").resolve())
    minimal_base_config = PREBUILT_SHADER_CONFIGS["minimal"]
    default_base_config = PREBUILT_SHADER_CONFIGS["default"]
    minimal_config = ShaderConfig(
        shader_pack=minimal_pack_path,
        texture_names=dict(minimal_base_config.texture_names),
        shader_pack_config=dict(minimal_base_config.shader_pack_config),
        texture_transforms=dict(minimal_base_config.texture_transforms),
    )
    default_config = ShaderConfig(
        shader_pack=default_pack_path,
        texture_names=dict(default_base_config.texture_names),
        shader_pack_config=dict(default_base_config.shader_pack_config),
        texture_transforms=dict(default_base_config.texture_transforms),
    )
    PREBUILT_SHADER_CONFIGS[minimal_pack_path] = minimal_config
    PREBUILT_SHADER_CONFIGS[default_pack_path] = default_config
    return minimal_config, default_config


def _shader_configs(
    aperture: RectangularLightApertureSpec,
) -> tuple[Any, Any]:
    with _SHADER_BUILD_LOCK:
        cached_configs = _SHADER_CONFIG_CACHE.get(aperture)
        if cached_configs is None:
            cached_configs = _copy_and_patch_shader_pack(aperture)
            _SHADER_CONFIG_CACHE[aperture] = cached_configs
        return cached_configs


def rectangular_aperture_camera_shader_config(
    aperture: RectangularLightApertureSpec,
) -> Any:
    """Return the patched minimal shader used by RGB/depth sensor cameras."""

    try:
        import sapien
    except Exception as exc:  # pragma: no cover - optional simulator dependency.
        raise ImportError("SAPIEN is required to configure camera render formats.") from exc
    sapien.render.set_picture_format("Color", "r8g8b8a8unorm")
    sapien.render.set_picture_format("ColorRaw", "r8g8b8a8unorm")
    sapien.render.set_picture_format(
        "PositionSegmentation",
        "r16g16b16a16sint",
    )
    return _shader_configs(aperture)[0]


def rectangular_aperture_viewer_shader_config(
    aperture: RectangularLightApertureSpec,
) -> Any:
    """Return the patched default PBR shader used by the interactive viewer."""

    return _shader_configs(aperture)[1]

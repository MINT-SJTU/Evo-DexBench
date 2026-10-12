"""Generate benchmark URDF overlays from an official PartNet download.

The public repository does not redistribute PartNet geometry.  These routines
derive the small collision and mesh-reference URDF overlays required by the
benchmark after the user has downloaded the corresponding official models under
the PartNet terms. Visual meshes remain in the official dataset directory.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import xml.etree.ElementTree as ET


_OUTPUT_NAME = "mobility_collision_override.urdf"


def _link(root: ET.Element, name: str) -> ET.Element:
    result = root.find(f"./link[@name='{name}']")
    if result is None:
        raise ValueError(f"PartNet URDF is missing link {name!r}.")
    return result


def _rewrite_mesh_paths(root: ET.Element, model_id: str) -> None:
    prefix = f"../../partnet_mobility/{model_id}/"
    for mesh in root.findall(".//mesh"):
        filename = mesh.get("filename")
        if filename is None:
            raise ValueError("PartNet mesh element is missing filename.")
        mesh.set("filename", prefix + filename.lstrip("./"))


def _clear_collisions(link: ET.Element) -> None:
    for collision in tuple(link.findall("collision")):
        link.remove(collision)


def _collision(
    link: ET.Element,
    *,
    name: str | None,
    kind: str,
    value: str,
    xyz: str,
    rpy: str | None = None,
) -> ET.Element:
    collision = ET.SubElement(link, "collision")
    if name is not None:
        collision.set("name", name)
    origin = ET.SubElement(collision, "origin", {"xyz": xyz})
    if rpy is not None:
        origin.set("rpy", rpy)
    geometry = ET.SubElement(collision, "geometry")
    attribute = {"mesh": "filename", "box": "size", "sphere": "radius"}[kind]
    ET.SubElement(geometry, kind, {attribute: value})
    return collision


def _replace_with_boxes(
    link: ET.Element,
    specs: tuple[tuple[str, str, str, str], ...],
) -> None:
    _clear_collisions(link)
    for name, size, xyz, rpy in specs:
        _collision(
            link,
            name=name,
            kind="box",
            value=size,
            xyz=xyz,
            rpy=rpy,
        )


def _override_10143(root: ET.Element) -> None:
    _replace_with_boxes(
        _link(root, "link_1"),
        (
            ("override_left_wall", "0.060000 1.525921 0.848270", "-0.408293 -0.042770 -0.041494", "0 0 0"),
            ("override_right_wall", "0.060000 1.525921 0.848270", "0.390557 -0.042770 -0.041494", "0 0 0"),
            ("override_bottom_wall", "0.738850 0.060000 0.848270", "-0.008868 -0.775731 -0.041494", "0 0 0"),
            ("override_top_wall", "0.738850 0.060000 0.848270", "-0.008868 0.690190 -0.041494", "0 0 0"),
            ("override_back_wall", "0.738850 1.405921 0.060000", "-0.008868 -0.042770 -0.435629", "0 0 0"),
        ),
    )


def _override_9992(root: ET.Element) -> None:
    _replace_with_boxes(
        _link(root, "link_0"),
        (("keyboard_base_simplified", "1.285505 0.067783 0.923993", "0.0054765 -0.1814015 0.2450985", "0 0 0"),),
    )
    _replace_with_boxes(
        _link(root, "link_1"),
        (("screen_simplified", "0.0284906044 0.9080502103 1.285505", "-0.6427525 0.4112979057 -0.1669850781", "1.5707963268 1.2636184574 -1.5707963268"),),
    )


def _ring_collisions(
    link: ET.Element,
    *,
    prefix: str,
    mesh_specs: tuple[tuple[str, str, str], ...],
    centers: tuple[str, ...],
) -> None:
    _clear_collisions(link)
    for name, filename, xyz in mesh_specs:
        _collision(link, name=name, kind="mesh", value=filename, xyz=xyz)
    for index, xyz in enumerate(centers):
        _collision(
            link,
            name=f"{prefix}_ring_{index:02d}",
            kind="sphere",
            value="0.038",
            xyz=xyz,
        )


def _override_10449(root: ET.Element) -> None:
    _ring_collisions(
        _link(root, "link_0"),
        prefix="moving",
        mesh_specs=(
            ("moving_blade", "../../partnet_mobility/10449/textured_objs/new-5.obj", "-0.006112207895012212 -0.015478999999999798 -0.010956438587861106"),
            ("moving_connector", "../../partnet_mobility/10449/textured_objs/new-4.obj", "-0.006112207895012212 -0.015478999999999798 -0.010956438587861106"),
        ),
        centers=(
            "0.824 -0.023 -0.341", "0.8066 -0.023 -0.276",
            "0.759 -0.023 -0.2284", "0.694 -0.023 -0.211",
            "0.629 -0.023 -0.2284", "0.5814 -0.023 -0.276",
            "0.564 -0.023 -0.341", "0.5814 -0.023 -0.406",
            "0.629 -0.023 -0.4536", "0.694 -0.023 -0.471",
            "0.759 -0.023 -0.4536", "0.8066 -0.023 -0.406",
        ),
    )
    _ring_collisions(
        _link(root, "link_1"),
        prefix="support",
        mesh_specs=(
            ("support_blade", "../../partnet_mobility/10449/textured_objs/new-6.obj", "0 0 0"),
            ("support_connector", "../../partnet_mobility/10449/textured_objs/new-7.obj", "0 0 0"),
        ),
        centers=(
            "0.710 0.007 0.370", "0.6926 0.007 0.435",
            "0.645 0.007 0.4826", "0.580 0.007 0.500",
            "0.515 0.007 0.4826", "0.4674 0.007 0.435",
            "0.450 0.007 0.370", "0.4674 0.007 0.305",
            "0.515 0.007 0.2574", "0.580 0.007 0.240",
            "0.645 0.007 0.2574", "0.6926 0.007 0.305",
        ),
    )


def _override_100189(root: ET.Element) -> None:
    material = ET.Element("material", {"name": "b14_handle_material"})
    ET.SubElement(material, "color", {"rgba": "0.20 0.22 0.24 1.0"})
    root.insert(0, material)
    link = _link(root, "link_0")
    visual = ET.SubElement(link, "visual", {"name": "b14_lid_front_handle"})
    ET.SubElement(
        visual,
        "origin",
        {
            "xyz": "-0.71462027 0.75739866 0.72585260",
            "rpy": "-3.14159265 0.87790045 -1.57079633",
        },
    )
    geometry = ET.SubElement(visual, "geometry")
    ET.SubElement(geometry, "box", {"size": "0.11363636 0.54545455 0.02727273"})
    ET.SubElement(visual, "material", {"name": "b14_handle_material"})
    _collision(
        link,
        name="b14_lid_front_handle",
        kind="box",
        value="0.11363636 0.54545455 0.02727273",
        xyz="-0.71462027 0.75739866 0.72585260",
        rpy="-3.14159265 0.87790045 -1.57079633",
    )


def _override_100664(root: ET.Element) -> None:
    _replace_with_boxes(
        _link(root, "link_1"),
        (
            ("box_bottom", "1.160 0.034 0.702", "0 -0.465 0.099", "0 0 0"),
            ("box_wall_x_negative", "0.032 0.638 0.702", "-0.535 -0.129 0.099", "0 0 0"),
            ("box_wall_x_positive", "0.032 0.638 0.702", "0.593 -0.129 0.099", "0 0 0"),
            ("box_wall_z_negative", "1.096 0.638 0.040", "0 -0.129 -0.232", "0 0 0"),
            ("box_wall_z_positive", "1.096 0.638 0.032", "0 -0.129 0.434", "0 0 0"),
        ),
    )


def _override_7167(root: ET.Element) -> None:
    link = _link(root, "link_3")
    retained: list[ET.Element] = []
    keep = {"new-3.obj", "new-2.obj", "new-1.obj", "new-0.obj"}
    for collision in link.findall("collision"):
        mesh = collision.find("./geometry/mesh")
        if mesh is not None and Path(mesh.get("filename", "")).name in keep:
            retained.append(deepcopy(collision))
    _clear_collisions(link)
    link.extend(retained)
    for name, size, xyz in (
        ("m12_control_panel", "0.338427 0.879911 0.077838", "0.6260775 -0.0046405 0.476788"),
        ("m12_body_back", "1.556766 0.879911 0.074454", "0.016908 -0.0046405 -0.530034"),
        ("m12_body_side_negative", "0.490719 0.879911 1.00513", "0.5499315 -0.0046405 -0.064696"),
        ("m12_body_side_positive", "0.101528 0.879911 1.00513", "-0.710711 -0.0046405 -0.064696"),
        ("m12_body_floor", "1.556766 0.101528 1.00513", "0.016908 -0.393832 -0.064696"),
        ("m12_body_ceiling", "0.964519 0.101528 1.00513", "-0.1776875 0.384551 -0.064696"),
    ):
        _collision(
            link,
            name=name,
            kind="box",
            value=size,
            xyz=xyz,
            rpy="0 0 0",
        )


def _override_101408(root: ET.Element) -> None:
    """Expose a slightly larger wheel without redistributing the mouse URDF."""

    link = _link(root, "link_2")
    for element_name in ("visual", "collision"):
        element = link.find(element_name)
        if element is None:
            raise ValueError(f"PartNet mouse has no {element_name} wheel element.")
        origin = element.find("origin")
        mesh = element.find("./geometry/mesh")
        if origin is None or mesh is None:
            raise ValueError("PartNet mouse wheel structure changed upstream.")
        origin.set("xyz", "0 -0.06 -0.6072")
        mesh.set("scale", "1.5 1.2 1.2")


def _override_25144(root: ET.Element) -> None:
    """Preserve the historical workbench fix for an absent official mesh."""
    for mesh in root.iter("mesh"):
        filename = mesh.get("filename", "")
        if filename.endswith("/textured_objs/new-24.obj"):
            mesh.set("filename", filename.removesuffix("new-24.obj") + "original-24.obj")


_URDF_OVERRIDES = {
    "25144": _override_25144,
    "10143": _override_10143,
    "9992": _override_9992,
    "10449": _override_10449,
    "100189": _override_100189,
    "100664": _override_100664,
    "7167": _override_7167,
    "101408": _override_101408,
}


def _adjust_obj_index(token: str, offsets: tuple[int, int, int]) -> str:
    parts = token.split("/")
    adjusted: list[str] = []
    for index, part in enumerate(parts):
        if not part:
            adjusted.append(part)
            continue
        value = int(part)
        if value <= 0:
            raise ValueError("PartNet override generation requires positive OBJ indices.")
        adjusted.append(str(value + offsets[index]))
    return "/".join(adjusted)


def _merge_obj_files(inputs: list[Path], output: Path) -> None:
    vertices: list[str] = []
    texture_coordinates: list[str] = []
    normals: list[str] = []
    faces: list[str] = []
    offsets = (0, 0, 0)
    for source in inputs:
        source_vertices: list[str] = []
        source_texture_coordinates: list[str] = []
        source_normals: list[str] = []
        source_faces: list[str] = []
        for raw_line in source.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if line.startswith("v "):
                source_vertices.append(line)
            elif line.startswith("vt "):
                source_texture_coordinates.append(line)
            elif line.startswith("vn "):
                source_normals.append(line)
            elif line.startswith("f "):
                source_faces.append(
                    "f " + " ".join(
                        _adjust_obj_index(token, offsets)
                        for token in line.split()[1:]
                    )
                )
        vertices.extend(source_vertices)
        texture_coordinates.extend(source_texture_coordinates)
        normals.extend(source_normals)
        faces.extend(source_faces)
        offsets = (
            offsets[0] + len(source_vertices),
            offsets[1] + len(source_texture_coordinates),
            offsets[2] + len(source_normals),
        )
    output.write_text(
        "\n".join(
            ["# generated from the user's official PartNet-Mobility download", *vertices,
             *texture_coordinates, *normals, *faces, ""]
        ),
        encoding="utf-8",
    )


def generate_partnet_overrides(
    official_root: str | Path,
    output_root: str | Path,
) -> tuple[Path, ...]:
    """Generate every URDF overlay required by Evo-DexBench."""

    official_root = Path(official_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    generated: list[Path] = []
    for model_id, transform in _URDF_OVERRIDES.items():
        source = official_root / model_id / "mobility.urdf"
        if not source.is_file():
            raise FileNotFoundError(f"Missing official PartNet model {model_id}: {source}")
        tree = ET.parse(source)
        root = tree.getroot()
        _rewrite_mesh_paths(root, model_id)
        transform(root)
        destination = output_root / model_id / _OUTPUT_NAME
        for mesh in root.iter("mesh"):
            mesh_path = (destination.parent / mesh.attrib["filename"]).resolve()
            if not mesh_path.is_file():
                raise FileNotFoundError(
                    f"PartNet model {model_id} overlay references missing mesh: {mesh_path}"
                )
        ET.indent(tree, space="  ")
        destination.parent.mkdir(parents=True, exist_ok=True)
        tree.write(destination, encoding="utf-8", xml_declaration=True)
        generated.append(destination)

    pot_source = official_root / "100038" / "textured_objs"
    pot_output = output_root / "100038" / "pot_lid.obj"
    pot_output.parent.mkdir(parents=True, exist_ok=True)
    _merge_obj_files(
        [pot_source / f"original-{index}.obj" for index in (1, 3, 4, 5, 6, 7)],
        pot_output,
    )
    generated.append(pot_output)
    return tuple(generated)

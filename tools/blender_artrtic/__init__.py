bl_info = {
    "name": "ArtRTic",
    "author": "ArtRTic",
    "version": (1, 1, 0),
    "blender": (4, 0, 0),
    "location": "View3D > Sidebar > ArtRTic",
    "description": "Export scene.json and one cooked .tar per mesh (reusable assets)",
    "category": "Import-Export",
}

import json
import os
import shutil
import subprocess
import tempfile

import bpy
from bpy.props import EnumProperty, StringProperty
from bpy.types import AddonPreferences, Operator, Panel


def addon_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def addon_bin_dir() -> str:
    return os.path.join(addon_dir(), "bin")


def resolve_cook(scene: bpy.types.Scene) -> tuple[list[str], str | None]:
    """Return (argv prefix, ARTRTIC_TOOLS_DIR or None)."""
    override = (scene.artrtic_cook_path or "").strip()
    if override:
        return override.split(), None
    bundled = os.path.join(addon_bin_dir(), "artrtic-cook")
    if os.path.isfile(bundled):
        return [bundled], addon_bin_dir()
    return ["artrtic-cook"], None


def cook_status_label(scene: bpy.types.Scene) -> str:
    argv, tools = resolve_cook(scene)
    cook = argv[0]
    if tools:
        comp = os.path.join(tools, "compressonatorcli")
        if os.path.isfile(comp):
            return "Bundled cook + compressonator in addon/bin"
        return "Bundled artrtic-cook (add compressonatorcli to addon/bin)"
    return f"Cook: {cook}"


def ensure_images_on_disk() -> list[str]:
    """Return list of image names that have no file on disk."""
    missing = []
    for image in bpy.data.images:
        if image.packed_file is not None:
            continue
        if not image.filepath:
            if image.size[0] > 0:
                missing.append(image.name)
            continue
        path = bpy.path.abspath(image.filepath)
        if not os.path.isfile(path):
            missing.append(image.name)
    return missing


def unique_asset_slug(base: str, used: set[str]) -> str:
    slug = bpy.path.clean_name(base)
    if not slug:
        slug = "mesh"
    candidate = slug
    index = 2
    while candidate in used:
        candidate = f"{slug}_{index}"
        index += 1
    used.add(candidate)
    return candidate


def export_mesh_tar(
    context: bpy.types.Context,
    obj: bpy.types.Object,
    cook_argv: list[str],
    tar_path: str,
    tools_dir: str | None,
) -> tuple[bool, str]:
    temp_dir = tempfile.mkdtemp(prefix="artrtic-blender-")
    try:
        obj_name = bpy.path.clean_name(obj.name)
        obj_path = os.path.join(temp_dir, f"{obj_name}.obj")

        view_layer = context.view_layer
        prev_active = view_layer.objects.active
        prev_selected = [o for o in context.scene.objects if o.select_get()]
        for o in prev_selected:
            o.select_set(False)
        obj.select_set(True)
        view_layer.objects.active = obj

        try:
            bpy.ops.wm.obj_export(
                filepath=obj_path,
                export_selected_objects=True,
                export_uv=True,
                export_normals=True,
                export_materials=True,
                export_triangulated_mesh=True,
                path_mode="COPY",
            )
        except Exception as err:
            return False, f"OBJ export failed for '{obj.name}': {err}"
        finally:
            for o in prev_selected:
                o.select_set(True)
            view_layer.objects.active = prev_active

        if not os.path.isfile(obj_path):
            return False, f"no OBJ written for '{obj.name}'"

        env = os.environ.copy()
        if tools_dir:
            env["ARTRTIC_TOOLS_DIR"] = tools_dir
        completed = subprocess.run(
            [*cook_argv, "obj", temp_dir, tar_path],
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "").strip()
            return False, detail or "artrtic-cook obj failed"
        return True, tar_path
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


class ARTRTIC_OT_install_tools(Operator):
    bl_idname = "artrtic.install_tools"
    bl_label = "Install cook tools into addon"
    bl_description = "Build artrtic-cook and link Compressonator into addon/bin (run once)"

    def execute(self, context):
        script = os.path.join(addon_dir(), "install.sh")
        if not os.path.isfile(script):
            self.report({"ERROR"}, "install.sh missing next to the add-on")
            return {"CANCELLED"}
        completed = subprocess.run(
            ["bash", script],
            cwd=addon_dir(),
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            tail = (completed.stderr or completed.stdout or "").strip()[-300:]
            self.report({"ERROR"}, tail or "install.sh failed")
            return {"CANCELLED"}
        self.report({"INFO"}, "addon/bin is ready (artrtic-cook + tools)")
        return {"FINISHED"}


class ARTRTIC_AddonPreferences(AddonPreferences):
    bl_idname = __name__

    def draw(self, context):
        layout = self.layout
        layout.operator("artrtic.install_tools", icon="FILE_REFRESH")
        layout.label(text="Or run tools/blender_artrtic/install.sh from the repo.", icon="INFO")


class ARTRTIC_OT_export(Operator):
    bl_idname = "artrtic.export_scene"
    bl_label = "Export ArtRTic scene"
    bl_description = (
        "Write scene.json and assets/<mesh>.tar per mesh via artrtic-cook obj "
        "(uses addon/bin/artrtic-cook when present)"
    )

    def execute(self, context):
        scene = context.scene
        export_root = bpy.path.abspath(scene.artrtic_export_dir)
        assets_dir = os.path.join(export_root, "assets")
        os.makedirs(assets_dir, exist_ok=True)

        missing_images = ensure_images_on_disk()
        if missing_images:
            names = ", ".join(missing_images[:8])
            if len(missing_images) > 8:
                names += ", …"
            self.report(
                {"ERROR"},
                f"Pack or save image files first: {names}",
            )
            return {"CANCELLED"}

        cook_argv, tools_dir = resolve_cook(scene)
        if not os.path.isfile(cook_argv[0]) and cook_argv[0] != "artrtic-cook":
            self.report({"ERROR"}, f"Cook binary not found: {cook_argv[0]}")
            return {"CANCELLED"}
        used_slugs: set[str] = set()
        nodes = []
        cooked = 0
        errors = []

        mesh_objects = [
            obj
            for obj in context.scene.objects
            if obj.type == "MESH" and obj.data and len(obj.data.polygons) > 0
        ]
        if scene.artrtic_export_scope == "SELECTED":
            selected = set(context.selected_objects)
            mesh_objects = [obj for obj in mesh_objects if obj in selected]

        for obj in context.scene.objects:
            if obj.type not in {"MESH", "EMPTY"}:
                continue
            if scene.artrtic_export_scope == "SELECTED" and obj.type == "EMPTY":
                if obj not in context.selected_objects:
                    continue

            parent = obj.parent.name if obj.parent else None
            translation = [obj.location.x, obj.location.y, obj.location.z]
            object_tar = None

            if obj.type == "MESH" and obj in mesh_objects:
                slug = unique_asset_slug(obj.name, used_slugs)
                tar_abs = os.path.join(assets_dir, f"{slug}.tar")
                ok, message = export_mesh_tar(
                    context, obj, cook_argv, tar_abs, tools_dir
                )
                if not ok:
                    errors.append(f"{obj.name}: {message}")
                    continue
                object_tar = f"assets/{slug}.tar"
                cooked += 1

            nodes.append(
                {
                    "name": obj.name,
                    "parent": parent,
                    "translation": translation,
                    "object": object_tar,
                }
            )

        scene_path = os.path.join(export_root, "scene.json")
        with open(scene_path, "w", encoding="utf-8") as handle:
            json.dump({"nodes": nodes}, handle, indent=2)

        if errors:
            self.report(
                {"ERROR"},
                f"Wrote scene.json; {cooked} tar(s) ok, {len(errors)} failed. First: {errors[0][:200]}",
            )
            return {"CANCELLED"}

        if cooked == 0:
            self.report(
                {"WARNING"},
                f"Wrote {scene_path} but no mesh tars (select meshes or add geometry).",
            )
        else:
            self.report(
                {"INFO"},
                f"Wrote {scene_path} and {cooked} asset tar(s) under assets/",
            )
        return {"FINISHED"}


class ARTRTIC_PT_panel(Panel):
    bl_label = "ArtRTic"
    bl_idname = "ARTRTIC_PT_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "ArtRTic"

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        layout.prop(scene, "artrtic_export_dir")
        layout.prop(scene, "artrtic_export_scope")
        layout.label(text=cook_status_label(scene), icon="TOOL_SETTINGS")
        layout.prop(scene, "artrtic_cook_path")
        layout.operator("artrtic.install_tools", icon="IMPORT")
        layout.operator("artrtic.export_scene", icon="EXPORT")
        box = layout.box()
        box.label(text="Output layout:", icon="FILE_FOLDER")
        box.label(text="  scene.json  — node graph + transforms")
        box.label(text="  assets/*.tar — one reusable object each")
        box.label(text="Run the engine with CWD = export folder.")


classes = (
    ARTRTIC_AddonPreferences,
    ARTRTIC_OT_install_tools,
    ARTRTIC_OT_export,
    ARTRTIC_PT_panel,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.artrtic_export_dir = StringProperty(
        name="Export directory",
        subtype="DIR_PATH",
        default="//artrtic_export",
    )
    bpy.types.Scene.artrtic_cook_path = StringProperty(
        name="Cook override",
        description="Leave empty to use addon/bin/artrtic-cook",
        default="",
    )
    bpy.types.Scene.artrtic_export_scope = EnumProperty(
        name="Meshes",
        items=[
            ("ALL", "All meshes", "Export every mesh object in the scene"),
            ("SELECTED", "Selected only", "Export only selected mesh objects"),
        ],
        default="ALL",
    )


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    del bpy.types.Scene.artrtic_export_dir
    del bpy.types.Scene.artrtic_cook_path
    del bpy.types.Scene.artrtic_export_scope

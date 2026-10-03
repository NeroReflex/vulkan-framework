# 12. Moving surfels

A surfel is born on a surface that can move. The slot id does not move. The world center does.

## What is stored

`init_surfel` writes two positions.

- `bind_*` is the hit in the instance's local space: `inverse(node_world[instance_id]) * world_hit`.
- `position_*` is the world center the hash, mark, and lighting actually read.
- `bone` is `0xFFFFFFFF` for a rigid instance. A skinned hit can store the joint with the largest weight. The world pass then uses `bone_palette[bone]` instead of the node matrix.

Static meshes stay on the 32-byte vertex stream and a BLAS built with `PREFER_FAST_TRACE`. A model whose tar entry is `models/<name>/skin` is built with `PREFER_FAST_BUILD | ALLOW_UPDATE`. `shaders/skin/deform.comp` writes the skinned positions that rebuild consumes. `shaders/skin/animate.comp` writes the palette with the same parent walk as the CPU reference in `scene/skin.rs`: `global = local * global`, then `palette = global * inverse_bind`.

## When the centers are written

`surfel_world.comp` runs twice in the GI command buffer.

1. Before mark, so the far-plane test and the radius update see this frame's world center.
2. After commit, so newborns that just landed in the committed half are transformed before the hash inserts them.

The hash still keys off `position_*`. It does not know about nodes. Moving a node sets `index_topology_dirty`, which forces the index plan to rebuild. Slot ids stay where commit put them.

The node buffer has 4096 matrices, one per draw instance, in the same order as the TLAS instances. A scene node with many meshes writes the same world matrix into each of those slots. The raster shader builds that matrix with `row_major_3x4`, matching Vulkan's row-major 3x4 instance transform.

## Scene file

`scene.json` next to the working directory replaces the hardcoded Sponza instance. After loading, the renderer still binds the TLAS for ray tracing and adds the default directional lights (same as the legacy `crytek_sponza.tar` path). Each node has a name, an optional parent, a translation, and an optional tar path. The world matrix is parent times local. `System::load_scene_file` loads each tar and adds one TLAS instance at that matrix. Later edits go through `replace_instance` and one TLAS rebuild.

## Preview

`ART_RTIC_PREVIEW=1` listens on `ws://127.0.0.1:9761`. Each time frame slot 0 is reused, the previous slot's GI image has already been blitted to 640×360 and copied out. That JPEG is what the VS Code extension shows. The lag is one full cycle of the frames in flight. The extension sends `look dx dy forward strafe` and the spectator camera applies it.

Creating a tar still requires Compressonator and `toktx`. The engine never invokes them. `artrtic-cook` does, and only when you ask it to cook.

Blender artists: `tools/blender_artrtic/install.sh` builds `artrtic-cook` into `addon/bin/` and links Compressonator there — no PATH setup. See `tools/blender_artrtic/README.md`.

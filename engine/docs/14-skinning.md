# 14. Skinning and animation barriers

Skinned object archives carry `skeleton/original`, `skeleton/armature`, optional `animations/<clip>/channels`, and a `skinned_vertex_buffer` for the deform pass. A model whose tar entry is `models/<name>/skin` gets a BLAS built with `PREFER_FAST_BUILD | ALLOW_UPDATE`. Static meshes stay on the 32-byte vertex stream with `PREFER_FAST_TRACE`.

Each frame, before the G-buffer, `record_skinning` runs:

1. **`animate_channels.comp` or `animate_bind_pose.comp`** writes the bone skinning matrices into a per-frame SSBO (`per_frame_skeleton`). Push constants supply the animation time in ticks and the active channel count.
2. **`animate.comp`** copies `per_frame_skeleton` into the shared `bone_palette` buffer, using the same parent walk as the CPU reference in `scene/skin.rs`: `global = local * global`, then `palette = global * inverse_bind`. The GI set binds the palette at binding 16.
3. **`deform.comp`** applies `bone_palette` to the packed skinned vertex stream and writes deformed positions.

## Barriers (same queue family)

- After step 1, a **buffer memory barrier** on `per_frame_skeleton`: `COMPUTE_SHADER` / `SHADER_WRITE` to `COMPUTE_SHADER` / `SHADER_READ`.
- After step 2, the same compute-to-compute barrier on **`bone_palette`**, before deform and before any pass that reads the palette.
- After step 3, barrier the **deformed vertex buffer** from compute write to acceleration-structure build and vertex fetch, before the skinned BLAS is rebuilt.

Channel and bind-pose dispatches use `local_size_x = 32`; palette and deform use `64`. Group counts are `(bone_count + 31) / 32` and `(count + 63) / 64`.

Cook skinned assets with `artrtic-cook fbx <in.fbx> <out.tar>`. Play clips through `System::play_animation`, and list them with `System::list_animations`. Surfels on skinned meshes currently follow the node matrix, not the joints (chapter 13).

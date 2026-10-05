# Skinning and animation barriers

Skinned object archives carry `skeleton/original`, `skeleton/armature`, optional `animations/<clip>/channels`, plus `skinned_vertex_buffer` for the deform pass. Each frame the GI command buffer runs, in order:

1. **`animate_channels.comp` or `animate_bind_pose.comp`** — writes bone skinning matrices into a per-frame SSBO (`per_frame_skeleton`). Push constants supply animation time in ticks and the active channel count.
2. **`animate.comp`** — copies `per_frame_skeleton` into the shared `bone_palette` buffer used by surfel world and deform.
3. **`deform.comp`** — applies `bone_palette` to the packed skinned vertex stream and writes deformed positions.

## Barriers (same queue family)

After step 1, emit a **buffer memory barrier** on `per_frame_skeleton`:

- `srcStageMask`: `COMPUTE_SHADER`, `srcAccessMask`: `SHADER_WRITE`
- `dstStageMask`: `COMPUTE_SHADER`, `dstAccessMask`: `SHADER_READ`

After step 2, barrier **`bone_palette`** with the same compute→compute write→read pattern before deform and before any pass that reads the palette (surfel world, BLAS update).

After step 3, barrier the **deformed vertex buffer** from compute write to acceleration-structure build / vertex fetch read when rebuilding skinned BLASes.

Channel and bind-pose dispatches use `local_size_x = 32`; palette and deform use `64`. Group counts are `(bone_count + 31) / 32` and `(count + 63) / 64` respectively.

Cook skinned assets with `artrtic-cook fbx <in.fbx> <out.tar>`. Play clips through `System::play_animation` and list names with `System::list_animations`.

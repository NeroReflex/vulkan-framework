# 9. Composite

After the GI command buffer returns, `system.rs` barriers from ray tracing to graphics and runs three passes.

## Final shading

`final_rendering.frag` samples:

| Binding | Source |
|---|---|
| G-buffer | world position, normal, albedo (albedo is not what fills the screen today) |
| `surfels_buffer` | the overlapping image, as a `usampler2D` |
| `gibuffer[0]` | indirect image |
| `gibuffer[1]` | direct image |

The output color is `direct + indirect`. If `SHOW_SURFELS` is 1 and the overlapping id is not `0xFFFFFFFF`, it adds `vec3(250, 0, 0)`. That is the red you see on covered pixels. It is not lighting. A bright red field means discovery hit. A flickering red field means the hash or the lifetime is changing which pixels hit, not that the tonemap is unstable.

The G-buffer position and normal are loaded and unused by the color equation. They are available if a later change wants to evaluate `projected_irradiance` here instead of in the VPL pass.

## HDR

`hdr_transform` reads the shaded image and writes a display-referred image. It does not know about surfels.

## UI overlay (optional)

When UI is enabled (default), Dear ImGui is rendered offscreen into an `R8G8B8A8` image (`VulkanUiRenderer` in `artrtic_imgui`). `ui_composite` samples the HDR image and the UI texture, alpha-blends UI over the scene, and writes the same `R32G32B32A32` format `renderquad` already expects. ImGui never targets the swapchain.

Set `ART_RTIC_NO_UI=1` to skip ImGui recording and compositing; `renderquad` reads the HDR image directly (headless/CI).

## Present

`renderquad` draws the composed (or HDR-only) image to the swapchain. `prev_frame_gi_reuse` increments after a successful present.

## Debug bisect

`ART_RTIC_STOP_AFTER` skips everything after the named stage: `none`, `mesh`, `gi`, `final`, `hdr`, `ui`, or `all` (default). `gi` still includes discovery, spawn, surfel RT, the empty GI ray generator, and VPL, because those are all inside `record_rendering_commands`. To see the red overlay you must run at least through `final`. Stopping after `gi` presents nothing new from this book. The previous swapchain image just stays up, or the pass returns before present, depending on the no-present flag. Use `ui` to record through UI composite without `renderquad`.

Validation with `debugPrintf` is off unless `ENABLE_DEBUG_PRINTF` is 1. Leaving it on instruments every invocation under GPU-assisted validation and will reset the device. The comment in `config.glsl` is there because that happened.

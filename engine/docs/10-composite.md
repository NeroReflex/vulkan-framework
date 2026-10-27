# 10. Composite

After the GI section, `system.rs` runs final shading, HDR, the optional UI composite, and the present quad.

## Final shading

`final_rendering.frag` reads the G-buffer albedo, the overlay (`surfels_buffer`), and four GI images (`gibuffer[4]` = `IMG_EIND`, `IMG_EDIR`, `IMG_EREF`, `IMG_COVERAGE`). What it writes depends on `debug_data.view_mode`:

| View | Index | Output |
|---|---|---|
| Final | 0 | \(\rho\,(E_\text{dir} + E_\text{ind})\) |
| Surfel ids | 1 | final, tinted by a hash color of the dominant surfel |
| Surfel coverage | 2 | cover `sum_ok`: red→yellow below 0.5, green near 1, green→blue above 1.5 |
| Indirect | 3 | \(\rho\,E_\text{ind}\) |
| Indirect irradiance | 4 | \(E_\text{ind}\) |
| Reference | 5 | \(\rho\,(E_\text{dir} + E_\text{ref})\) |
| Reference indirect | 6 | \(\rho\,E_\text{ref}\) |
| Difference | 7 | relative luminance error of \(E_\text{ind}\) against \(E_\text{ref}\): red = too bright, blue = too dark, grey = match |

Sky pixels are black in every view except coverage. The three reference views switch the path tracer on automatically (`DebugState::reference_active`). They are meaningful only once it has accumulated for a while: it restarts whenever the camera moves. The GI image does not converge to the reference. With the camera still it converges to the surfel cache's own answer, which differs from the reference by the error measured in chapter 11.

"Show surfels" with the Final view selects the surfel-id view.

## The debug panel

The ImGui panel has the view combo, the "Show surfels" checkbox, and one checkbox per GI switch: multi-bounce, ReSTIR temporal reuse, ReSTIR spatial reuse, accumulate indirect, spawn surfels, converge when still (chapter 9), and path traced reference. Changing any switch other than converge and the reference resets the GI history, because the old history no longer means the same thing. "Reset GI" resets it explicitly. The console command `resetsurfels` does the same.

The same settings can be given at startup:

- `ART_RTIC_VIEW=<index>` selects the view (index from the table);
- `ART_RTIC_GI_DISABLE=multibounce,temporal,spatial,accumulate,spawn,converge` turns switches off;
- `ART_RTIC_GI_REFERENCE=1` turns the reference on.

## HDR, UI, present

`hdr_transform` maps the shaded image to display range. With UI enabled (the default), Dear ImGui renders offscreen into an `R8G8B8A8` image, and `ui_composite` blends it over the HDR image. ImGui never targets the swapchain. `ART_RTIC_NO_UI=1` skips the UI, and `renderquad` reads the HDR image directly. `renderquad` draws to the swapchain, and `prev_frame_gi_reuse` increments after the present.

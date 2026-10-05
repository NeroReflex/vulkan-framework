//! Dear ImGui + SDL2 input; Vulkan drawing via [`VulkanUiRenderer`].

mod renderer;
mod sdl2_input;

pub use imgui;
pub use renderer::{snapshot_draw_data, UiDrawSnapshot, VulkanUiRenderer};
pub use sdl2_input::Sdl2Input;

use imgui::{Context, TextureId};
use sdl2::video::Window;

/// Default texture id assigned to the font atlas (must match [`VulkanUiRenderer`]).
pub const FONT_TEXTURE_ID: TextureId = TextureId::new(1);

/// Dear ImGui requires a built font atlas before every `Context::frame()`.
/// Call this after [`VulkanUiRenderer::upload_font`] clears CPU tex data, and once at init.
pub fn ensure_font_atlas_built(imgui: &mut Context) {
    let mut fonts = imgui.fonts();
    if fonts.is_built() {
        return;
    }
    fonts.tex_id = FONT_TEXTURE_ID;
    fonts.build_rgba32_texture();
}

pub fn attach_sdl2(imgui: &mut Context, window: &Window) -> Sdl2Input {
    imgui.set_ini_filename(None);
    Sdl2Input::new(imgui, window)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn font_atlas_rebuild_after_clear_tex_data() {
        let mut ctx = Context::create();
        ensure_font_atlas_built(&mut ctx);
        // Matches GPU upload path in `VulkanUiRenderer::upload_font`.
        ctx.fonts().clear_tex_data();
        ensure_font_atlas_built(&mut ctx);
        ctx.io_mut().display_size = [640.0, 480.0];
        let ui = ctx.frame();
        ui.render();
    }
}

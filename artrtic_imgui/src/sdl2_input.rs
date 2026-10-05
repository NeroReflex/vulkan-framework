//! SDL2 → Dear ImGui input without `SDL_CaptureMouse` (breaks window-space coordinates on Linux).

use imgui::{ConfigFlags, Context, Io, Key, MouseCursor, Ui};
use sdl2::event::Event;
use sdl2::keyboard::Scancode;
use sdl2::mouse::{Cursor, MouseButton, MouseState, SystemCursor};
use sdl2::video::Window;

pub struct Sdl2Input {
    mouse_press: [bool; 5],
    cursor: Option<MouseCursor>,
    sdl_cursor: Option<Cursor>,
}

impl Sdl2Input {
    pub fn new(imgui: &mut Context, window: &Window) -> Self {
        map_keys(imgui);
        let clipboard_util = window.subsystem().clipboard();
        imgui.set_clipboard_backend(Sdl2ClipboardBackend(clipboard_util));
        Self {
            mouse_press: [false; 5],
            cursor: None,
            sdl_cursor: None,
        }
    }

    pub fn handle_event(&mut self, imgui: &mut Context, event: &Event) {
        use sdl2::keyboard;

        fn set_mod(imgui: &mut Context, keymod: keyboard::Mod) {
            let io = imgui.io_mut();
            io.key_ctrl = keymod.intersects(keyboard::Mod::RCTRLMOD | keyboard::Mod::LCTRLMOD);
            io.key_alt = keymod.intersects(keyboard::Mod::RALTMOD | keyboard::Mod::LALTMOD);
            io.key_shift = keymod.intersects(keyboard::Mod::RSHIFTMOD | keyboard::Mod::LSHIFTMOD);
            io.key_super = keymod.intersects(keyboard::Mod::RGUIMOD | keyboard::Mod::LGUIMOD);
        }

        match *event {
            Event::MouseWheel { y, .. } => {
                imgui.io_mut().mouse_wheel = y as f32;
            }
            Event::MouseButtonDown { mouse_btn, .. } => {
                if let Some(index) = mouse_button_index(mouse_btn) {
                    self.mouse_press[index] = true;
                }
            }
            Event::TextInput { ref text, .. } => {
                for chr in text.chars() {
                    imgui.io_mut().add_input_character(chr);
                }
            }
            Event::KeyDown {
                scancode, keymod, ..
            } => {
                set_mod(imgui, keymod);
                if let Some(scancode) = scancode {
                    imgui.io_mut().keys_down[scancode as usize] = true;
                }
            }
            Event::KeyUp {
                scancode, keymod, ..
            } => {
                set_mod(imgui, keymod);
                if let Some(scancode) = scancode {
                    imgui.io_mut().keys_down[scancode as usize] = false;
                }
            }
            _ => {}
        }
    }

    pub fn prepare_io(
        &mut self,
        io: &mut Io,
        window: &Window,
        mouse_state: &MouseState,
        mouse_pos: [f32; 2],
    ) {
        let (win_w, win_h) = window.size();
        let (draw_w, draw_h) = window.drawable_size();

        let scale_x = if win_w > 0 {
            draw_w as f32 / win_w as f32
        } else {
            1.0
        };
        let scale_y = if win_h > 0 {
            draw_h as f32 / win_h as f32
        } else {
            1.0
        };
        io.display_size = [win_w as f32, win_h as f32];
        io.display_framebuffer_scale = [scale_x, scale_y];
        // Keep hit-testing in the same space as `display_size` (SDL reports logical coords).
        io.mouse_pos = [
            mouse_pos[0].clamp(0.0, win_w as f32),
            mouse_pos[1].clamp(0.0, win_h as f32),
        ];

        io.mouse_down = [
            self.mouse_press[0] || mouse_state.left(),
            self.mouse_press[1] || mouse_state.right(),
            self.mouse_press[2] || mouse_state.middle(),
            self.mouse_press[3] || mouse_state.x1(),
            self.mouse_press[4] || mouse_state.x2(),
        ];
        self.mouse_press = [false; 5];

        // Never capture: imgui-sdl2's `mouse_util.capture(any_mouse_down)` warps coords while dragging.
        window.subsystem().sdl().mouse().capture(false);
    }

    pub fn prepare_render(&mut self, ui: &Ui, window: &Window) {
        let io = ui.io();
        if io.config_flags.contains(ConfigFlags::NO_MOUSE_CURSOR_CHANGE) {
            return;
        }
        let mouse_util = window.subsystem().sdl().mouse();
        match ui.mouse_cursor() {
            Some(mouse_cursor) if !io.mouse_draw_cursor => {
                mouse_util.show_cursor(true);
                let sdl_cursor = match mouse_cursor {
                    MouseCursor::Arrow => SystemCursor::Arrow,
                    MouseCursor::TextInput => SystemCursor::IBeam,
                    MouseCursor::ResizeAll => SystemCursor::SizeAll,
                    MouseCursor::ResizeNS => SystemCursor::SizeNS,
                    MouseCursor::ResizeEW => SystemCursor::SizeWE,
                    MouseCursor::ResizeNESW => SystemCursor::SizeNESW,
                    MouseCursor::ResizeNWSE => SystemCursor::SizeNWSE,
                    MouseCursor::Hand => SystemCursor::Hand,
                    MouseCursor::NotAllowed => SystemCursor::No,
                };
                if self.cursor != Some(mouse_cursor) {
                    let sdl_cursor = Cursor::from_system(sdl_cursor).unwrap();
                    sdl_cursor.set();
                    self.cursor = Some(mouse_cursor);
                    self.sdl_cursor = Some(sdl_cursor);
                }
            }
            _ => {
                self.cursor = None;
                self.sdl_cursor = None;
                mouse_util.show_cursor(false);
            }
        }
    }
}

fn mouse_button_index(mouse_btn: MouseButton) -> Option<usize> {
    match mouse_btn {
        MouseButton::Left => Some(0),
        MouseButton::Right => Some(1),
        MouseButton::Middle => Some(2),
        MouseButton::X1 => Some(3),
        MouseButton::X2 => Some(4),
        MouseButton::Unknown => None,
    }
}

fn map_keys(imgui: &mut Context) {
    let io = imgui.io_mut();
    io.key_map[Key::Tab as usize] = Scancode::Tab as u32;
    io.key_map[Key::LeftArrow as usize] = Scancode::Left as u32;
    io.key_map[Key::RightArrow as usize] = Scancode::Right as u32;
    io.key_map[Key::UpArrow as usize] = Scancode::Up as u32;
    io.key_map[Key::DownArrow as usize] = Scancode::Down as u32;
    io.key_map[Key::PageUp as usize] = Scancode::PageUp as u32;
    io.key_map[Key::PageDown as usize] = Scancode::PageDown as u32;
    io.key_map[Key::Home as usize] = Scancode::Home as u32;
    io.key_map[Key::End as usize] = Scancode::End as u32;
    io.key_map[Key::Delete as usize] = Scancode::Delete as u32;
    io.key_map[Key::Backspace as usize] = Scancode::Backspace as u32;
    io.key_map[Key::Enter as usize] = Scancode::Return as u32;
    io.key_map[Key::Escape as usize] = Scancode::Escape as u32;
    io.key_map[Key::Space as usize] = Scancode::Space as u32;
    io.key_map[Key::A as usize] = Scancode::A as u32;
    io.key_map[Key::C as usize] = Scancode::C as u32;
    io.key_map[Key::V as usize] = Scancode::V as u32;
    io.key_map[Key::X as usize] = Scancode::X as u32;
    io.key_map[Key::Y as usize] = Scancode::Y as u32;
    io.key_map[Key::Z as usize] = Scancode::Z as u32;
}

struct Sdl2ClipboardBackend(sdl2::clipboard::ClipboardUtil);

impl imgui::ClipboardBackend for Sdl2ClipboardBackend {
    fn get(&mut self) -> Option<String> {
        if !self.0.has_clipboard_text() {
            return None;
        }
        self.0.clipboard_text().ok().map(String::from)
    }

    fn set(&mut self, value: &str) {
        let _ = self.0.set_clipboard_text(value);
    }
}

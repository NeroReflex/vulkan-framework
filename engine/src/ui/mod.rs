//! Dear ImGui scene editor and console.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use artrtic_imgui::{
    attach_sdl2, ensure_font_atlas_built, imgui, snapshot_draw_data, Sdl2Input, UiDrawSnapshot,
    VulkanUiRenderer,
};
use imgui::{ChildWindow, Condition, StyleColor};
use sdl2::event::Event;
use vulkan_framework::{
    command_buffer::CommandBufferRecorder,
    image::{
        CommonImageFormat, ConcreteImageDescriptor, Image, Image2DDimensions, ImageDimensions,
        ImageFlags, ImageFormat, ImageMultisampling, ImageTiling, ImageUseAs,
    },
    image_view::{ImageView, ImageViewType},
    memory_heap::MemoryType,
    memory_management::{MemoryManagementTags, MemoryManagerTrait},
    memory_pool::MemoryPoolFeatures,
    queue_family::QueueFamily,
};

use crate::rendering::pipeline::ui_composite::UiComposite;
use crate::rendering::system::System;
use crate::scene::{Mat4, Node};

pub struct UiLayer {
    pub imgui: imgui::Context,
    pub input: Sdl2Input,
    pub renderers: Vec<VulkanUiRenderer>,
    pub composite: UiComposite,
    pub composed_views: Vec<Arc<ImageView>>,
    font_uploaded: bool,
    pub scene_base: PathBuf,
    console_lines: Vec<String>,
    cli_buffer: String,
    selected_node: Option<usize>,
    pub camera_locked: bool,
    mesh_path_buf: String,
    scene_save_path: String,
    framebuffer_width: u32,
    framebuffer_height: u32,
    output_format: ImageFormat,
    device: Arc<vulkan_framework::device::Device>,
    /// Window-relative cursor from SDL pointer events (logical client coordinates).
    cursor_pos: Option<[f32; 2]>,
    draw_snapshot: Option<UiDrawSnapshot>,
}

impl UiLayer {
    pub fn new(
        window: &sdl2::video::Window,
        device: Arc<vulkan_framework::device::Device>,
        memory_manager: &mut dyn MemoryManagerTrait,
        output_format: ImageFormat,
        width: u32,
        height: u32,
        frames_in_flight: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let mut imgui = imgui::Context::create();
        ensure_font_atlas_built(&mut imgui);
        apply_editor_theme(&mut imgui);
        let input = attach_sdl2(&mut imgui, window);
        let (width, height) = window.drawable_size();
        let composite = UiComposite::new(device.clone(), output_format, width, height)?;
        let mut renderers = Vec::new();
        let mut composed_views = Vec::new();
        let dimensions = Image2DDimensions::new(width, height);
        for index in 0..frames_in_flight {
            renderers.push(VulkanUiRenderer::new(device.clone(), memory_manager, width, height)?);
            let image = memory_manager.allocate_resources(
                &MemoryType::device_local(),
                &MemoryPoolFeatures::new(false),
                vec![
                    Image::new(
                        device.clone(),
                        ConcreteImageDescriptor::new(
                            ImageDimensions::from(dimensions),
                            [ImageUseAs::ColorAttachment, ImageUseAs::Sampled].as_slice().into(),
                            ImageMultisampling::SamplesPerPixel1,
                            1,
                            1,
                            output_format,
                            ImageFlags::empty(),
                            ImageTiling::Optimal,
                        ),
                        None,
                        Some(&format!("composed_ui[{index}]")),
                    )?
                    .into(),
                ],
                MemoryManagementTags::default().with_name("ui_composed".to_string()),
            )?;
            composed_views.push(ImageView::new(
                image[0].image(),
                Some(ImageViewType::Image2D),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                Some(&format!("composed_ui_view[{index}]")),
            )?);
        }
        Ok(Self {
            imgui,
            input,
            renderers,
            composite,
            composed_views,
            font_uploaded: false,
            scene_base: PathBuf::from("."),
            console_lines: Vec::new(),
            cli_buffer: String::new(),
            selected_node: None,
            camera_locked: false,
            mesh_path_buf: String::new(),
            scene_save_path: String::from("scene.json"),
            framebuffer_width: width,
            framebuffer_height: height,
            output_format,
            device,
            cursor_pos: None,
            draw_snapshot: None,
        })
    }

    pub fn ensure_framebuffer_size(
        &mut self,
        window: &sdl2::video::Window,
        memory_manager: &mut dyn MemoryManagerTrait,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let (width, height) = window.drawable_size();
        if width == 0 || height == 0 {
            return Ok(());
        }
        if width == self.framebuffer_width && height == self.framebuffer_height {
            return Ok(());
        }
        self.framebuffer_width = width;
        self.framebuffer_height = height;
        let dimensions = Image2DDimensions::new(width, height);
        for renderer in &mut self.renderers {
            renderer.resize(memory_manager, width, height)?;
        }
        self.composite =
            UiComposite::new(self.device.clone(), self.output_format, width, height)?;
        let frames = self.composed_views.len();
        self.composed_views.clear();
        for index in 0..frames {
            let image = memory_manager.allocate_resources(
                &MemoryType::device_local(),
                &MemoryPoolFeatures::new(false),
                vec![
                    Image::new(
                        self.device.clone(),
                        ConcreteImageDescriptor::new(
                            ImageDimensions::from(dimensions),
                            [ImageUseAs::ColorAttachment, ImageUseAs::Sampled].as_slice().into(),
                            ImageMultisampling::SamplesPerPixel1,
                            1,
                            1,
                            self.output_format,
                            ImageFlags::empty(),
                            ImageTiling::Optimal,
                        ),
                        None,
                        Some(&format!("composed_ui[{index}]")),
                    )?
                    .into(),
                ],
                MemoryManagementTags::default().with_name("ui_composed".to_string()),
            )?;
            self.composed_views.push(ImageView::new(
                image[0].image(),
                Some(ImageViewType::Image2D),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                Some(&format!("composed_ui_view[{index}]")),
            )?);
        }
        if self.font_uploaded {
            let font = self.renderers[0].font_descriptor();
            for renderer in &mut self.renderers {
                renderer.set_font_descriptor(font.clone());
            }
        }
        Ok(())
    }

    pub fn handle_event(&mut self, event: &Event) {
        match event {
            Event::MouseMotion { x, y, .. }
            | Event::MouseButtonDown { x, y, .. }
            | Event::MouseButtonUp { x, y, .. } => {
                self.cursor_pos = Some([*x as f32, *y as f32]);
            }
            _ => {}
        }
        self.input.handle_event(&mut self.imgui, event);
    }

    pub fn wants_capture_mouse(&self) -> bool {
        self.imgui.io().want_capture_mouse
    }

    pub fn wants_capture_keyboard(&self) -> bool {
        self.imgui.io().want_capture_keyboard
    }

    pub fn build_frame(
        &mut self,
        window: &sdl2::video::Window,
        mouse_state: &sdl2::mouse::MouseState,
        system: &mut System,
    ) {
        let mut selected = self.selected_node;
        let mut mesh_path_buf = self.mesh_path_buf.clone();
        let mut scene_save_path = self.scene_save_path.clone();
        let mut translation = selected.map(|index| {
            system
                .scene()
                .nodes()[index]
                .local
                .transform_point([0.0, 0.0, 0.0])
        });
        let mut cli_buffer = self.cli_buffer.clone();
        let console_lines = self.console_lines.clone();
        let mut submit_command = false;
        let mut attach_error: Option<String> = None;

        ensure_font_atlas_built(&mut self.imgui);
        let mouse_pos = self.cursor_pos.unwrap_or([
            mouse_state.x() as f32,
            mouse_state.y() as f32,
        ]);
        self.input.prepare_io(self.imgui.io_mut(), window, mouse_state, mouse_pos);
        let ui = self.imgui.frame();
        // One shell window — two sibling ImGui windows were sharing focus/geometry when the
        // Scene title bar was clicked (Console appeared to "steal" Scene's layout).
        const SHELL_W: f32 = 584.0;
        const SHELL_H: f32 = 720.0;
        imgui::Window::new("Editor##artrtic_editor_shell")
            .position([16.0, 16.0], Condition::Always)
            .size([SHELL_W, SHELL_H], Condition::Always)
            .title_bar(false)
            .movable(false)
            .resizable(false)
            .collapsible(false)
            .draw_background(true)
            .save_settings(false)
            .build(&ui, || {
                ui.text("Scene");
                ui.separator();
                ChildWindow::new("Scene body##artrtic_scene_body")
                    .border(true)
                    .size([0.0, 348.0])
                    .movable(false)
                    .build(&ui, || {
                        if ui.button("Add root node") {
                            let next = system.scene().nodes().len();
                            system.scene_mut().add_node(Node {
                                name: format!("node_{next}"),
                                parent: None,
                                local: Mat4::identity(),
                                object: None,
                                object_slot: None,
                            });
                        }
                        ChildWindow::new("Scene tree##artrtic_scene_tree")
                            .border(true)
                            .size([0.0, 200.0])
                            .movable(false)
                            .build(&ui, || {
                                draw_scene_tree(
                                    &ui,
                                    system.scene().nodes(),
                                    None,
                                    &mut selected,
                                );
                            });
                        if let Some(node_index) = selected {
                            if ui.button("Add child node") {
                                let next = system.scene().nodes().len();
                                system.scene_mut().add_node(Node {
                                    name: format!("node_{next}"),
                                    parent: Some(node_index),
                                    local: Mat4::identity(),
                                    object: None,
                                    object_slot: None,
                                });
                            }
                            if ui.button("Remove subtree") {
                                if let Some(slot) = system.scene().nodes()[node_index].object_slot
                                {
                                    let _ = system.detach_object(slot);
                                }
                                system.scene_mut().remove_subtree(node_index);
                                selected = None;
                                translation = None;
                            }
                            if let Some(mut t) = translation {
                                if ui.input_float3("Translation", &mut t).build() {
                                    system.scene_mut().set_local(
                                        node_index,
                                        Mat4::translation(t[0], t[1], t[2]),
                                    );
                                    translation = Some(t);
                                }
                            }
                            ui.input_text("Mesh .tar", &mut mesh_path_buf).build();
                            if ui.button("Attach mesh") && !mesh_path_buf.is_empty() {
                                let path = PathBuf::from(&mesh_path_buf);
                                if let Err(err) = system.attach_object_to_node(node_index, &path)
                                {
                                    attach_error = Some(format!("attach failed: {err}"));
                                }
                            }
                            if ui.button("Detach mesh") {
                                if let Some(slot) =
                                    system.scene().nodes()[node_index].object_slot
                                {
                                    if system.detach_object(slot).is_ok() {
                                        system.scene_mut().nodes_mut()[node_index].object = None;
                                        system.scene_mut().nodes_mut()[node_index].object_slot =
                                            None;
                                    }
                                }
                            }
                            ui.input_text("Save scene.json", &mut scene_save_path).build();
                            if ui.button("Save scene") && !scene_save_path.is_empty() {
                                let path = PathBuf::from(&scene_save_path);
                                if let Err(err) = system.save_scene_file(&path) {
                                    attach_error = Some(format!("save failed: {err}"));
                                }
                            }
                        }
                    });
                ui.text("Console");
                ui.separator();
                ChildWindow::new("Console body##artrtic_console_body")
                    .border(true)
                    .size([0.0, 248.0])
                    .movable(false)
                    .build(&ui, || {
                        ui.text_wrapped(
                            "Commands: load NAME PATH | move NODE x y z | list | play NODE INDEX | lock | unlock",
                        );
                        ChildWindow::new("Log##artrtic_console_log")
                            .border(true)
                            .size([0.0, -36.0])
                            .build(&ui, || {
                                if console_lines.is_empty() {
                                    ui.text_disabled("(output appears here)");
                                } else {
                                    for line in &console_lines {
                                        ui.text(line);
                                    }
                                }
                            });
                        if ui
                            .input_text("##artrtic_console_command", &mut cli_buffer)
                            .hint("type command, press Enter")
                            .enter_returns_true(true)
                            .build()
                        {
                            submit_command = true;
                        }
                    });
            });
        self.input.prepare_render(&ui, window);
        ui.render();
        self.draw_snapshot = Some(snapshot_draw_data(unsafe {
            &*(imgui::sys::igGetDrawData() as *const imgui::DrawData)
        }));

        self.selected_node = selected;
        self.mesh_path_buf = mesh_path_buf;
        self.scene_save_path = scene_save_path;
        if let Some(message) = attach_error {
            self.console_lines.push(message);
        }
        if submit_command {
            let line = cli_buffer;
            self.cli_buffer.clear();
            self.run_command(system, &line);
        } else {
            self.cli_buffer = cli_buffer;
        }
    }

    fn run_command(&mut self, system: &mut System, line: &str) {
        let tokens: Vec<&str> = line.split_whitespace().collect();
        if tokens.is_empty() {
            return;
        }
        self.console_lines.push(line.to_string());
        match tokens[0] {
            "load" if tokens.len() >= 3 => {
                let name = tokens[1];
                let path = tokens[2..].join(" ");
                let index = system.scene_mut().add_node(Node {
                    name: name.to_string(),
                    parent: None,
                    local: Mat4::identity(),
                    object: Some(path.clone()),
                    object_slot: None,
                });
                match system.attach_object_to_node(index, Path::new(&path)) {
                    Ok(()) => self.console_lines.push(format!("Loaded {name}")),
                    Err(err) => self.console_lines.push(format!("load failed: {err}")),
                }
            }
            "move" if tokens.len() == 5 => {
                let name = tokens[1];
                if let (Ok(x), Ok(y), Ok(z)) = (
                    tokens[2].parse::<f32>(),
                    tokens[3].parse::<f32>(),
                    tokens[4].parse::<f32>(),
                ) {
                    if let Some(index) = system.scene().find_by_name(name) {
                        system.scene_mut().set_local(index, Mat4::translation(x, y, z));
                    }
                }
            }
            "list" => {
                for node in system.scene().nodes() {
                    let mut message = format!(
                        "{} — {}",
                        node.name,
                        node.object.as_deref().unwrap_or("(no mesh)")
                    );
                    if let Some(slot) = node.object_slot {
                        let clips = system.list_animations(slot);
                        message.push_str(&format!(" | clips: [{}]", clips.join(", ")));
                    }
                    self.console_lines.push(message);
                }
            }
            "play" if tokens.len() == 3 => {
                let name = tokens[1];
                if let Ok(clip_index) = tokens[2].parse::<usize>() {
                    if let Some(node_index) = system.scene().find_by_name(name) {
                        if let Some(slot) = system.scene().nodes()[node_index].object_slot {
                            let clips = system.list_animations(slot);
                            if let Some(clip_name) = clips.get(clip_index) {
                                let _ = system.play_animation(slot, clip_name);
                            }
                        }
                    }
                }
            }
            "lock" => self.camera_locked = true,
            "unlock" => self.camera_locked = false,
            _ => {}
        }
    }

    pub fn record_gpu(
        &mut self,
        frame_index: usize,
        draw_size: Image2DDimensions,
        memory_manager: &mut dyn MemoryManagerTrait,
        queue_family: Arc<QueueFamily>,
        recorder: &mut CommandBufferRecorder,
        hdr_view: Arc<ImageView>,
    ) -> Result<Arc<ImageView>, Box<dyn std::error::Error>> {
        let draw_data = self
            .draw_snapshot
            .as_ref()
            .ok_or("ImGui draw snapshot missing (build_ui_frame not called)")?;
        if !self.font_uploaded {
            self.renderers[frame_index].upload_font(
                &mut self.imgui,
                memory_manager,
                queue_family.clone(),
                recorder,
            )?;
            let font = self.renderers[frame_index].font_descriptor();
            for renderer in &mut self.renderers {
                renderer.set_font_descriptor(font.clone());
            }
            self.font_uploaded = true;
        }
        self.renderers[frame_index].record_snapshot(
            draw_data,
            memory_manager,
            queue_family.clone(),
            recorder,
        )?;
        let ui_view = self.renderers[frame_index].color_view();
        let composed = self.composed_views[frame_index].clone();
        self.composite.record_rendering_commands(
            queue_family,
            draw_size,
            hdr_view,
            ui_view,
            composed.clone(),
            recorder,
        );
        Ok(composed)
    }
}

pub fn output_format() -> ImageFormat {
    CommonImageFormat::r32g32b32a32_sfloat.into()
}

const PANEL_BG: [f32; 4] = [0.12, 0.12, 0.14, 0.96];
const PANEL_TEXT: [f32; 4] = [0.92, 0.92, 0.94, 1.0];

fn apply_editor_theme(imgui: &mut imgui::Context) {
    let style = imgui.style_mut();
    style.colors[StyleColor::WindowBg as usize] = PANEL_BG;
    style.colors[StyleColor::ChildBg as usize] = PANEL_BG;
    style.colors[StyleColor::Text as usize] = PANEL_TEXT;
    style.colors[StyleColor::TextDisabled as usize] = [0.55, 0.55, 0.58, 1.0];
}

fn draw_scene_tree(
    ui: &imgui::Ui,
    nodes: &[Node],
    parent: Option<usize>,
    selected: &mut Option<usize>,
) {
    for (index, node) in nodes.iter().enumerate() {
        if node.parent != parent {
            continue;
        }
        let label = format!(
            "{}{}",
            node.name,
            node
                .object
                .as_deref()
                .map(|path| format!(" ({path})"))
                .unwrap_or_default()
        );
        let item_id = format!("{label}##scene_node_{index}");
        if imgui::Selectable::new(item_id)
            .selected(*selected == Some(index))
            .build(ui)
        {
            *selected = Some(index);
        }
        draw_scene_tree(ui, nodes, Some(index), selected);
    }
}

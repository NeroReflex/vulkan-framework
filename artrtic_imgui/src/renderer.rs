use std::sync::Arc;

use imgui::{DrawCmd, DrawData, TextureId};
use inline_spirv::inline_spirv;
use vulkan_framework::{
    buffer::{
        AllocatedBuffer, Buffer, BufferSubresourceRange, BufferTrait, BufferUseAs,
        ConcreteBufferDescriptor,
    },
    command_buffer::CommandBufferRecorder,
    descriptor_pool::{
        DescriptorPool, DescriptorPoolConcreteDescriptor, DescriptorPoolSizesConcreteDescriptor,
    },
    descriptor_set::DescriptorSet,
    descriptor_set_layout::DescriptorSetLayout,
    device::Device,
    dynamic_rendering::{
        AttachmentStoreOp, DynamicRendering, DynamicRenderingColorAttachment,
        DynamicRenderingColorDefinition, RenderingAttachmentSetup,
    },
    graphics_pipeline::{
        AttributeType, CullMode, FrontFace, GraphicsPipeline, IndexType, PolygonMode, Rasterizer,
        Scissor, VertexInputAttribute, VertexInputBinding, VertexInputRate, Viewport,
    },
    image::{
        CommonImageFormat, ConcreteImageDescriptor, Image, Image1DTrait, Image2DDimensions,
        Image2DTrait, ImageDimensions, ImageFlags, ImageFormat, ImageLayout, ImageMultisampling,
        ImageTiling, ImageTrait, ImageUseAs,
    },
    image_view::{ImageView, ImageViewType},
    memory_barriers::{BufferMemoryBarrier, ImageMemoryBarrier, MemoryAccess, MemoryAccessAs},
    memory_heap::MemoryType,
    memory_management::{MemoryManagementTags, MemoryManagerTrait},
    memory_pool::{MemoryMap, MemoryPoolBacked},
    pipeline_layout::PipelineLayout,
    pipeline_stage::{PipelineStage, PipelineStages},
    push_constant_range::PushConstanRange,
    queue_family::QueueFamily,
    sampler::{Filtering, MipmapMode, Sampler},
    shader_layout_binding::{BindingDescriptor, BindingType, NativeBindingType},
    shader_stage_access::{ShaderStageAccessIn, ShaderStagesAccess},
    shaders::{fragment_shader::FragmentShader, vertex_shader::VertexShader},
};

const UI_VERTEX_SPV: &[u32] = inline_spirv!(
    r#"
#version 460
layout(location = 0) in vec2 in_pos;
layout(location = 1) in vec2 in_uv;
layout(location = 2) in uint in_color;
layout(location = 0) out vec4 out_color;
layout(location = 1) out vec2 out_uv;
layout(push_constant) uniform Push {
    vec2 scale;
    vec2 translate;
} pc;
void main() {
    out_color = unpackUnorm4x8(in_color);
    out_uv = in_uv;
    gl_Position = vec4(in_pos * pc.scale + pc.translate, 0.0, 1.0);
}
"#,
    glsl,
    vert,
    vulkan1_0,
    entry = "main"
);

const UI_FRAGMENT_SPV: &[u32] = inline_spirv!(
    r#"
#version 460
layout(location = 0) in vec4 in_color;
layout(location = 1) in vec2 in_uv;
layout(binding = 0, set = 0) uniform sampler2D font_tex;
layout(location = 0) out vec4 out_color;
void main() {
    out_color = in_color * texture(font_tex, in_uv);
}
"#,
    glsl,
    frag,
    vulkan1_0,
    entry = "main"
);

/// CPU copy of an ImGui frame's geometry and draw commands (safe across font GPU upload).
#[derive(Clone)]
pub struct UiDrawSnapshot {
    pub total_vtx_count: i32,
    pub vertex_bytes: Vec<u8>,
    pub index_bytes: Vec<u8>,
    pub display_pos: [f32; 2],
    pub display_size: [f32; 2],
    pub framebuffer_scale: [f32; 2],
    pub elements: Vec<UiDrawElement>,
}

#[derive(Clone, Copy)]
pub struct UiDrawElement {
    pub count: u32,
    pub clip_rect: [f32; 4],
    pub idx_offset: u32,
    pub vtx_offset: u32,
}

pub fn snapshot_draw_data(draw_data: &DrawData) -> UiDrawSnapshot {
    let (vertex_bytes, index_bytes) = pack_draw_data(draw_data);
    let mut elements = Vec::new();
    let mut global_idx_offset = 0u32;
    let mut global_vtx_offset = 0u32;
    for draw_list in draw_data.draw_lists() {
        for cmd in draw_list.commands() {
            if let DrawCmd::Elements { count, cmd_params } = cmd {
                elements.push(UiDrawElement {
                    count: count as u32,
                    clip_rect: cmd_params.clip_rect,
                    // Merged VBO/IBO: offsets must include per-draw-list base (see imgui_impl_vulkan.cpp).
                    idx_offset: cmd_params.idx_offset as u32 + global_idx_offset,
                    vtx_offset: cmd_params.vtx_offset as u32 + global_vtx_offset,
                });
            }
        }
        global_idx_offset += draw_list.idx_buffer().len() as u32;
        global_vtx_offset += draw_list.vtx_buffer().len() as u32;
    }
    UiDrawSnapshot {
        total_vtx_count: draw_data.total_vtx_count,
        vertex_bytes,
        index_bytes,
        display_pos: draw_data.display_pos,
        display_size: draw_data.display_size,
        framebuffer_scale: draw_data.framebuffer_scale,
        elements,
    }
}

pub struct VulkanUiRenderer {
    device: Arc<Device>,
    pipeline_layout: Arc<PipelineLayout>,
    graphics_pipeline: Arc<GraphicsPipeline>,
    font_descriptor: Arc<DescriptorSet>,
    color_view: Arc<ImageView>,
    color_dimensions: Image2DDimensions,
    vertex_buffer: Arc<AllocatedBuffer>,
    index_buffer: Arc<AllocatedBuffer>,
    vertex_capacity: u64,
    index_capacity: u64,
    font_texture_id: TextureId,
    /// After the first GPU clear/draw, the UI color image is left in `SHADER_READ_ONLY`.
    color_image_initialized: bool,
}

impl VulkanUiRenderer {
    pub fn new(
        device: Arc<Device>,
        mem_manager: &mut dyn MemoryManagerTrait,
        width: u32,
        height: u32,
    ) -> Result<Self, vulkan_framework::prelude::VulkanError> {
        let dimensions = Image2DDimensions::new(width, height);
        let color_format = CommonImageFormat::r8g8b8a8_unorm.into();
        let color_view = allocate_color_image(device.clone(), mem_manager, dimensions, color_format)?;

        let binding = BindingDescriptor::new(
            ShaderStagesAccess::graphics(),
            BindingType::Native(NativeBindingType::CombinedImageSampler),
            0,
            1,
        );
        let descriptor_set_layout =
            DescriptorSetLayout::new(device.clone(), [binding].as_slice())?;

        let push_access = [ShaderStageAccessIn::Vertex].as_slice().into();
        let pipeline_layout = PipelineLayout::new(
            device.clone(),
            &[descriptor_set_layout],
            &[PushConstanRange::new(0, 16, push_access)],
            Some("ui.pipeline_layout"),
        )?;

        let vertex_shader = VertexShader::new(device.clone(), UI_VERTEX_SPV)?;
        let fragment_shader = FragmentShader::new(device.clone(), UI_FRAGMENT_SPV)?;

        let graphics_pipeline = GraphicsPipeline::new(
            None,
            DynamicRendering::new(
                [DynamicRenderingColorDefinition::new_with_alpha_blend(color_format)].as_slice(),
                None,
                None,
            ),
            ImageMultisampling::SamplesPerPixel1,
            None,
            None,
            None,
            pipeline_layout.clone(),
            [
                VertexInputBinding::new(
                    VertexInputRate::PerVertex,
                    20,
                    [
                        VertexInputAttribute::new(0, 0, AttributeType::Vec2),
                        VertexInputAttribute::new(1, 8, AttributeType::Vec2),
                        VertexInputAttribute::new(2, 16, AttributeType::Uint),
                    ]
                    .as_slice(),
                ),
            ]
            .as_slice(),
            Rasterizer::new(
                PolygonMode::Fill,
                FrontFace::CounterClockwise,
                CullMode::None,
                None,
            ),
            (vertex_shader, None),
            (fragment_shader, None),
            Some("ui.graphics_pipeline"),
        )?;

        let (vertex_buffer, index_buffer) =
            allocate_mesh_buffers(device.clone(), mem_manager, 64 * 1024, 128 * 1024)?;

        let font_descriptor = placeholder_font_descriptor(device.clone())?;

        Ok(Self {
            device,
            pipeline_layout,
            graphics_pipeline,
            font_descriptor,
            color_view,
            color_dimensions: dimensions,
            vertex_buffer,
            index_buffer,
            vertex_capacity: 64 * 1024,
            index_capacity: 128 * 1024,
            font_texture_id: crate::FONT_TEXTURE_ID,
            color_image_initialized: false,
        })
    }

    pub fn font_texture_id(&self) -> TextureId {
        self.font_texture_id
    }

    pub fn color_view(&self) -> Arc<ImageView> {
        self.color_view.clone()
    }

    pub fn font_descriptor(&self) -> Arc<DescriptorSet> {
        self.font_descriptor.clone()
    }

    pub fn set_font_descriptor(&mut self, descriptor: Arc<DescriptorSet>) {
        self.font_descriptor = descriptor;
    }

    pub fn resize(
        &mut self,
        mem_manager: &mut dyn MemoryManagerTrait,
        width: u32,
        height: u32,
    ) -> Result<(), vulkan_framework::prelude::VulkanError> {
        if self.color_dimensions.width() == width && self.color_dimensions.height() == height {
            return Ok(());
        }
        let dimensions = Image2DDimensions::new(width, height);
        let color_format = CommonImageFormat::r8g8b8a8_unorm.into();
        self.color_view =
            allocate_color_image(self.device.clone(), mem_manager, dimensions, color_format)?;
        self.color_dimensions = dimensions;
        self.color_image_initialized = false;
        Ok(())
    }

    pub fn upload_font(
        &mut self,
        imgui: &mut imgui::Context,
        mem_manager: &mut dyn MemoryManagerTrait,
        queue_family: Arc<QueueFamily>,
        recorder: &mut CommandBufferRecorder,
    ) -> Result<(), vulkan_framework::prelude::VulkanError> {
        let mut fonts = imgui.fonts();
        fonts.tex_id = self.font_texture_id;
        let texture = fonts.build_rgba32_texture();
        let (font_descriptor, _) = upload_font_image(
            self.device.clone(),
            mem_manager,
            queue_family,
            recorder,
            texture.width,
            texture.height,
            texture.data,
        )?;
        self.font_descriptor = font_descriptor;
        fonts.clear_tex_data();
        Ok(())
    }

    pub fn record_snapshot(
        &mut self,
        draw_data: &UiDrawSnapshot,
        mem_manager: &mut dyn MemoryManagerTrait,
        queue_family: Arc<QueueFamily>,
        recorder: &mut CommandBufferRecorder,
    ) -> Result<(), vulkan_framework::prelude::VulkanError> {
        if draw_data.total_vtx_count == 0 {
            self.clear_color_target(queue_family.clone(), recorder);
            return Ok(());
        }

        let vertex_bytes = &draw_data.vertex_bytes;
        let index_bytes = &draw_data.index_bytes;
        ensure_mesh_capacity(
            &self.device,
            mem_manager,
            &mut self.vertex_buffer,
            &mut self.index_buffer,
            &mut self.vertex_capacity,
            &mut self.index_capacity,
            vertex_bytes.len() as u64,
            index_bytes.len() as u64,
        )?;

        write_buffer(&self.vertex_buffer, vertex_bytes)?;
        write_buffer(&self.index_buffer, index_bytes)?;

        recorder.pipeline_barriers([
            BufferMemoryBarrier::new(
                [PipelineStage::Host].as_slice().into(),
                [MemoryAccessAs::HostWrite].as_slice().into(),
                [PipelineStage::VertexInput].as_slice().into(),
                [
                    MemoryAccessAs::VertexAttribureRead,
                    MemoryAccessAs::IndexRead,
                ]
                .as_slice()
                .into(),
                BufferSubresourceRange::new(
                    self.vertex_buffer.clone(),
                    0,
                    vertex_bytes.len() as u64,
                ),
                queue_family.clone(),
                queue_family.clone(),
            )
            .into(),
            BufferMemoryBarrier::new(
                [PipelineStage::Host].as_slice().into(),
                [MemoryAccessAs::HostWrite].as_slice().into(),
                [PipelineStage::VertexInput].as_slice().into(),
                [MemoryAccessAs::IndexRead].as_slice().into(),
                BufferSubresourceRange::new(self.index_buffer.clone(), 0, index_bytes.len() as u64),
                queue_family.clone(),
                queue_family.clone(),
            )
            .into(),
        ]);

        let color_image: Arc<dyn ImageTrait> = self.color_view.image();
        self.transition_color_to_attachment(color_image.clone(), queue_family.clone(), recorder);

        let clear = vulkan_framework::clear_values::ColorClearValues::Vec4(0.0, 0.0, 0.0, 0.0);
        recorder.graphics_rendering(
            self.color_dimensions,
            [DynamicRenderingColorAttachment::new(
                self.color_view.clone(),
                RenderingAttachmentSetup::clear(clear),
                AttachmentStoreOp::Store,
            )]
            .as_slice(),
            None,
            None,
            |recorder| {
                let width = draw_data.display_size[0];
                let height = draw_data.display_size[1];
                let display_pos = draw_data.display_pos;
                // Clip rects are in `display_size` space; the UI target is `color_dimensions` pixels.
                // `io.display_framebuffer_scale` can be wrong on some SDL setups — derive scale here.
                let fb = [
                    self.color_dimensions.width() as f32 / width.max(1.0),
                    self.color_dimensions.height() as f32 / height.max(1.0),
                ];
                let imgui_viewport = Viewport::new(
                    0.0,
                    0.0,
                    width * fb[0],
                    height * fb[1],
                    0.0,
                    1.0,
                );
                let full_scissor = Scissor::new(
                    0,
                    0,
                    Image2DDimensions::new(
                        (width * fb[0]).max(1.0) as u32,
                        (height * fb[1]).max(1.0) as u32,
                    ),
                );
                recorder.bind_graphics_pipeline(
                    self.graphics_pipeline.clone(),
                    Some(imgui_viewport),
                    Some(full_scissor),
                );
                recorder.bind_descriptor_sets_for_graphics_pipeline(
                    self.pipeline_layout.clone(),
                    0,
                    &[self.font_descriptor.clone()],
                );
                let scale = [2.0 / width.max(1.0), 2.0 / height.max(1.0)];
                let translate = [
                    -1.0 - display_pos[0] * scale[0],
                    -1.0 - display_pos[1] * scale[1],
                ];
                let push: [f32; 4] = [scale[0], scale[1], translate[0], translate[1]];
                recorder.push_constant(
                    self.pipeline_layout.clone(),
                    [ShaderStageAccessIn::Vertex].as_slice().into(),
                    0,
                    f32_slice_as_bytes(&push),
                );

                recorder.bind_vertex_buffers(
                    0,
                    &[(0, self.vertex_buffer.clone() as Arc<dyn BufferTrait>)],
                );
                recorder.bind_index_buffer(
                    0,
                    self.index_buffer.clone() as Arc<dyn BufferTrait>,
                    IndexType::UInt16,
                );

                for element in &draw_data.elements {
                    let clip = element.clip_rect;
                    let x = ((clip[0] - display_pos[0]) * fb[0]).max(0.0) as i32;
                    let y = ((clip[1] - display_pos[1]) * fb[1]).max(0.0) as i32;
                    let w = ((clip[2] - clip[0]) * fb[0]).max(0.0) as u32;
                    let h = ((clip[3] - clip[1]) * fb[1]).max(0.0) as u32;
                    recorder.bind_graphics_pipeline(
                        self.graphics_pipeline.clone(),
                        Some(imgui_viewport),
                        Some(Scissor::new(x, y, Image2DDimensions::new(w, h))),
                    );
                    recorder.draw_indexed(
                        element.count,
                        1,
                        element.idx_offset,
                        element.vtx_offset as i32,
                        0,
                    );
                }
            },
        );

        self.transition_color_to_shader_read(color_image.clone(), queue_family.clone(), recorder);

        Ok(())
    }

    /// Clears the offscreen UI target and leaves it in `SHADER_READ_ONLY_OPTIMAL` for compositing.
    fn clear_color_target(
        &mut self,
        queue_family: Arc<QueueFamily>,
        recorder: &mut CommandBufferRecorder,
    ) {
        let color_image: Arc<dyn ImageTrait> = self.color_view.image();
        self.transition_color_to_attachment(color_image.clone(), queue_family.clone(), recorder);

        let clear = vulkan_framework::clear_values::ColorClearValues::Vec4(0.0, 0.0, 0.0, 0.0);
        recorder.graphics_rendering(
            self.color_dimensions,
            [DynamicRenderingColorAttachment::new(
                self.color_view.clone(),
                RenderingAttachmentSetup::clear(clear),
                AttachmentStoreOp::Store,
            )]
            .as_slice(),
            None,
            None,
            |_| {},
        );

        self.transition_color_to_shader_read(color_image, queue_family, recorder);
    }

    fn transition_color_to_attachment(
        &mut self,
        color_image: Arc<dyn ImageTrait>,
        queue_family: Arc<QueueFamily>,
        recorder: &mut CommandBufferRecorder,
    ) {
        let old_layout = if self.color_image_initialized {
            ImageLayout::ShaderReadOnlyOptimal
        } else {
            ImageLayout::Undefined
        };
        recorder.pipeline_barriers([ImageMemoryBarrier::new(
            [PipelineStage::TopOfPipe].as_slice().into(),
            MemoryAccess::default(),
            [PipelineStage::ColorAttachmentOutput].as_slice().into(),
            [MemoryAccessAs::ColorAttachmentWrite].as_slice().into(),
            color_image.into(),
            old_layout,
            ImageLayout::ColorAttachmentOptimal,
            queue_family.clone(),
            queue_family.clone(),
        )
        .into()]);
    }

    fn transition_color_to_shader_read(
        &mut self,
        color_image: Arc<dyn ImageTrait>,
        queue_family: Arc<QueueFamily>,
        recorder: &mut CommandBufferRecorder,
    ) {
        recorder.pipeline_barriers([ImageMemoryBarrier::new(
            [PipelineStage::ColorAttachmentOutput].as_slice().into(),
            [MemoryAccessAs::ColorAttachmentWrite].as_slice().into(),
            [PipelineStage::FragmentShader].as_slice().into(),
            [MemoryAccessAs::ShaderRead].as_slice().into(),
            color_image.into(),
            ImageLayout::ColorAttachmentOptimal,
            ImageLayout::ShaderReadOnlyOptimal,
            queue_family.clone(),
            queue_family.clone(),
        )
        .into()]);
        self.color_image_initialized = true;
    }
}

fn f32_slice_as_bytes(values: &[f32; 4]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(values.as_ptr() as *const u8, 16) }
}

fn pack_draw_data(draw_data: &DrawData) -> (Vec<u8>, Vec<u8>) {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    for draw_list in draw_data.draw_lists() {
        for v in draw_list.vtx_buffer() {
            vertices.extend_from_slice(&v.pos[0].to_ne_bytes());
            vertices.extend_from_slice(&v.pos[1].to_ne_bytes());
            vertices.extend_from_slice(&v.uv[0].to_ne_bytes());
            vertices.extend_from_slice(&v.uv[1].to_ne_bytes());
            vertices.extend_from_slice(&v.col);
        }
        // Indices stay draw-list-local; `VtxOffset` / `IdxOffset` in each draw command are adjusted
        // when recording (same as Dear ImGui's Vulkan backend).
        for i in draw_list.idx_buffer() {
            indices.extend_from_slice(&i.to_ne_bytes());
        }
    }
    (vertices, indices)
}

fn write_buffer(
    buffer: &Arc<AllocatedBuffer>,
    data: &[u8],
) -> Result<(), vulkan_framework::prelude::VulkanError> {
    let map = MemoryMap::new(buffer.get_backing_memory_pool())?;
    let mut range = map.range::<u8>(buffer.clone() as Arc<dyn MemoryPoolBacked>)?;
    range.as_mut_slice()[..data.len()].copy_from_slice(data);
    Ok(())
}

fn allocate_color_image(
    device: Arc<Device>,
    mem_manager: &mut dyn MemoryManagerTrait,
    dimensions: Image2DDimensions,
    color_format: ImageFormat,
) -> Result<Arc<ImageView>, vulkan_framework::prelude::VulkanError> {
    let color_image = mem_manager.allocate_resources(
        &MemoryType::device_local(),
        &vulkan_framework::memory_pool::MemoryPoolFeatures::new(false),
        vec![
            Image::new(
                device.clone(),
                ConcreteImageDescriptor::new(
                    ImageDimensions::from(dimensions),
                    [ImageUseAs::ColorAttachment, ImageUseAs::Sampled]
                        .as_slice()
                        .into(),
                    ImageMultisampling::SamplesPerPixel1,
                    1,
                    1,
                    color_format,
                    ImageFlags::empty(),
                    ImageTiling::Optimal,
                ),
                None,
                Some("ui_color_image"),
            )?
            .into(),
        ],
        MemoryManagementTags::default().with_name("ui".to_string()),
    )?;
    ImageView::new(
        color_image[0].image(),
        Some(ImageViewType::Image2D),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        Some("ui_color_view"),
    )
}

fn allocate_mesh_buffers(
    device: Arc<Device>,
    mem_manager: &mut dyn MemoryManagerTrait,
    vertex_bytes: u64,
    index_bytes: u64,
) -> Result<(Arc<AllocatedBuffer>, Arc<AllocatedBuffer>), vulkan_framework::prelude::VulkanError> {
    let usage =
        vulkan_framework::buffer::BufferUsage::from([BufferUseAs::VertexBuffer, BufferUseAs::IndexBuffer].as_slice());
    let v = Buffer::new(
        device.clone(),
        ConcreteBufferDescriptor::new(usage, vertex_bytes),
        None,
        Some("ui_vertices"),
    )?;
    let i = Buffer::new(
        device.clone(),
        ConcreteBufferDescriptor::new(usage, index_bytes),
        None,
        Some("ui_indices"),
    )?;
    let allocs = mem_manager.allocate_resources(
        &MemoryType::host_visible_and_coherent(),
        &vulkan_framework::memory_pool::MemoryPoolFeatures::new(false),
        vec![v.into(), i.into()],
        MemoryManagementTags::default(),
    )?;
    Ok((allocs[0].buffer(), allocs[1].buffer()))
}

fn ensure_mesh_capacity(
    device: &Arc<Device>,
    mem_manager: &mut dyn MemoryManagerTrait,
    vertex: &mut Arc<AllocatedBuffer>,
    index: &mut Arc<AllocatedBuffer>,
    vertex_cap: &mut u64,
    index_cap: &mut u64,
    need_v: u64,
    need_i: u64,
) -> Result<(), vulkan_framework::prelude::VulkanError> {
    if need_v <= *vertex_cap && need_i <= *index_cap {
        return Ok(());
    }
    let new_v = need_v.next_power_of_two().max(4096);
    let new_i = need_i.next_power_of_two().max(4096);
    let (vb, ib) = allocate_mesh_buffers(device.clone(), mem_manager, new_v, new_i)?;
    *vertex = vb;
    *index = ib;
    *vertex_cap = new_v;
    *index_cap = new_i;
    Ok(())
}

fn placeholder_font_descriptor(
    device: Arc<Device>,
) -> Result<Arc<DescriptorSet>, vulkan_framework::prelude::VulkanError> {
    let layout = DescriptorSetLayout::new(
        device.clone(),
        [BindingDescriptor::new(
            ShaderStagesAccess::graphics(),
            BindingType::Native(NativeBindingType::CombinedImageSampler),
            0,
            1,
        )]
        .as_slice(),
    )?;
    let pool = DescriptorPool::new(
        device.clone(),
        DescriptorPoolConcreteDescriptor::new(
            DescriptorPoolSizesConcreteDescriptor::new(0, 1, 0, 0, 0, 0, 0, 0, 0, None),
            1,
        ),
        Some("ui_font_pool_placeholder"),
    )?;
    DescriptorSet::new(pool, layout)
}

fn upload_font_image(
    device: Arc<Device>,
    mem_manager: &mut dyn MemoryManagerTrait,
    queue_family: Arc<QueueFamily>,
    recorder: &mut CommandBufferRecorder,
    width: u32,
    height: u32,
    rgba: &[u8],
) -> Result<(Arc<DescriptorSet>, Arc<ImageView>), vulkan_framework::prelude::VulkanError> {
    let format = CommonImageFormat::r8g8b8a8_unorm.into();
    let dimensions = Image2DDimensions::new(width, height);
    let image = Image::new(
        device.clone(),
        ConcreteImageDescriptor::new(
            ImageDimensions::from(dimensions),
            [ImageUseAs::Sampled, ImageUseAs::TransferDst].as_slice().into(),
            ImageMultisampling::SamplesPerPixel1,
            1,
            1,
            format,
            ImageFlags::empty(),
            ImageTiling::Optimal,
        ),
        None,
        Some("ui_font"),
    )?;
    let allocated = mem_manager.allocate_resources(
        &MemoryType::device_local(),
        &vulkan_framework::memory_pool::MemoryPoolFeatures::new(false),
        vec![image.into()],
        MemoryManagementTags::default(),
    )?;
    let view = ImageView::new(
        allocated[0].image(),
        Some(ImageViewType::Image2D),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        Some("ui_font_view"),
    )?;
    let font_image: Arc<dyn ImageTrait> = allocated[0].image();

    let staging = Buffer::new(
        device.clone(),
        ConcreteBufferDescriptor::new(
            vulkan_framework::buffer::BufferUsage::from([BufferUseAs::TransferSrc].as_slice()),
            rgba.len() as u64,
        ),
        None,
        Some("ui_font_staging"),
    )?;
    let staging_alloc = mem_manager.allocate_resources(
        &MemoryType::host_visible_and_coherent(),
        &vulkan_framework::memory_pool::MemoryPoolFeatures::new(false),
        vec![staging.into()],
        MemoryManagementTags::default(),
    )?;
    let staging_buf = staging_alloc[0].buffer();
    write_buffer(&staging_buf, rgba)?;

    recorder.pipeline_barriers([ImageMemoryBarrier::new(
        [PipelineStage::TopOfPipe].as_slice().into(),
        MemoryAccess::default(),
        [PipelineStage::Transfer].as_slice().into(),
        [MemoryAccessAs::TransferWrite].as_slice().into(),
        font_image.clone().into(),
        ImageLayout::Undefined,
        ImageLayout::TransferDstOptimal,
        queue_family.clone(),
        queue_family.clone(),
    )
    .into()]);

    recorder.copy_buffer_to_image(
        staging_buf.clone() as Arc<dyn BufferTrait>,
        ImageLayout::TransferDstOptimal,
        font_image.clone().into(),
        font_image.clone(),
        ImageDimensions::from(dimensions),
        0,
    );

    recorder.pipeline_barriers([ImageMemoryBarrier::new(
        [PipelineStage::Transfer].as_slice().into(),
        [MemoryAccessAs::TransferWrite].as_slice().into(),
        [PipelineStage::FragmentShader].as_slice().into(),
        [MemoryAccessAs::ShaderRead].as_slice().into(),
        font_image.clone().into(),
        ImageLayout::TransferDstOptimal,
        ImageLayout::ShaderReadOnlyOptimal,
        queue_family.clone(),
        queue_family.clone(),
    )
    .into()]);

    let sampler = Sampler::new(
        device.clone(),
        Filtering::Linear,
        Filtering::Linear,
        MipmapMode::ModeNearest,
        0.0,
    )?;
    let layout = DescriptorSetLayout::new(
        device.clone(),
        [BindingDescriptor::new(
            ShaderStagesAccess::graphics(),
            BindingType::Native(NativeBindingType::CombinedImageSampler),
            0,
            1,
        )]
        .as_slice(),
    )?;
    let pool = DescriptorPool::new(
        device.clone(),
        DescriptorPoolConcreteDescriptor::new(
            DescriptorPoolSizesConcreteDescriptor::new(0, 1, 0, 0, 0, 0, 0, 0, 0, None),
            1,
        ),
        Some("ui_font_pool"),
    )?;
    let set = DescriptorSet::new(pool, layout)?;
    set.bind_resources(|b| {
        b.bind_combined_images_samplers(
            0,
            &[(ImageLayout::ShaderReadOnlyOptimal, view.clone(), sampler)],
        )
        .unwrap();
    })?;

    Ok((set, view))
}

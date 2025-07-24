use std::sync::Arc;

use inline_spirv::*;

use crate::rendering::{MAX_FRAMES_IN_FLIGHT_NO_MALLOC, RenderingResult};

use vulkan_framework::{
    command_buffer::{ClearValues, ColorClearValues, CommandBufferRecorder},
    device::Device,
    framebuffer::Framebuffer,
    graphics_pipeline::{
        CullMode, DepthCompareOp, DepthConfiguration, FrontFace, GraphicsPipeline, PolygonMode,
        Rasterizer, Scissor, Viewport,
    },
    image::{
        AllocatedImage, CommonImageFormat, ConcreteImageDescriptor, Image, Image1DTrait,
        Image2DDimensions, Image2DTrait, ImageDimensions, ImageFlags, ImageFormat, ImageLayout,
        ImageLayoutSwapchainKHR, ImageMultisampling, ImageSubresourceRange, ImageTiling,
        ImageTrait, ImageUsage, ImageUseAs,
    },
    image_view::{ImageView, ImageViewType},
    memory_allocator::DefaultAllocator,
    memory_heap::{ConcreteMemoryHeapDescriptor, MemoryHeap, MemoryType},
    memory_pool::{MemoryPool, MemoryPoolFeatures},
    memory_requiring::MemoryRequiring,
    pipeline_layout::PipelineLayout,
    renderpass::{
        AttachmentDescription, AttachmentLoadOp, AttachmentStoreOp, RenderPass,
        RenderSubPassDescription,
    },
    shaders::{fragment_shader::FragmentShader, vertex_shader::VertexShader},
};

const FINAL_RENDERING_VERTEX_SPV: &[u32] = inline_spirv!(
    r#"
#version 450 core

vec2 positions[3] = vec2[](
    vec2(0.0, -0.5),
    vec2(0.5, 0.5),
    vec2(-0.5, 0.5)
);

void main() {
    gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0);
}
"#,
    vert
);

const FINAL_RENDERING_FRAGMENT_SPV: &[u32] = inline_spirv!(
    r#"
#version 450 core

layout(location = 0) out vec4 outColor;

void main() {
    outColor = vec4(1.0, 0.0, 0.0, 1.0);
}
"#,
    frag
);

pub struct FinalRendering {
    _memory_pool: Arc<MemoryPool>,
    image_dimensions: Image2DDimensions,
    images: smallvec::SmallVec<[Arc<AllocatedImage>; MAX_FRAMES_IN_FLIGHT_NO_MALLOC]>,
    image_views: smallvec::SmallVec<[Arc<ImageView>; MAX_FRAMES_IN_FLIGHT_NO_MALLOC]>,
    framebuffers: smallvec::SmallVec<[Arc<Framebuffer>; MAX_FRAMES_IN_FLIGHT_NO_MALLOC]>,
    graphics_pipeline: Arc<GraphicsPipeline>,
}

impl FinalRendering {
    fn output_image_layout() -> ImageLayout {
        ImageLayout::SwapchainKHR(ImageLayoutSwapchainKHR::PresentSrc)
    }

    fn output_image_format() -> ImageFormat {
        ImageFormat::from(CommonImageFormat::r32g32b32a32_sfloat)
    }

    pub fn new(
        device: Arc<Device>,
        width: u32,
        height: u32,
        frames_in_flight: u32,
    ) -> RenderingResult<Self> {
        let image_dimensions = Image2DDimensions::new(width, height);

        let mut image_handles: smallvec::SmallVec<[_; MAX_FRAMES_IN_FLIGHT_NO_MALLOC]> =
            smallvec::smallvec![];
        for index in 0..(frames_in_flight as usize) {
            let image_name = format!("final_rendering_image[{index}]");
            let image = Image::new(
                device.clone(),
                ConcreteImageDescriptor::new(
                    ImageDimensions::Image2D {
                        extent: image_dimensions,
                    },
                    ImageUsage::from([ImageUseAs::Sampled, ImageUseAs::ColorAttachment].as_slice()),
                    ImageMultisampling::SamplesPerPixel1,
                    1,
                    1,
                    Self::output_image_format(),
                    ImageFlags::empty(),
                    ImageTiling::Optimal,
                ),
                None,
                Some(image_name.as_str()),
            )?;

            image_handles.push(image);
        }

        let memory_required: u64 = image_handles
            .iter()
            .map(|obj| obj.memory_requirements().size() + obj.memory_requirements().alignment())
            .sum();

        let minimum_memory = memory_required + (4096u64 * (frames_in_flight as u64 + 4u64));
        // add space for frames_in_flight images

        // add some leftover space to account for alignment
        //+ (1024 * 1024* 128);

        let hints: smallvec::SmallVec<[&dyn MemoryRequiring; MAX_FRAMES_IN_FLIGHT_NO_MALLOC]> =
            image_handles
                .iter()
                .map(|a| a as &dyn MemoryRequiring)
                .collect();
        let memory_heap = MemoryHeap::new(
            device.clone(),
            ConcreteMemoryHeapDescriptor::new(MemoryType::DeviceLocal(None), minimum_memory),
            hints.as_slice(),
        )?;
        drop(hints);

        let memory_pool = MemoryPool::new(
            memory_heap,
            Arc::new(DefaultAllocator::new(minimum_memory)),
            MemoryPoolFeatures::from([].as_slice()),
        )?;

        let mut image_views: smallvec::SmallVec<[Arc<ImageView>; MAX_FRAMES_IN_FLIGHT_NO_MALLOC]> =
            smallvec::smallvec![];
        let mut images: smallvec::SmallVec<[Arc<AllocatedImage>; MAX_FRAMES_IN_FLIGHT_NO_MALLOC]> =
            smallvec::smallvec![];
        for (index, image) in image_handles.into_iter().enumerate() {
            println!(
                "Allocating {} bytes out of {minimum_memory}",
                image.memory_requirements().size()
            );

            let allocated_image = AllocatedImage::new(memory_pool.clone(), image)?;

            images.push(allocated_image.clone());

            let image_view_name = format!("final_rendering_image_view[{index}]");
            let image_view = ImageView::new(
                allocated_image,
                Some(ImageViewType::Image2D),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                Some(image_view_name.as_str()),
            )?;

            image_views.push(image_view);
        }

        let renderpass = RenderPass::new(
            device.clone(),
            &[
                AttachmentDescription::new(
                    Self::output_image_format(),
                    ImageMultisampling::SamplesPerPixel1,
                    ImageLayout::Undefined,
                    Self::output_image_layout(),
                    AttachmentLoadOp::Clear,
                    AttachmentStoreOp::Store,
                    AttachmentLoadOp::Clear,
                    AttachmentStoreOp::Store,
                ),
                /*
                // depth
                AttachmentDescription::new(
                    final_format,
                    ImageMultisampling::SamplesPerPixel1,
                    ImageLayout::Undefined,
                    ImageLayout::SwapchainKHR(ImageLayoutSwapchainKHR::PresentSrc),
                    AttachmentLoadOp::Clear,
                    AttachmentStoreOp::Store,
                    AttachmentLoadOp::Clear,
                    AttachmentStoreOp::Store,
                )*/
            ],
            &[RenderSubPassDescription::new(&[], &[0], None)],
        )?;

        let vertex_shader =
            VertexShader::new(device.clone(), &[], &[], FINAL_RENDERING_VERTEX_SPV).unwrap();

        let fragment_shader =
            FragmentShader::new(device.clone(), &[], &[], FINAL_RENDERING_FRAGMENT_SPV).unwrap();

        let pipeline_layout = PipelineLayout::new(
            device.clone(),
            &[
                        /*BindingDescriptor::new(
                            shader_access,
                            binding_type,
                            binding_point,
                            binding_count
                        )*/
                    ],
            &[],
            Some("pipeline_layout"),
        )?;

        let graphics_pipeline = GraphicsPipeline::new(
            None,
            renderpass.clone(),
            0,
            ImageMultisampling::SamplesPerPixel1,
            Some(DepthConfiguration::new(
                true,
                DepthCompareOp::Always,
                Some((0.0, 1.0)),
            )),
            None,
            None,
            pipeline_layout,
            &[
                        /*VertexInputBinding::new(
                        VertexInputRate::PerVertex,
                        0,
                        &[VertexInputAttribute::new(0, 0, AttributeType::Vec4)],
                        )*/
                    ],
            Rasterizer::new(
                PolygonMode::Fill,
                FrontFace::CounterClockwise,
                CullMode::None,
                None,
            ),
            (vertex_shader, None),
            (fragment_shader, None),
            Some("final_rendering_graphics_pipeline"),
        )?;

        let mut framebuffers = smallvec::smallvec![];
        for iw in 0..image_views.len() {
            let framebuffer = Framebuffer::new(
                renderpass.clone(),
                [image_views[iw].clone()].as_slice(),
                image_dimensions,
                1,
            )?;

            framebuffers.push(framebuffer);
        }

        Ok(Self {
            image_dimensions,
            _memory_pool: memory_pool,
            graphics_pipeline,
            framebuffers,
            image_views,
            images,
        })
    }

    pub fn record_rendering_commands(
        &self,
        current_frame: usize,
        recorder: &mut CommandBufferRecorder,
    ) -> (Arc<ImageView>, ImageSubresourceRange, ImageLayout) {
        let image_view = self.image_views[current_frame].clone();

        recorder.begin_renderpass(
            self.framebuffers[current_frame].clone(),
            &[ClearValues::new(Some(ColorClearValues::Vec4(
                0.0, 0.0, 0.0, 1.0,
            )))],
        );

        recorder.bind_graphics_pipeline(
            self.graphics_pipeline.clone(),
            Some(Viewport::new(
                0.0f32,
                0.0f32,
                self.image_dimensions.width() as f32,
                self.image_dimensions.height() as f32,
                0.0f32,
                0.0f32,
            )),
            Some(Scissor::new(0, 0, self.image_dimensions)),
        );

        recorder.draw(0, 3, 0, 1);

        recorder.end_renderpass();

        (
            image_view,
            ImageSubresourceRange::from(self.images[current_frame].clone() as Arc<dyn ImageTrait>),
            Self::output_image_layout(),
        )
    }
}

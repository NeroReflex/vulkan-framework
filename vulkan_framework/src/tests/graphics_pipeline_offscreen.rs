#[cfg(test)]
mod graphics_pipeline_offscreen_tests {
    use std::sync::Arc;

    use crate::memory_management::MemoryManagerTrait;
    use crate::prelude::*;
    use inline_spirv::inline_spirv;

    const VERT_SPV: &[u32] = inline_spirv!(
        r#"
#version 450
void main() {
    const vec2 positions[3] = vec2[](vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
    gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0);
}
"#,
        glsl,
        vert,
        vulkan1_2,
        entry = "main"
    );

    const FRAG_SPV: &[u32] = inline_spirv!(
        r#"
#version 450
layout(location = 0) out vec4 out_color;
void main() {
    out_color = vec4(1.0, 0.0, 0.0, 1.0);
}
"#,
        glsl,
        frag,
        vulkan1_2,
        entry = "main"
    );

    #[test]
    fn test_graphics_pipeline_offscreen() -> VulkanResult<()> {
        // Create an Instance + Device requesting a graphics-capable queue (needed for rendering)
        match (|| -> VulkanResult<(Arc<crate::instance::Instance>, Arc<crate::device::Device>)> {
            let instance = crate::instance::Instance::new(
                &[],
                &[],
                &"vulkan_framework_tests".to_string(),
                &"test".to_string(),
            )?;
            let queue_descriptor = crate::queue_family::ConcreteQueueFamilyDescriptor::new(
                &[crate::queue_family::QueueFamilySupportedOperationType::Graphics],
                &[1.0],
            );
            let device = crate::device::Device::new(
                instance.clone(),
                &[queue_descriptor],
                &[],
                Some("test_device_graphics"),
            )?;
            Ok((instance, device))
        })() {
            Ok((_instance, device)) => {
                let queue_family = crate::queue_family::QueueFamily::new(device.clone(), 0)?;
                let command_pool =
                    crate::command_pool::CommandPool::new(queue_family.clone(), Some("test_pool"))?;
                let cmd_buffer = crate::command_buffer::PrimaryCommandBuffer::new(
                    command_pool.clone(),
                    Some("test_cb"),
                )?;
                let queue = crate::queue::Queue::new(queue_family.clone(), Some("test_queue"))?;

                let width = 4u32;
                let height = 4u32;

                let image_desc = crate::image::ConcreteImageDescriptor::new(
                    crate::image::ImageDimensions::from(crate::image::Image2DDimensions::new(
                        width, height,
                    )),
                    crate::image::ImageUsage::from(
                        &[crate::image::ImageUseAs::ColorAttachment][..],
                    ),
                    crate::image::ImageMultisampling::SamplesPerPixel1,
                    1,
                    1,
                    crate::image::CommonImageFormat::r8g8b8a8_unorm.into(),
                    crate::image::ImageFlags::empty(),
                    crate::image::ImageTiling::Optimal,
                );

                let image_unalloc =
                    crate::image::Image::new(device.clone(), image_desc, None, Some("test_image"))?;

                let mut mem_mgr =
                    crate::memory_management::DefaultMemoryManager::new(device.clone());
                let allocations = mem_mgr.allocate_resources(
                    &crate::memory_heap::MemoryType::device_local(),
                    &crate::memory_pool::MemoryPoolFeatures::new(false),
                    vec![image_unalloc.into()],
                    crate::memory_management::MemoryManagementTags::default(),
                )?;

                let image_alloc = allocations[0].image();

                let image_trait: Arc<dyn crate::image::ImageTrait> = image_alloc.clone();

                let image_view = crate::image_view::ImageView::new(
                    image_trait,
                    None::<crate::image_view::ImageViewType>,
                    None::<crate::image::ImageFormat>,
                    None::<crate::image_view::ImageViewAspect>,
                    None::<crate::image_view::ImageViewColorMapping>,
                    None::<u32>,
                    None::<u32>,
                    None::<u32>,
                    None::<u32>,
                    Some("test_image_view"),
                )?;

                let color_def = crate::dynamic_rendering::DynamicRenderingColorDefinition::new(
                    crate::image::CommonImageFormat::r8g8b8a8_unorm.into(),
                );

                let dyn_render =
                    crate::dynamic_rendering::DynamicRendering::new(&[color_def], None, None);

                let pipeline_layout = crate::pipeline_layout::PipelineLayout::new(
                    device.clone(),
                    &[],
                    &[],
                    Some("test_pipeline_layout"),
                )?;

                let vertex_shader =
                    crate::shaders::vertex_shader::VertexShader::new(device.clone(), VERT_SPV)?;
                let fragment_shader =
                    crate::shaders::fragment_shader::FragmentShader::new(device.clone(), FRAG_SPV)?;

                let rasterizer = crate::graphics_pipeline::Rasterizer::new(
                    crate::graphics_pipeline::PolygonMode::Fill,
                    crate::graphics_pipeline::FrontFace::CounterClockwise,
                    crate::graphics_pipeline::CullMode::None,
                    None,
                );

                let viewport = crate::graphics_pipeline::Viewport::new(
                    0.0,
                    0.0,
                    width as f32,
                    height as f32,
                    0.0,
                    1.0,
                );
                let scissor = crate::graphics_pipeline::Scissor::new(
                    0,
                    0,
                    crate::image::Image2DDimensions::new(width, height),
                );

                let pipeline = crate::graphics_pipeline::GraphicsPipeline::new(
                    None,
                    dyn_render,
                    crate::image::ImageMultisampling::SamplesPerPixel1,
                    None,
                    Some(viewport),
                    Some(scissor),
                    pipeline_layout.clone(),
                    &[],
                    rasterizer,
                    (vertex_shader.clone(), None),
                    (fragment_shader.clone(), None),
                    Some("test_graphics_pipeline"),
                )?;

                let color_attachment =
                    crate::dynamic_rendering::DynamicRenderingColorAttachment::new(
                        image_view.clone(),
                        crate::dynamic_rendering::RenderingAttachmentSetup::clear(
                            crate::clear_values::ColorClearValues::Vec4(0.0, 0.0, 0.0, 1.0),
                        ),
                        crate::dynamic_rendering::AttachmentStoreOp::Store,
                    );

                let render_extent = crate::image::Image2DDimensions::new(width, height);

                cmd_buffer.record_one_time_submit(|rec| {
                    // pipeline was created with a static viewport/scissor, so pass None here
                    rec.bind_graphics_pipeline(pipeline.clone(), None, None);
                    rec.graphics_rendering(render_extent, &[color_attachment], None, None, |rec| {
                        rec.draw(0, 3, 0, 1);
                    });
                })?;

                let fence = crate::fence::Fence::new(device.clone(), false, Some("test_fence"))?;
                let cbs: Vec<Arc<dyn crate::command_buffer::CommandBufferTrait>> =
                    vec![cmd_buffer.clone()];
                let waiter = queue.submit(cbs.as_slice(), &[], &[], fence.clone())?;
                drop(waiter);

                Ok(())
            }
            Err(err) => {
                eprintln!("Skipping test_graphics_pipeline_offscreen: {}", err);
                Ok(())
            }
        }
    }
}

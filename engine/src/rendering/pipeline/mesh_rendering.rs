use std::sync::{Arc, Mutex};

use inline_spirv::*;

use crate::rendering::{
    RenderingResult, rendering_dimensions::RenderingDimensions, resources::object::Manager,
};

use vulkan_framework::{
    clear_values::{ColorClearValues, DepthClearValues},
    command_buffer::CommandBufferRecorder,
    descriptor_pool::{
        DescriptorPool, DescriptorPoolConcreteDescriptor, DescriptorPoolSizesConcreteDescriptor,
    },
    descriptor_set::DescriptorSet,
    descriptor_set_layout::DescriptorSetLayout,
    dynamic_rendering::{
        AttachmentStoreOp, DynamicRendering, DynamicRenderingColorAttachment,
        DynamicRenderingDepthAttachment, RenderingAttachmentSetup,
    },
    graphics_pipeline::{
        AttributeType, CullMode, DepthCompareOp, DepthConfiguration, FrontFace, GraphicsPipeline,
        PolygonMode, Rasterizer, Scissor, VertexInputAttribute, VertexInputBinding,
        VertexInputRate, Viewport,
    },
    image::{
        CommonImageFormat, ConcreteImageDescriptor, Image, Image1DTrait, Image2DDimensions,
        Image2DTrait, ImageDimensions, ImageFlags, ImageFormat, ImageLayout, ImageMultisampling,
        ImageTiling, ImageUsage, ImageUseAs,
    },
    image_view::{ImageView, ImageViewAspect, ImageViewType, RecognisedImageAspect},
    memory_barriers::{ImageMemoryBarrier, MemoryAccess, MemoryAccessAs},
    memory_heap::MemoryType,
    memory_management::{MemoryManagementTagSize, MemoryManagementTags, MemoryManagerTrait},
    memory_pool::MemoryPoolFeatures,
    pipeline_layout::{PipelineLayout, PipelineLayoutDependant},
    pipeline_stage::{PipelineStage, PipelineStages},
    push_constant_range::PushConstanRange,
    queue_family::QueueFamily,
    sampler::{Filtering, MipmapMode, Sampler},
    semaphore::Semaphore,
    shader_layout_binding::{BindingDescriptor, BindingType, NativeBindingType},
    shader_stage_access::{ShaderStageAccessIn, ShaderStagesAccess},
    shaders::{fragment_shader::FragmentShader, vertex_shader::VertexShader},
};

const MESH_RENDERING_VERTEX_SPV: &[u32] = inline_spirv!(
    r#"
#version 450 core

#include "engine/shaders/mesh_rendering/mesh_rendering.vert"
"#,
    vert
);

const MESH_RENDERING_FRAGMENT_SPV: &[u32] = inline_spirv!(
    r#"
#version 450 core

#include "engine/shaders/mesh_rendering/mesh_rendering.frag"
"#,
    frag
);

/// Represents the stage of the pipeline responsible for drawing static meshes.
pub struct MeshRendering {
    image_dimensions: Image2DDimensions,

    gbuffer_descriptor_set_layout: Arc<DescriptorSetLayout>,
    gbuffer_descriptor_set: Arc<DescriptorSet>,

    push_constants_stages: ShaderStagesAccess,
    graphics_pipeline: Arc<GraphicsPipeline>,

    semaphore: Arc<Semaphore>,

    gbuffer_depth_stencil_image_view: Arc<ImageView>,
    gbuffer_position_image_view: Arc<ImageView>,
    gbuffer_normal_image_view: Arc<ImageView>,
    gbuffer_diffuse_texture_image_view: Arc<ImageView>,
    gbuffer_specular_texture_image_view: Arc<ImageView>,
    gbuffer_instance_id_image_view: Arc<ImageView>,
}

impl MeshRendering {
    #[inline(always)]
    fn output_image_color_layout() -> ImageLayout {
        ImageLayout::ColorAttachmentOptimal
    }

    #[inline(always)]
    fn output_image_depth_stencil_layout() -> ImageLayout {
        ImageLayout::DepthStencilAttachmentOptimal
    }

    #[inline(always)]
    fn output_instance_format() -> ImageFormat {
        ImageFormat::from(CommonImageFormat::r32_sfloat)
    }

    #[inline(always)]
    fn output_image_color_format() -> ImageFormat {
        ImageFormat::from(CommonImageFormat::r32g32b32a32_sfloat)
    }

    #[inline(always)]
    fn output_image_depth_stencil_format() -> ImageFormat {
        ImageFormat::from(CommonImageFormat::d32_sfloat_s8_uint)
    }

    /// Returns the descriptor set layout for the gbuffer
    #[inline(always)]
    pub fn descriptor_set_layout(&self) -> Arc<DescriptorSetLayout> {
        self.gbuffer_descriptor_set_layout.clone()
    }

    /// Returns the list of stages that has to wait on a specific semaphore
    pub fn wait_semaphores(&self) -> (PipelineStages, Arc<Semaphore>) {
        (
            [
                PipelineStage::EarlyFragmentTests,
                PipelineStage::LateFragmentTests,
                PipelineStage::FragmentShader,
            ]
            .as_slice()
            .into(),
            self.semaphore.clone(),
        )
    }

    pub fn signal_semaphores(&self) -> Arc<Semaphore> {
        self.semaphore.clone()
    }

    pub fn record_init_commands(&self, _recorder: &mut CommandBufferRecorder) {}

    pub fn new(
        memory_manager: Arc<Mutex<dyn MemoryManagerTrait>>,
        textures_descriptor_set_layout: Arc<DescriptorSetLayout>,
        materials_descriptor_set_layout: Arc<DescriptorSetLayout>,
        view_projection_descriptor_set_layout: Arc<DescriptorSetLayout>,
        render_area: &RenderingDimensions,
    ) -> RenderingResult<Self> {
        let mut mem_manager = memory_manager.lock().unwrap();

        let device = mem_manager.get_parent_device();

        let image_dimensions = render_area.into();

        let gbuffer_unallocated_handles = vec![
            Image::new(
                device.clone(),
                ConcreteImageDescriptor::new(
                    ImageDimensions::Image2D {
                        extent: image_dimensions,
                    },
                    ImageUsage::from(
                        [ImageUseAs::Sampled, ImageUseAs::DepthStencilAttachment].as_slice(),
                    ),
                    ImageMultisampling::SamplesPerPixel1,
                    1,
                    1,
                    Self::output_image_depth_stencil_format(),
                    ImageFlags::empty(),
                    ImageTiling::Optimal,
                ),
                None,
                Some("mesh_rendering.gbuffer_depth_stencil_image"),
            )?
            .into(),
            Image::new(
                device.clone(),
                ConcreteImageDescriptor::new(
                    ImageDimensions::Image2D {
                        extent: image_dimensions,
                    },
                    ImageUsage::from([ImageUseAs::Sampled, ImageUseAs::ColorAttachment].as_slice()),
                    ImageMultisampling::SamplesPerPixel1,
                    1,
                    1,
                    Self::output_image_color_format(),
                    ImageFlags::empty(),
                    ImageTiling::Optimal,
                ),
                None,
                Some("mesh_rendering.gbuffer_position_image"),
            )?
            .into(),
            Image::new(
                device.clone(),
                ConcreteImageDescriptor::new(
                    ImageDimensions::Image2D {
                        extent: image_dimensions,
                    },
                    ImageUsage::from([ImageUseAs::Sampled, ImageUseAs::ColorAttachment].as_slice()),
                    ImageMultisampling::SamplesPerPixel1,
                    1,
                    1,
                    Self::output_image_color_format(),
                    ImageFlags::empty(),
                    ImageTiling::Optimal,
                ),
                None,
                Some("mesh_rendering.gbuffer_normal_image"),
            )?
            .into(),
            Image::new(
                device.clone(),
                ConcreteImageDescriptor::new(
                    ImageDimensions::Image2D {
                        extent: image_dimensions,
                    },
                    ImageUsage::from([ImageUseAs::Sampled, ImageUseAs::ColorAttachment].as_slice()),
                    ImageMultisampling::SamplesPerPixel1,
                    1,
                    1,
                    Self::output_image_color_format(),
                    ImageFlags::empty(),
                    ImageTiling::Optimal,
                ),
                None,
                Some("mesh_rendering.gbuffer_diffuse_texture_image"),
            )?
            .into(),
            Image::new(
                device.clone(),
                ConcreteImageDescriptor::new(
                    ImageDimensions::Image2D {
                        extent: image_dimensions,
                    },
                    ImageUsage::from([ImageUseAs::Sampled, ImageUseAs::ColorAttachment].as_slice()),
                    ImageMultisampling::SamplesPerPixel1,
                    1,
                    1,
                    Self::output_image_color_format(),
                    ImageFlags::empty(),
                    ImageTiling::Optimal,
                ),
                None,
                Some("mesh_rendering.gbuffer_specular_texture_image"),
            )?
            .into(),
            Image::new(
                device.clone(),
                ConcreteImageDescriptor::new(
                    ImageDimensions::Image2D {
                        extent: image_dimensions,
                    },
                    ImageUsage::from([ImageUseAs::Sampled, ImageUseAs::ColorAttachment].as_slice()),
                    ImageMultisampling::SamplesPerPixel1,
                    1,
                    1,
                    Self::output_instance_format(),
                    ImageFlags::empty(),
                    ImageTiling::Optimal,
                ),
                None,
                Some("mesh_rendering.gbuffer_instance_image"),
            )?
            .into(),
        ];

        let allocated_handles = mem_manager.allocate_resources(
            &MemoryType::device_local(),
            &MemoryPoolFeatures::new(false),
            gbuffer_unallocated_handles,
            MemoryManagementTags::default()
                .with_name("mesh_rendering".to_string())
                .with_size(MemoryManagementTagSize::MediumSmall),
        )?;

        let gbuffer_depth_stencil_image_view = ImageView::new(
            allocated_handles[0].image(),
            Some(ImageViewType::Image2D),
            None,
            Some(ImageViewAspect::Recognised(RecognisedImageAspect::new(
                false, true, false, false,
            ))),
            None,
            None,
            None,
            None,
            None,
            Some("gbuffer_depth_stencil_image_view"),
        )?;

        let gbuffer_position_image_view = ImageView::new(
            allocated_handles[1].image(),
            Some(ImageViewType::Image2D),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            Some("gbuffer_position_image_views"),
        )?;

        let gbuffer_normal_image_view = ImageView::new(
            allocated_handles[2].image(),
            Some(ImageViewType::Image2D),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            Some("gbuffer_normal_image_view"),
        )?;

        let gbuffer_diffuse_texture_image_view = ImageView::new(
            allocated_handles[3].image(),
            Some(ImageViewType::Image2D),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            Some("gbuffer_texture_image_view"),
        )?;

        let gbuffer_specular_texture_image_view = ImageView::new(
            allocated_handles[4].image(),
            Some(ImageViewType::Image2D),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            Some("gbuffer_texture_image_view"),
        )?;

        let gbuffer_instance_id_image_view = ImageView::new(
            allocated_handles[5].image(),
            Some(ImageViewType::Image2D),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            Some("gbuffer_instance_id_view"),
        )?;

        let gbuffer_descriptor_pool = DescriptorPool::new(
            device.clone(),
            DescriptorPoolConcreteDescriptor::new(
                DescriptorPoolSizesConcreteDescriptor::new(0, 6u32, 0, 0, 0, 0, 0, 0, 0, None),
                1,
            ),
            Some("mesh_rendering.gbuffer_descriptor_pool"),
        )?;

        let gbuffer_sampler = Sampler::new(
            device.clone(),
            Filtering::Linear,
            Filtering::Linear,
            MipmapMode::ModeLinear,
            1.0,
        )?;

        let gbuffer_descriptor_set_layout = DescriptorSetLayout::new(
            device.clone(),
            [
                BindingDescriptor::new(
                    [ShaderStageAccessIn::RayTracing(vulkan_framework::shader_stage_access::ShaderStageAccessInRayTracingKHR::RayGen), ShaderStageAccessIn::Fragment].as_slice().into(),
                    BindingType::Native(NativeBindingType::CombinedImageSampler),
                    0,
                    1u32,
                ),
                BindingDescriptor::new(
                    [ShaderStageAccessIn::RayTracing(vulkan_framework::shader_stage_access::ShaderStageAccessInRayTracingKHR::RayGen), ShaderStageAccessIn::Fragment].as_slice().into(),
                    BindingType::Native(NativeBindingType::CombinedImageSampler),
                    1,
                    1u32,
                ),
                BindingDescriptor::new(
                    [ShaderStageAccessIn::RayTracing(vulkan_framework::shader_stage_access::ShaderStageAccessInRayTracingKHR::RayGen), ShaderStageAccessIn::Fragment].as_slice().into(),
                    BindingType::Native(NativeBindingType::CombinedImageSampler),
                    2,
                    4u32,
                ),
            ]
            .as_slice(),
        )?;

        let gbuffer_descriptor_set = DescriptorSet::new(
            gbuffer_descriptor_pool.clone(),
            gbuffer_descriptor_set_layout.clone(),
        )?;

        gbuffer_descriptor_set.bind_resources(|binder| {
            binder
                .bind_combined_images_samplers(
                    0,
                    [(
                        ImageLayout::ShaderReadOnlyOptimal,
                        gbuffer_depth_stencil_image_view.clone(),
                        gbuffer_sampler.clone(),
                    )]
                    .as_slice(),
                )
                .unwrap();

            binder
                .bind_combined_images_samplers(
                    1,
                    [(
                        ImageLayout::ShaderReadOnlyOptimal,
                        gbuffer_instance_id_image_view.clone(),
                        gbuffer_sampler.clone(),
                    )]
                    .as_slice(),
                )
                .unwrap();

            binder
                .bind_combined_images_samplers_with_same_layout_and_sampler(
                    2,
                    ImageLayout::ShaderReadOnlyOptimal,
                    gbuffer_sampler.clone(),
                    [
                        gbuffer_position_image_view.clone(),
                        gbuffer_normal_image_view.clone(),
                        gbuffer_diffuse_texture_image_view.clone(),
                        gbuffer_specular_texture_image_view.clone(),
                    ]
                    .as_slice(),
                )
                .unwrap();
        })?;

        let vertex_shader =
            VertexShader::new(device.clone(), MESH_RENDERING_VERTEX_SPV).unwrap();

        let fragment_shader =
            FragmentShader::new(device.clone(), MESH_RENDERING_FRAGMENT_SPV).unwrap();

        let push_constants_stages = [ShaderStageAccessIn::Vertex, ShaderStageAccessIn::Fragment]
            .as_slice()
            .into();
        let pipeline_layout = PipelineLayout::new(
            device.clone(),
            [
                textures_descriptor_set_layout,
                materials_descriptor_set_layout,
                view_projection_descriptor_set_layout,
            ]
            .as_slice(),
            [PushConstanRange::new(0, 52u32, push_constants_stages)].as_slice(),
            Some("mesh_rendering.pipeline_layout"),
        )?;

        let graphics_pipeline =
            GraphicsPipeline::new(
                None,
                DynamicRendering::new(
                    [
                        Self::output_image_color_format(),
                        Self::output_image_color_format(),
                        Self::output_image_color_format(),
                        Self::output_image_color_format(),
                        Self::output_instance_format(),
                    ]
                    .as_slice(),
                    Some(Self::output_image_depth_stencil_format()),
                    None,
                ),
                ImageMultisampling::SamplesPerPixel1,
                Some(DepthConfiguration::new(
                    true,
                    DepthCompareOp::Less,
                    Some((0.0, 1.0)),
                )),
                Some(Viewport::new(
                    0.0f32,
                    0.0f32,
                    image_dimensions.width() as f32,
                    image_dimensions.height() as f32,
                    0.0f32,
                    1.0f32,
                )),
                Some(Scissor::new(0, 0, image_dimensions)),
                pipeline_layout,
                [
                    VertexInputBinding::new(
                        VertexInputRate::PerVertex,
                        (4u32) * (3u32 + 3u32 + 2u32),
                        [
                            // vertex position data
                            VertexInputAttribute::new(0, 0, AttributeType::Vec3),
                            // vertex normal data
                            VertexInputAttribute::new(1, 4 * 3, AttributeType::Vec3),
                            // vertex text coords
                            VertexInputAttribute::new(2, 4 * (3 + 3), AttributeType::Vec2),
                        ]
                        .as_slice(),
                    ),
                    VertexInputBinding::new(
                        VertexInputRate::PerInstance,
                        // mat 4x3
                        core::mem::size_of::<
                            vulkan_framework::ash::vk::AccelerationStructureInstanceKHR,
                        >() as u32,
                        [
                            // model matrix first row
                            VertexInputAttribute::new(3, 0, AttributeType::Vec4),
                            // model matrix second row
                            VertexInputAttribute::new(4, 4 * 4, AttributeType::Vec4),
                            // model matrix third row
                            VertexInputAttribute::new(5, 4 * 8, AttributeType::Vec4),
                        ]
                        .as_slice(),
                    ),
                ]
                .as_slice(),
                Rasterizer::new(
                    PolygonMode::Fill,
                    FrontFace::Clockwise,
                    CullMode::Back,
                    None,
                ),
                (vertex_shader, None),
                (fragment_shader, None),
                Some("mesh_rendering.graphics_pipeline"),
            )?;

        let semaphore = Semaphore::new(device.clone(), Some("mesh_rendering.semaphore"))?;

        Ok(Self {
            image_dimensions,

            gbuffer_descriptor_set_layout,
            gbuffer_descriptor_set,

            push_constants_stages,
            graphics_pipeline,

            semaphore,

            gbuffer_depth_stencil_image_view,
            gbuffer_position_image_view,
            gbuffer_normal_image_view,
            gbuffer_diffuse_texture_image_view,
            gbuffer_specular_texture_image_view,
            gbuffer_instance_id_image_view,
        })
    }

    /// Record commands used to rendering to the gbuffer, that will be returned
    ///
    /// gbuffer_stages and gbuffer_access are used to build an image barrier for
    /// images on the gbuffer: the caller must tell this function how these images
    /// will be used
    pub fn record_rendering_commands<ManagerT>(
        &self,
        view_projection_descriptor_set: Arc<DescriptorSet>,
        queue_family: Arc<QueueFamily>,
        gbuffer_stages: PipelineStages,
        gbuffer_access: MemoryAccess,
        current_frame: usize,
        meshes: ManagerT,
        recorder: &mut CommandBufferRecorder,
    ) -> Arc<DescriptorSet>
    where
        ManagerT: std::ops::Deref<Target = Manager>,
    {
        let position_imageview = self.gbuffer_position_image_view.clone();
        let normal_imageview = self.gbuffer_normal_image_view.clone();
        let diffuse_texture_imageview = self.gbuffer_diffuse_texture_image_view.clone();
        let specular_texture_imageview = self.gbuffer_specular_texture_image_view.clone();
        let instance_id_imageview = self.gbuffer_instance_id_image_view.clone();
        let depth_stencil_imageview = self.gbuffer_depth_stencil_image_view.clone();

        // update materials descriptor sets (to make them relevants to this frame)
        meshes.update_buffers(recorder, current_frame, queue_family.clone());

        // Transition the framebuffer images into depth/color attachment optimal layout,
        // so that the graphics pipeline has it in the best format
        recorder.pipeline_barriers([
            ImageMemoryBarrier::new(
                PipelineStages::from([PipelineStage::TopOfPipe].as_slice()),
                MemoryAccess::from([].as_slice()),
                PipelineStages::from([PipelineStage::ColorAttachmentOutput].as_slice()),
                MemoryAccess::from([MemoryAccessAs::ColorAttachmentWrite].as_slice()),
                position_imageview.image().into(),
                ImageLayout::Undefined,
                Self::output_image_color_layout(),
                queue_family.clone(),
                queue_family.clone(),
            )
            .into(),
            ImageMemoryBarrier::new(
                PipelineStages::from([PipelineStage::TopOfPipe].as_slice()),
                MemoryAccess::from([].as_slice()),
                PipelineStages::from([PipelineStage::ColorAttachmentOutput].as_slice()),
                MemoryAccess::from([MemoryAccessAs::ColorAttachmentWrite].as_slice()),
                normal_imageview.image().into(),
                ImageLayout::Undefined,
                Self::output_image_color_layout(),
                queue_family.clone(),
                queue_family.clone(),
            )
            .into(),
            ImageMemoryBarrier::new(
                PipelineStages::from([PipelineStage::TopOfPipe].as_slice()),
                MemoryAccess::from([].as_slice()),
                PipelineStages::from([PipelineStage::ColorAttachmentOutput].as_slice()),
                MemoryAccess::from([MemoryAccessAs::ColorAttachmentWrite].as_slice()),
                diffuse_texture_imageview.image().into(),
                ImageLayout::Undefined,
                Self::output_image_color_layout(),
                queue_family.clone(),
                queue_family.clone(),
            )
            .into(),
            ImageMemoryBarrier::new(
                PipelineStages::from([PipelineStage::TopOfPipe].as_slice()),
                MemoryAccess::from([].as_slice()),
                PipelineStages::from([PipelineStage::ColorAttachmentOutput].as_slice()),
                MemoryAccess::from([MemoryAccessAs::ColorAttachmentWrite].as_slice()),
                specular_texture_imageview.image().into(),
                ImageLayout::Undefined,
                Self::output_image_color_layout(),
                queue_family.clone(),
                queue_family.clone(),
            )
            .into(),
            ImageMemoryBarrier::new(
                PipelineStages::from([PipelineStage::TopOfPipe].as_slice()),
                MemoryAccess::from([].as_slice()),
                PipelineStages::from([PipelineStage::ColorAttachmentOutput].as_slice()),
                MemoryAccess::from([MemoryAccessAs::ColorAttachmentWrite].as_slice()),
                instance_id_imageview.image().into(),
                ImageLayout::Undefined,
                Self::output_image_color_layout(),
                queue_family.clone(),
                queue_family.clone(),
            )
            .into(),
            ImageMemoryBarrier::new(
                PipelineStages::from([PipelineStage::TopOfPipe].as_slice()),
                MemoryAccess::from([].as_slice()),
                PipelineStages::from(
                    [
                        PipelineStage::EarlyFragmentTests,
                        PipelineStage::LateFragmentTests,
                    ]
                    .as_slice(),
                ),
                MemoryAccess::from(
                    [
                        MemoryAccessAs::DepthStencilAttachmentWrite,
                        MemoryAccessAs::DepthStencilAttachmentRead,
                    ]
                    .as_slice(),
                ),
                depth_stencil_imageview.image().into(),
                ImageLayout::Undefined,
                Self::output_image_depth_stencil_layout(),
                queue_family.clone(),
                queue_family.clone(),
            )
            .into(),
        ]);

        let rendering_color_attachments = [
            DynamicRenderingColorAttachment::new(
                position_imageview.clone(),
                RenderingAttachmentSetup::clear(ColorClearValues::Vec4(0.0, 0.0, 0.0, 0.0)),
                AttachmentStoreOp::Store,
            ),
            DynamicRenderingColorAttachment::new(
                normal_imageview.clone(),
                RenderingAttachmentSetup::clear(ColorClearValues::Vec4(0.0, 0.0, 0.0, 0.0)),
                AttachmentStoreOp::Store,
            ),
            DynamicRenderingColorAttachment::new(
                diffuse_texture_imageview.clone(),
                RenderingAttachmentSetup::clear(ColorClearValues::Vec4(0.0, 0.0, 0.0, 0.0)),
                AttachmentStoreOp::Store,
            ),
            DynamicRenderingColorAttachment::new(
                specular_texture_imageview.clone(),
                RenderingAttachmentSetup::clear(ColorClearValues::Vec4(0.0, 0.0, 0.0, 0.0)),
                AttachmentStoreOp::Store,
            ),
            DynamicRenderingColorAttachment::new(
                instance_id_imageview.clone(),
                RenderingAttachmentSetup::clear(ColorClearValues::UVec4(0, 0, 0, 0)),
                AttachmentStoreOp::Store,
            ),
        ];
        let rendering_depth_attachment = DynamicRenderingDepthAttachment::new(
            depth_stencil_imageview.clone(),
            RenderingAttachmentSetup::clear(DepthClearValues::new(1.0)),
            AttachmentStoreOp::Store,
        );
        recorder.graphics_rendering(
            self.image_dimensions,
            rendering_color_attachments.as_slice(),
            Some(&rendering_depth_attachment),
            None,
            |recorder| {
                recorder.bind_graphics_pipeline(self.graphics_pipeline.clone(), None, None);

                // bind the view-projection matrix
                recorder.bind_descriptor_sets_for_graphics_pipeline(
                    self.graphics_pipeline.get_parent_pipeline_layout(),
                    2,
                    [view_projection_descriptor_set].as_slice(),
                );

                // performs the actual rendering
                meshes.deref().guided_rendering(
                    recorder,
                    current_frame,
                    self.graphics_pipeline.get_parent_pipeline_layout(),
                    0,
                    1,
                    0,
                    self.push_constants_stages,
                );
            },
        );

        // gbuffers are bound to a descriptor set and will be used along the pipeline that way:
        // place image barriers to transition them to the right format
        recorder.pipeline_barriers([
            ImageMemoryBarrier::new(
                PipelineStages::from(
                    [
                        PipelineStage::EarlyFragmentTests,
                        PipelineStage::LateFragmentTests,
                    ]
                    .as_slice(),
                ),
                MemoryAccess::from(
                    [
                        MemoryAccessAs::DepthStencilAttachmentWrite,
                        MemoryAccessAs::DepthStencilAttachmentRead,
                    ]
                    .as_slice(),
                ),
                gbuffer_stages,
                gbuffer_access,
                depth_stencil_imageview.image().into(),
                Self::output_image_depth_stencil_layout(),
                ImageLayout::ShaderReadOnlyOptimal,
                queue_family.clone(),
                queue_family.clone(),
            )
            .into(),
            ImageMemoryBarrier::new(
                PipelineStages::from([PipelineStage::ColorAttachmentOutput].as_slice()),
                MemoryAccess::from([MemoryAccessAs::ColorAttachmentWrite].as_slice()),
                gbuffer_stages,
                gbuffer_access,
                position_imageview.image().into(),
                Self::output_image_color_layout(),
                ImageLayout::ShaderReadOnlyOptimal,
                queue_family.clone(),
                queue_family.clone(),
            )
            .into(),
            ImageMemoryBarrier::new(
                PipelineStages::from([PipelineStage::ColorAttachmentOutput].as_slice()),
                MemoryAccess::from([MemoryAccessAs::ColorAttachmentWrite].as_slice()),
                gbuffer_stages,
                gbuffer_access,
                normal_imageview.image().into(),
                Self::output_image_color_layout(),
                ImageLayout::ShaderReadOnlyOptimal,
                queue_family.clone(),
                queue_family.clone(),
            )
            .into(),
            ImageMemoryBarrier::new(
                PipelineStages::from([PipelineStage::ColorAttachmentOutput].as_slice()),
                MemoryAccess::from([MemoryAccessAs::ColorAttachmentWrite].as_slice()),
                gbuffer_stages,
                gbuffer_access,
                diffuse_texture_imageview.image().into(),
                Self::output_image_color_layout(),
                ImageLayout::ShaderReadOnlyOptimal,
                queue_family.clone(),
                queue_family.clone(),
            )
            .into(),
            ImageMemoryBarrier::new(
                PipelineStages::from([PipelineStage::ColorAttachmentOutput].as_slice()),
                MemoryAccess::from([MemoryAccessAs::ColorAttachmentWrite].as_slice()),
                gbuffer_stages,
                gbuffer_access,
                specular_texture_imageview.image().into(),
                Self::output_image_color_layout(),
                ImageLayout::ShaderReadOnlyOptimal,
                queue_family.clone(),
                queue_family.clone(),
            )
            .into(),
            ImageMemoryBarrier::new(
                PipelineStages::from([PipelineStage::ColorAttachmentOutput].as_slice()),
                MemoryAccess::from([MemoryAccessAs::ColorAttachmentWrite].as_slice()),
                gbuffer_stages,
                gbuffer_access,
                instance_id_imageview.image().into(),
                Self::output_image_color_layout(),
                ImageLayout::ShaderReadOnlyOptimal,
                queue_family.clone(),
                queue_family.clone(),
            )
            .into(),
        ]);

        self.gbuffer_descriptor_set.clone()
    }
}

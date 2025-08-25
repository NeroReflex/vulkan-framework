use std::{
    ops::DerefMut,
    sync::{Arc, Mutex},
};

use inline_spirv::inline_spirv;
use vulkan_framework::{
    binding_tables::RaytracingBindingTables,
    buffer::{
        AllocatedBuffer, Buffer, BufferSubresourceRange, BufferTrait, BufferUseAs,
        ConcreteBufferDescriptor,
    },
    clear_values::ColorClearValues,
    command_buffer::CommandBufferRecorder,
    compute_pipeline::ComputePipeline,
    descriptor_pool::{
        DescriptorPool, DescriptorPoolConcreteDescriptor, DescriptorPoolSizesConcreteDescriptor,
    },
    descriptor_set::DescriptorSet,
    descriptor_set_layout::DescriptorSetLayout,
    device::DeviceOwned,
    image::{
        CommonImageFormat, ConcreteImageDescriptor, Image, Image3DDimensions, ImageFlags,
        ImageFormat, ImageLayout, ImageMultisampling, ImageSubresourceRange, ImageTiling,
        ImageUseAs,
    },
    image_view::ImageView,
    memory_barriers::{BufferMemoryBarrier, ImageMemoryBarrier, MemoryAccess, MemoryAccessAs},
    memory_heap::MemoryType,
    memory_management::{MemoryManagementTagSize, MemoryManagementTags, MemoryManagerTrait},
    memory_pool::MemoryPoolFeatures,
    pipeline_layout::{PipelineLayout, PipelineLayoutDependant},
    pipeline_stage::{PipelineStage, PipelineStageRayTracingPipelineKHR, PipelineStages},
    push_constant_range::PushConstanRange,
    queue_family::QueueFamily,
    raytracing_pipeline::RaytracingPipeline,
    sampler::{Filtering, MipmapMode, Sampler},
    semaphore::Semaphore,
    shader_layout_binding::{BindingDescriptor, BindingType, NativeBindingType},
    shader_stage_access::{
        ShaderStageAccessIn, ShaderStageAccessInRayTracingKHR, ShaderStagesAccess,
    },
    shaders::{
        closest_hit_shader::ClosestHitShader, compute_shader::ComputeShader,
        miss_shader::MissShader, raygen_shader::RaygenShader,
    },
};

use crate::rendering::{RenderingResult, rendering_dimensions::RenderingDimensions};

const SURFELS_MORTON_SPV: &[u32] = inline_spirv!(
    r#"
#version 460

#include "engine/shaders/surfel_reorder/surfel_morton.comp"
"#,
    glsl,
    comp,
    vulkan1_2,
    entry = "main"
);

const SURFELS_REORDER_SPV: &[u32] = inline_spirv!(
    r#"
#version 460

#include "engine/shaders/surfel_reorder/surfel_reorder.comp"
"#,
    glsl,
    comp,
    vulkan1_2,
    entry = "main"
);

const RAYGEN_SPV: &[u32] = inline_spirv!(
    r#"
#version 460

#include "engine/shaders/global_illumination/global_illumination.rgen"
"#,
    glsl,
    rgen,
    vulkan1_2,
    entry = "main"
);

const MISS_SPV: &[u32] = inline_spirv!(
    r#"
#version 460

#include "engine/shaders/global_illumination/global_illumination.rmiss"
"#,
    glsl,
    rmiss,
    vulkan1_2,
    entry = "main"
);

const CHIT_SPV: &[u32] = inline_spirv!(
    r#"
#version 460

#include "engine/shaders/global_illumination/global_illumination.rchit"
"#,
    glsl,
    rchit,
    vulkan1_2,
    entry = "main"
);

pub struct GILighting {
    queue_family: Arc<QueueFamily>,

    raytracing_semaphore: Arc<Semaphore>,

    surfel_morton_pipeline: Arc<ComputePipeline>,
    surfel_reorder_pipeline: Arc<ComputePipeline>,

    raytracing_pipeline: Arc<RaytracingPipeline>,

    raytracing_surfel_stats_buffer: Arc<AllocatedBuffer>,
    raytracing_surfels: Arc<AllocatedBuffer>,

    raytracing_gibuffer: Arc<ImageView>,
    raytracing_dlbuffer: Arc<ImageView>,

    raytracing_sbt: Arc<RaytracingBindingTables>,

    output_descriptor_set: Arc<DescriptorSet>,

    gibuffer_descriptor_set_layout: Arc<DescriptorSetLayout>,
    gibuffer_descriptor_set: Arc<DescriptorSet>,

    renderarea_width: u32,
    renderarea_height: u32,
}

// this MUST be kept in sync with config.glsl
const SURFELS_MORTON_GROUP_SIZE_X: u32 = 256;
const SURFELS_REORDER_GROUP_SIZE_X: u32 = 256;

const MAX_SURFELS: u32 = u32::pow(2, 20) * 16;
const SURFEL_SIZE: u32 = 16 * 4;

impl GILighting {
    /// Returns the descriptor set layout for the gbuffer
    #[inline(always)]
    pub fn descriptor_set_layout(&self) -> Arc<DescriptorSetLayout> {
        self.gibuffer_descriptor_set_layout.clone()
    }

    /// Returns the list of stages that has to wait on a specific semaphore
    pub fn wait_semaphores(&self) -> (PipelineStages, Arc<Semaphore>) {
        (
            [
                //PipelineStage::Transfer,
                PipelineStage::ComputeShader,
                //PipelineStage::RayTracingPipelineKHR(
                //    PipelineStageRayTracingPipelineKHR::RayTracingShader,
                //),
            ]
            .as_slice()
            .into(),
            self.raytracing_semaphore.clone(),
        )
    }

    pub fn signal_semaphores(&self) -> Arc<Semaphore> {
        self.raytracing_semaphore.clone()
    }

    pub fn new(
        queue_family: Arc<QueueFamily>,
        render_area: &RenderingDimensions,
        memory_manager: Arc<Mutex<dyn MemoryManagerTrait>>,
        rt_descriptor_set_layout: Arc<DescriptorSetLayout>,
        gbuffer_descriptor_set_layout: Arc<DescriptorSetLayout>,
        status_descriptor_set_layout: Arc<DescriptorSetLayout>,
        textures_descriptor_set_layout: Arc<DescriptorSetLayout>,
        materials_descriptor_set_layout: Arc<DescriptorSetLayout>,
    ) -> RenderingResult<Self> {
        let device = queue_family.get_parent_device();

        let output_descriptor_set_layout = DescriptorSetLayout::new(
            device.clone(),
            [
                // surfel_stats
                BindingDescriptor::new(
                    [
                        ShaderStageAccessIn::Compute,
                        ShaderStageAccessIn::RayTracing(ShaderStageAccessInRayTracingKHR::RayGen),
                    ]
                    .as_slice()
                    .into(),
                    BindingType::Native(NativeBindingType::StorageBuffer),
                    0,
                    1,
                ),
                // surfels
                BindingDescriptor::new(
                    [
                        ShaderStageAccessIn::Compute,
                        ShaderStageAccessIn::RayTracing(ShaderStageAccessInRayTracingKHR::RayGen),
                    ]
                    .as_slice()
                    .into(),
                    BindingType::Native(NativeBindingType::StorageBuffer),
                    1,
                    1,
                ),
                // outputImages
                BindingDescriptor::new(
                    [
                        ShaderStageAccessIn::Compute,
                        ShaderStageAccessIn::RayTracing(ShaderStageAccessInRayTracingKHR::RayGen),
                    ]
                    .as_slice()
                    .into(),
                    BindingType::Native(NativeBindingType::StorageImage),
                    2,
                    2,
                ),
            ]
            .as_slice(),
        )?;

        let surfel_morton_compute_shader = ComputeShader::new(device.clone(), SURFELS_MORTON_SPV)?;
        let surfel_morton_pipeline = ComputePipeline::new(
            None,
            PipelineLayout::new(
                device.clone(),
                [
                    status_descriptor_set_layout.clone(),
                    output_descriptor_set_layout.clone(),
                ]
                .as_slice(),
                [].as_slice(),
                Some("surfel_morton_pipeline_layout"),
            )?,
            (surfel_morton_compute_shader, None),
            Some("surfel_morton_pipeline"),
        )?;

        let surfel_reorder_compute_shader =
            ComputeShader::new(device.clone(), SURFELS_REORDER_SPV)?;
        let surfel_reorder_pipeline = ComputePipeline::new(
            None,
            PipelineLayout::new(
                device.clone(),
                [
                    status_descriptor_set_layout.clone(),
                    output_descriptor_set_layout.clone(),
                ]
                .as_slice(),
                [].as_slice(),
                Some("surfel_reorder_pipeline_layout"),
            )?,
            (surfel_reorder_compute_shader, None),
            Some("surfel_reorder_pipeline"),
        )?;

        let pipeline_layout = PipelineLayout::new(
            device.clone(),
            [
                rt_descriptor_set_layout,
                gbuffer_descriptor_set_layout,
                status_descriptor_set_layout,
                textures_descriptor_set_layout,
                materials_descriptor_set_layout,
                output_descriptor_set_layout.clone(),
            ]
            .as_slice(),
            [PushConstanRange::new(
                0,
                core::mem::size_of::<u32>() as u32,
                ShaderStagesAccess::raytracing(),
            )]
            .as_slice(),
            Some("gi_lighting_pipeline_layout"),
        )?;

        let raygen_shader = RaygenShader::new(device.clone(), RAYGEN_SPV)?;
        let miss_shader = MissShader::new(device.clone(), MISS_SPV).unwrap();
        //let anyhit_shader = AnyHitShader::new(dev.clone(), AHIT_SPV).unwrap();
        let closesthit_shader = ClosestHitShader::new(device.clone(), CHIT_SPV).unwrap();
        //let callable_shader = CallableShader::new(dev.clone(), CALLABLE_SPV).unwrap();

        let raytracing_pipeline = RaytracingPipeline::new(
            pipeline_layout.clone(),
            1,
            raygen_shader,
            None,
            miss_shader,
            None,
            closesthit_shader,
            None,
            Some("gi_lighting_raytracing_pipeline!"),
        )?;

        let raytracing_buffers_unallocated = vec![
            Image::new(
                device.clone(),
                ConcreteImageDescriptor::new(
                    render_area.into(),
                    [
                        ImageUseAs::TransferDst,
                        ImageUseAs::Storage,
                        ImageUseAs::Sampled,
                    ]
                    .as_slice()
                    .into(),
                    ImageMultisampling::SamplesPerPixel1,
                    1,
                    1,
                    ImageFormat::from(CommonImageFormat::r32g32b32a32_sfloat),
                    ImageFlags::empty(),
                    ImageTiling::Optimal,
                ),
                None,
                Some(format!("raytracing_global_illumination_image").as_str()),
            )?
            .into(),
            Image::new(
                device.clone(),
                ConcreteImageDescriptor::new(
                    render_area.into(),
                    [
                        ImageUseAs::TransferDst,
                        ImageUseAs::Storage,
                        ImageUseAs::Sampled,
                    ]
                    .as_slice()
                    .into(),
                    ImageMultisampling::SamplesPerPixel1,
                    1,
                    1,
                    ImageFormat::from(CommonImageFormat::r32g32b32a32_sfloat),
                    ImageFlags::empty(),
                    ImageTiling::Optimal,
                ),
                None,
                Some(format!("raytracing_global_dlbuffer_image").as_str()),
            )?
            .into(),
            Buffer::new(
                device.clone(),
                ConcreteBufferDescriptor::new(
                    [BufferUseAs::StorageBuffer, BufferUseAs::TransferDst]
                        .as_slice()
                        .into(),
                    4u64 * 3u64,
                ),
                None,
                Some("surfel_stats"),
            )?
            .into(),
            Buffer::new(
                device.clone(),
                ConcreteBufferDescriptor::new(
                    [BufferUseAs::StorageBuffer].as_slice().into(),
                    (MAX_SURFELS as u64) * (SURFEL_SIZE as u64),
                ),
                None,
                Some("surfel_pool"),
            )?
            .into(),
        ];

        let (
            raytracing_surfel_stats_buffer,
            raytracing_surfels,
            raytracing_gibuffer,
            raytracing_dlbuffer,
            raytracing_sbt,
        ) = {
            let mut mem_manager = memory_manager.lock().unwrap();

            let raytracing_buffers_allocated = mem_manager.allocate_resources(
                &MemoryType::device_local(),
                &MemoryPoolFeatures::new(false),
                raytracing_buffers_unallocated,
                MemoryManagementTags::default()
                    .with_name("gi_lighting_gibuffer".to_string())
                    .with_size(MemoryManagementTagSize::MediumSmall),
            )?;

            let raytracing_gibuffer = ImageView::new(
                raytracing_buffers_allocated[0].image(),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                Some(format!("raytracing_global_illumination_image_imageview").as_str()),
            )?;

            let raytracing_dlbuffer = ImageView::new(
                raytracing_buffers_allocated[1].image(),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                Some(format!("raytracing_global_dlbuffer_image_imageview").as_str()),
            )?;

            let raytracing_surfel_stats_buffer = raytracing_buffers_allocated[2].buffer();

            let raytracing_surfels = raytracing_buffers_allocated[3].buffer();

            let raytracing_sbt = RaytracingBindingTables::new(
                raytracing_pipeline.clone(),
                mem_manager.deref_mut(),
                MemoryManagementTags::default()
                    .with_name("global_illumination_lighting".to_string())
                    .with_size(MemoryManagementTagSize::MediumSmall),
            )?;

            (
                raytracing_surfel_stats_buffer,
                raytracing_surfels,
                raytracing_gibuffer,
                raytracing_dlbuffer,
                raytracing_sbt,
            )
        };

        let output_descriptor_pool = DescriptorPool::new(
            device.clone(),
            DescriptorPoolConcreteDescriptor::new(
                DescriptorPoolSizesConcreteDescriptor::new(0, 0, 0, 2, 0, 0, 2, 0, 0, None),
                1,
            ),
            Some("gi_lighting_descriptor_pool"),
        )?;

        let output_descriptor_set = DescriptorSet::new(
            output_descriptor_pool.clone(),
            output_descriptor_set_layout.clone(),
        )?;

        output_descriptor_set.bind_resources(|binder| {
            binder
                .bind_storage_buffers(
                    0,
                    [(
                        raytracing_surfel_stats_buffer.clone() as Arc<dyn BufferTrait>,
                        None,
                        None,
                    )]
                    .as_slice(),
                )
                .unwrap();

            binder
                .bind_storage_buffers(
                    1,
                    [(
                        raytracing_surfels.clone() as Arc<dyn BufferTrait>,
                        None,
                        None,
                    )]
                    .as_slice(),
                )
                .unwrap();

            // bind the output images
            binder
                .bind_storage_images_with_same_layout(
                    2,
                    ImageLayout::General,
                    [raytracing_gibuffer.clone(), raytracing_dlbuffer.clone()].as_slice(),
                )
                .unwrap();
        })?;

        // this is the descriptor set that is reserved for OTHER pipelines to use the data calculated in this one
        let gibuffer_descriptor_set_layout = DescriptorSetLayout::new(
            device.clone(),
            [BindingDescriptor::new(
                [ShaderStageAccessIn::Fragment].as_slice().into(),
                BindingType::Native(NativeBindingType::CombinedImageSampler),
                0,
                2,
            )]
            .as_slice(),
        )?;

        let gibuffer_descriptor_pool = DescriptorPool::new(
            device.clone(),
            DescriptorPoolConcreteDescriptor::new(
                DescriptorPoolSizesConcreteDescriptor::new(0, 2, 0, 0, 0, 0, 0, 0, 0, None),
                1,
            ),
            Some("global_illumination_buffer_descriptor_pool"),
        )?;

        let dlbuffer_sampler = Sampler::new(
            device.clone(),
            Filtering::Nearest,
            Filtering::Nearest,
            MipmapMode::ModeNearest,
            0.0,
        )?;

        let gibuffer_descriptor_set = DescriptorSet::new(
            gibuffer_descriptor_pool.clone(),
            gibuffer_descriptor_set_layout.clone(),
        )?;

        gibuffer_descriptor_set.bind_resources(|binder| {
            binder
                .bind_combined_images_samplers_with_same_layout_and_sampler(
                    0,
                    ImageLayout::ShaderReadOnlyOptimal,
                    dlbuffer_sampler.clone(),
                    [raytracing_gibuffer.clone(), raytracing_dlbuffer.clone()].as_slice(),
                )
                .unwrap();
        })?;

        let renderarea_width = render_area.width();
        let renderarea_height = render_area.height();

        let raytracing_semaphore = Semaphore::new(device.clone(), Some("gi_lighting_semaphore"))?;

        Ok(Self {
            queue_family,

            surfel_morton_pipeline,
            surfel_reorder_pipeline,
            raytracing_pipeline,

            raytracing_surfel_stats_buffer,
            raytracing_surfels,
            raytracing_gibuffer,
            raytracing_dlbuffer,

            raytracing_sbt,

            output_descriptor_set,

            gibuffer_descriptor_set_layout,
            gibuffer_descriptor_set,

            raytracing_semaphore,

            renderarea_width,
            renderarea_height,
        })
    }

    pub fn record_init_commands(&self, recorder: &mut CommandBufferRecorder) {
        let surfel_stats_srr = BufferSubresourceRange::new(
            self.raytracing_surfel_stats_buffer.clone(),
            0u64,
            self.raytracing_surfel_stats_buffer.size(),
        );

        recorder.pipeline_barriers([BufferMemoryBarrier::new(
            [PipelineStage::TopOfPipe].as_slice().into(),
            [].as_slice().into(),
            [PipelineStage::Transfer].as_slice().into(),
            [MemoryAccessAs::TransferWrite].as_slice().into(),
            surfel_stats_srr.clone(),
            self.queue_family.clone(),
            self.queue_family.clone(),
        )
        .into()]);

        assert!(MAX_SURFELS <= i32::MAX as u32);

        let clear_val = [MAX_SURFELS as i32, 0i32, 0i32];
        recorder.update_buffer(surfel_stats_srr.buffer(), 0, &clear_val);

        recorder.pipeline_barriers([BufferMemoryBarrier::new(
            [PipelineStage::Transfer].as_slice().into(),
            [MemoryAccessAs::TransferWrite].as_slice().into(),
            [PipelineStage::BottomOfPipe].as_slice().into(),
            [MemoryAccessAs::MemoryRead].as_slice().into(),
            surfel_stats_srr.clone(),
            self.queue_family.clone(),
            self.queue_family.clone(),
        )
        .into()]);
    }

    pub fn record_rendering_commands(
        &self,
        reuse_previous_frame: u32,
        raytracing_descriptor_set: Arc<DescriptorSet>,
        gbuffer_descriptor_set: Arc<DescriptorSet>,
        status_descriptor_set: Arc<DescriptorSet>,
        textures_descriptor_set: Arc<DescriptorSet>,
        materials_descriptor_set: Arc<DescriptorSet>,
        gibuffer_stages: PipelineStages,
        gibuffer_access: MemoryAccess,
        recorder: &mut CommandBufferRecorder,
    ) -> Arc<DescriptorSet> {
        // TODO: of all word positions from GBUFFER, (order them, maybe) and for each directional light
        // use those to decide the portion that has to be rendered as depth in the shadow map

        let surfel_stats_srr = BufferSubresourceRange::new(
            self.raytracing_surfel_stats_buffer.clone(),
            0u64,
            self.raytracing_surfel_stats_buffer.size(),
        );

        let surfels_srr = BufferSubresourceRange::new(
            self.raytracing_surfels.clone(),
            0u64,
            self.raytracing_surfels.size(),
        );

        // Here clear image(s) and transition its layout for using it in the raytracing pipeline
        let gibuffer_image_srr: ImageSubresourceRange = self.raytracing_gibuffer.image().into();
        let dlbuffer_image_srr: ImageSubresourceRange = self.raytracing_dlbuffer.image().into();
        if reuse_previous_frame == 0 {
            recorder.pipeline_barriers([
                ImageMemoryBarrier::new(
                    [PipelineStage::TopOfPipe].as_slice().into(),
                    [].as_slice().into(),
                    [PipelineStage::Transfer].as_slice().into(),
                    [MemoryAccessAs::TransferWrite].as_slice().into(),
                    gibuffer_image_srr.clone(),
                    ImageLayout::Undefined,
                    ImageLayout::TransferDstOptimal,
                    self.queue_family.clone(),
                    self.queue_family.clone(),
                )
                .into(),
                ImageMemoryBarrier::new(
                    [PipelineStage::TopOfPipe].as_slice().into(),
                    [].as_slice().into(),
                    [PipelineStage::Transfer].as_slice().into(),
                    [MemoryAccessAs::TransferWrite].as_slice().into(),
                    dlbuffer_image_srr.clone(),
                    ImageLayout::Undefined,
                    ImageLayout::TransferDstOptimal,
                    self.queue_family.clone(),
                    self.queue_family.clone(),
                )
                .into(),
            ]);

            recorder.clear_color_image(
                ColorClearValues::Vec4(0.0, 0.0, 0.0, 0.0),
                gibuffer_image_srr.clone(),
            );
            recorder.clear_color_image(
                ColorClearValues::Vec4(0.0, 0.0, 0.0, 0.0),
                dlbuffer_image_srr.clone(),
            );

            recorder.pipeline_barriers([
                ImageMemoryBarrier::new(
                    [PipelineStage::Transfer].as_slice().into(),
                    [MemoryAccessAs::TransferWrite].as_slice().into(),
                    [PipelineStage::RayTracingPipelineKHR(
                        PipelineStageRayTracingPipelineKHR::RayTracingShader,
                    )]
                    .as_slice()
                    .into(),
                    [MemoryAccessAs::ShaderRead, MemoryAccessAs::ShaderWrite]
                        .as_slice()
                        .into(),
                    gibuffer_image_srr.clone(),
                    ImageLayout::TransferDstOptimal,
                    ImageLayout::General,
                    self.queue_family.clone(),
                    self.queue_family.clone(),
                )
                .into(),
                ImageMemoryBarrier::new(
                    [PipelineStage::Transfer].as_slice().into(),
                    [MemoryAccessAs::TransferWrite].as_slice().into(),
                    [PipelineStage::RayTracingPipelineKHR(
                        PipelineStageRayTracingPipelineKHR::RayTracingShader,
                    )]
                    .as_slice()
                    .into(),
                    [MemoryAccessAs::ShaderRead, MemoryAccessAs::ShaderWrite]
                        .as_slice()
                        .into(),
                    dlbuffer_image_srr.clone(),
                    ImageLayout::TransferDstOptimal,
                    ImageLayout::General,
                    self.queue_family.clone(),
                    self.queue_family.clone(),
                )
                .into(),
            ]);
        } else {
            recorder.pipeline_barriers([
                ImageMemoryBarrier::new(
                    [PipelineStage::TopOfPipe].as_slice().into(),
                    [].as_slice().into(),
                    [PipelineStage::RayTracingPipelineKHR(
                        PipelineStageRayTracingPipelineKHR::RayTracingShader,
                    )]
                    .as_slice()
                    .into(),
                    [MemoryAccessAs::ShaderRead, MemoryAccessAs::ShaderWrite]
                        .as_slice()
                        .into(),
                    gibuffer_image_srr.clone(),
                    ImageLayout::ShaderReadOnlyOptimal,
                    ImageLayout::General,
                    self.queue_family.clone(),
                    self.queue_family.clone(),
                )
                .into(),
                ImageMemoryBarrier::new(
                    [PipelineStage::TopOfPipe].as_slice().into(),
                    [].as_slice().into(),
                    [PipelineStage::RayTracingPipelineKHR(
                        PipelineStageRayTracingPipelineKHR::RayTracingShader,
                    )]
                    .as_slice()
                    .into(),
                    [MemoryAccessAs::ShaderRead, MemoryAccessAs::ShaderWrite]
                        .as_slice()
                        .into(),
                    dlbuffer_image_srr.clone(),
                    ImageLayout::ShaderReadOnlyOptimal,
                    ImageLayout::General,
                    self.queue_family.clone(),
                    self.queue_family.clone(),
                )
                .into(),
            ]);
        }

        recorder.pipeline_barriers([
            BufferMemoryBarrier::new(
                [PipelineStage::TopOfPipe].as_slice().into(),
            [].as_slice().into(),
                [PipelineStage::ComputeShader].as_slice().into(),
                [MemoryAccessAs::ShaderRead, MemoryAccessAs::ShaderWrite]
                    .as_slice()
                    .into(),
                surfel_stats_srr.clone(),
                self.queue_family.clone(),
                self.queue_family.clone(),
            )
            .into(),
            BufferMemoryBarrier::new(
                [PipelineStage::TopOfPipe].as_slice().into(),
                [].as_slice().into(),
                [PipelineStage::ComputeShader].as_slice().into(),
                [MemoryAccessAs::ShaderWrite, MemoryAccessAs::ShaderRead].as_slice().into(),
                surfels_srr.clone(),
                self.queue_family.clone(),
                self.queue_family.clone(),
            )
            .into(),
        ]);

        // this step will calculate the morton codes for each surfel and also update
        // the number of ordered_surfels.
        recorder.bind_compute_pipeline(self.surfel_morton_pipeline.clone());
        recorder.bind_descriptor_sets_for_compute_pipeline(
            self.surfel_morton_pipeline.get_parent_pipeline_layout(),
            0,
            [
                status_descriptor_set.clone(),
                self.output_descriptor_set.clone(),
            ]
            .as_slice(),
        );

        let half_size = MAX_SURFELS / 2;
        let group_count_x = (half_size / SURFELS_MORTON_GROUP_SIZE_X) + 1;
        let group_count_y = 1;
        let group_count_z = 1;
        recorder.dispatch(group_count_x, group_count_y, group_count_z);

        recorder.pipeline_barriers([
            BufferMemoryBarrier::new(
                [PipelineStage::ComputeShader].as_slice().into(),
                [MemoryAccessAs::ShaderRead, MemoryAccessAs::ShaderWrite]
                    .as_slice()
                    .into(),
                [PipelineStage::ComputeShader].as_slice().into(),
                [MemoryAccessAs::ShaderRead, MemoryAccessAs::ShaderWrite]
                    .as_slice()
                    .into(),
                surfel_stats_srr.clone(),
                self.queue_family.clone(),
                self.queue_family.clone(),
            )
            .into(),
            BufferMemoryBarrier::new(
                [PipelineStage::ComputeShader].as_slice().into(),
                [MemoryAccessAs::ShaderWrite,MemoryAccessAs::ShaderRead].as_slice().into(),
                [PipelineStage::ComputeShader].as_slice().into(),
                [MemoryAccessAs::ShaderWrite, MemoryAccessAs::ShaderRead].as_slice().into(),
                surfels_srr.clone(),
                self.queue_family.clone(),
                self.queue_family.clone(),
            )
            .into(),
        ]);

        // this step will reorder surfels by morton code and also update the number of unallocated_surfels
        recorder.bind_compute_pipeline(self.surfel_morton_pipeline.clone());
        recorder.bind_descriptor_sets_for_compute_pipeline(
            self.surfel_reorder_pipeline.get_parent_pipeline_layout(),
            0,
            [
                status_descriptor_set.clone(),
                self.output_descriptor_set.clone(),
            ]
            .as_slice(),
        );

        let half_size = MAX_SURFELS / 2;
        let group_count_x = (half_size / SURFELS_REORDER_GROUP_SIZE_X) + 1;
        let group_count_y = 1;
        let group_count_z = 1;
        recorder.dispatch(group_count_x, group_count_y, group_count_z);

        // prepare surfel(s) buffer(s) for use within the raytracing shader
        recorder.pipeline_barriers([
            BufferMemoryBarrier::new(
                [PipelineStage::ComputeShader].as_slice().into(),
                [MemoryAccessAs::ShaderRead, MemoryAccessAs::ShaderWrite]
                    .as_slice()
                    .into(),
                [PipelineStage::RayTracingPipelineKHR(
                    PipelineStageRayTracingPipelineKHR::RayTracingShader,
                )]
                .as_slice()
                .into(),
                [MemoryAccessAs::ShaderWrite, MemoryAccessAs::ShaderRead]
                    .as_slice()
                    .into(),
                surfel_stats_srr.clone(),
                self.queue_family.clone(),
                self.queue_family.clone(),
            )
            .into(),
            BufferMemoryBarrier::new(
                [PipelineStage::ComputeShader].as_slice().into(),
                [MemoryAccessAs::ShaderWrite, MemoryAccessAs::ShaderRead].as_slice().into(),
                [PipelineStage::RayTracingPipelineKHR(
                    PipelineStageRayTracingPipelineKHR::RayTracingShader,
                )]
                .as_slice()
                .into(),
                [MemoryAccessAs::ShaderWrite, MemoryAccessAs::ShaderRead]
                    .as_slice()
                    .into(),
                surfels_srr.clone(),
                self.queue_family.clone(),
                self.queue_family.clone(),
            )
            .into(),
        ]);

        recorder.bind_ray_tracing_pipeline(self.raytracing_pipeline.clone());
        recorder.bind_descriptor_sets_for_ray_tracing_pipeline(
            self.raytracing_pipeline.get_parent_pipeline_layout(),
            0,
            [
                raytracing_descriptor_set,
                gbuffer_descriptor_set,
                status_descriptor_set,
                textures_descriptor_set,
                materials_descriptor_set,
                self.output_descriptor_set.clone(),
            ]
            .as_slice(),
        );

        let push_constant: [u32; 1] = [reuse_previous_frame];

        recorder.push_constant(
            self.raytracing_pipeline.get_parent_pipeline_layout(),
            ShaderStagesAccess::raytracing(),
            0,
            unsafe {
                ::core::slice::from_raw_parts(
                    (&push_constant[0] as *const _) as *const u8,
                    core::mem::size_of_val(&push_constant),
                )
            },
        );

        recorder.trace_rays(
            self.raytracing_sbt.clone(),
            Image3DDimensions::new(self.renderarea_width, self.renderarea_height, 1),
        );

        recorder.pipeline_barriers([
            ImageMemoryBarrier::new(
                [PipelineStage::RayTracingPipelineKHR(
                    PipelineStageRayTracingPipelineKHR::RayTracingShader,
                )]
                .as_slice()
                .into(),
                [MemoryAccessAs::ShaderWrite, MemoryAccessAs::ShaderRead]
                    .as_slice()
                    .into(),
                gibuffer_stages,
                gibuffer_access,
                gibuffer_image_srr,
                ImageLayout::General,
                ImageLayout::ShaderReadOnlyOptimal,
                self.queue_family.clone(),
                self.queue_family.clone(),
            )
            .into(),
            ImageMemoryBarrier::new(
                [PipelineStage::RayTracingPipelineKHR(
                    PipelineStageRayTracingPipelineKHR::RayTracingShader,
                )]
                .as_slice()
                .into(),
                [MemoryAccessAs::ShaderWrite, MemoryAccessAs::ShaderRead]
                    .as_slice()
                    .into(),
                gibuffer_stages,
                gibuffer_access,
                dlbuffer_image_srr,
                ImageLayout::General,
                ImageLayout::ShaderReadOnlyOptimal,
                self.queue_family.clone(),
                self.queue_family.clone(),
            )
            .into(),
            BufferMemoryBarrier::new(
                [PipelineStage::RayTracingPipelineKHR(
                    PipelineStageRayTracingPipelineKHR::RayTracingShader,
                )]
                .as_slice()
                .into(),
                [MemoryAccessAs::ShaderWrite, MemoryAccessAs::ShaderRead]
                    .as_slice()
                    .into(),
                [PipelineStage::BottomOfPipe].as_slice().into(),
                [MemoryAccessAs::MemoryRead].as_slice().into(),
                surfel_stats_srr.clone(),
                self.queue_family.clone(),
                self.queue_family.clone(),
            )
            .into(),
            BufferMemoryBarrier::new(
                [PipelineStage::RayTracingPipelineKHR(
                    PipelineStageRayTracingPipelineKHR::RayTracingShader,
                )]
                .as_slice()
                .into(),
                [MemoryAccessAs::ShaderWrite, MemoryAccessAs::ShaderRead]
                    .as_slice()
                    .into(),
                [PipelineStage::BottomOfPipe].as_slice().into(),
                [MemoryAccessAs::MemoryRead].as_slice().into(),
                surfels_srr.clone(),
                self.queue_family.clone(),
                self.queue_family.clone(),
            )
            .into(),
        ]);

        self.gibuffer_descriptor_set.clone()
    }
}

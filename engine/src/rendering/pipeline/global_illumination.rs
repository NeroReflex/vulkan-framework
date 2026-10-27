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
    device::{Device, DeviceOwned},
    image::{
        CommonImageFormat, ConcreteImageDescriptor, Image, Image2DDimensions, Image3DDimensions,
        ImageAspect, ImageAspects, ImageDimensions, ImageFlags, ImageFormat, ImageLayout,
        ImageMultisampling, ImageSubresourceLayers, ImageSubresourceRange, ImageTiling, ImageTrait,
        ImageUseAs,
    },
    image_view::ImageView,
    memory_barriers::{
        BufferMemoryBarrier, ImageMemoryBarrier, MemoryAccess, MemoryAccessAs, MemoryBarrier,
    },
    memory_heap::MemoryType,
    memory_management::{
        MemoryManagementTagSize, MemoryManagementTags, MemoryManagerTrait, UnallocatedResource,
    },
    memory_pool::{MemoryMap, MemoryPoolBacked, MemoryPoolFeatures},
    pipeline_layout::{PipelineLayout, PipelineLayoutDependant},
    pipeline_stage::{PipelineStage, PipelineStageRayTracingPipelineKHR, PipelineStages},
    push_constant_range::PushConstanRange,
    query_pool::TimestampQueryPool,
    queue_family::QueueFamily,
    raytracing_pipeline::RaytracingPipeline,
    sampler::{Filtering, MipmapMode, Sampler},
    semaphore::Semaphore,
    shader_layout_binding::{BindingDescriptor, BindingType, NativeBindingType},
    shader_stage_access::{ShaderStageAccessIn, ShaderStageAccessInRayTracingKHR},
    shaders::{
        closest_hit_shader::ClosestHitShader, compute_shader::ComputeShader,
        miss_shader::MissShader, raygen_shader::RaygenShader,
    },
};

use super::gi_math::{
    FLUX_SCAN_BLOCK, GRID_BUCKETS, GRID_MAX_ITEMS, GRID_SCAN_BLOCK, LIGHT_TILES, LIGHT_TILE_SIZE,
    LIGHT_RECORD_STRIDE, MAX_SURFELS, RESERVOIR_STRIDE, STATE_WORDS, SURFEL_STRIDE,
    TILE_ENTRY_STRIDE,
};
use crate::rendering::resources::skin_asset::SkinnedAsset;
use crate::rendering::{RenderingResult, rendering_dimensions::RenderingDimensions};

// Shader sources are embedded at compile time: touch this file after editing
// an included .glsl so the macro expands again.

const SKIN_ANIMATE_CHANNELS_SPV: &[u32] = inline_spirv!(
    r#"
#version 460

#include "engine/shaders/skin/animate_channels.comp"
"#,
    glsl,
    comp,
    vulkan1_2,
    entry = "main"
);

const SKIN_ANIMATE_BIND_POSE_SPV: &[u32] = inline_spirv!(
    r#"
#version 460

#include "engine/shaders/skin/animate_bind_pose.comp"
"#,
    glsl,
    comp,
    vulkan1_2,
    entry = "main"
);

const SKIN_ANIMATE_SPV: &[u32] = inline_spirv!(
    r#"
#version 460

#include "engine/shaders/skin/animate.comp"
"#,
    glsl,
    comp,
    vulkan1_2,
    entry = "main"
);

const SKIN_DEFORM_SPV: &[u32] = inline_spirv!(
    r#"
#version 460

#include "engine/shaders/skin/deform.comp"
"#,
    glsl,
    comp,
    vulkan1_2,
    entry = "main"
);

const POOL_RESET_SPV: &[u32] = inline_spirv!(
    r#"
#version 460
#extension GL_GOOGLE_include_directive : enable
#include "engine/shaders/gi/pool_reset.comp"
"#,
    glsl,
    comp,
    vulkan1_2,
    entry = "main"
);

const FRAME_BEGIN_SPV: &[u32] = inline_spirv!(
    r#"
#version 460
#extension GL_GOOGLE_include_directive : enable
#include "engine/shaders/gi/frame_begin.comp"
"#,
    glsl,
    comp,
    vulkan1_2,
    entry = "main"
);

const MAINTAIN_SPV: &[u32] = inline_spirv!(
    r#"
#version 460
#extension GL_GOOGLE_include_directive : enable
#include "engine/shaders/gi/maintain.comp"
"#,
    glsl,
    comp,
    vulkan1_2,
    entry = "main"
);

const GRID_BUILD_SPV: &[u32] = inline_spirv!(
    r#"
#version 460
#extension GL_GOOGLE_include_directive : enable
#include "engine/shaders/gi/grid_build.comp"
"#,
    glsl,
    comp,
    vulkan1_2,
    entry = "main"
);

const GRID_SCAN_SPV: &[u32] = inline_spirv!(
    r#"
#version 460
#extension GL_GOOGLE_include_directive : enable
#include "engine/shaders/gi/grid_scan.comp"
"#,
    glsl,
    comp,
    vulkan1_2,
    entry = "main"
);

const COVERAGE_SPV: &[u32] = inline_spirv!(
    r#"
#version 460
#extension GL_GOOGLE_include_directive : enable
#include "engine/shaders/gi/surfel_coverage.comp"
"#,
    glsl,
    comp,
    vulkan1_2,
    entry = "main"
);

const PUBLISH_SPV: &[u32] = inline_spirv!(
    r#"
#version 460
#extension GL_GOOGLE_include_directive : enable
#include "engine/shaders/gi/publish.comp"
"#,
    glsl,
    comp,
    vulkan1_2,
    entry = "main"
);

const PRESAMPLE_SPV: &[u32] = inline_spirv!(
    r#"
#version 460
#extension GL_GOOGLE_include_directive : enable
#include "engine/shaders/gi/presample.comp"
"#,
    glsl,
    comp,
    vulkan1_2,
    entry = "main"
);

const RESTIR_TEMPORAL_SPV: &[u32] = inline_spirv!(
    r#"
#version 460
#extension GL_GOOGLE_include_directive : enable
#include "engine/shaders/gi/restir_temporal.comp"
"#,
    glsl,
    comp,
    vulkan1_2,
    entry = "main"
);

const RESTIR_SPATIAL_SPV: &[u32] = inline_spirv!(
    r#"
#version 460
#extension GL_GOOGLE_include_directive : enable
#include "engine/shaders/gi/restir_spatial.comp"
"#,
    glsl,
    comp,
    vulkan1_2,
    entry = "main"
);

const ACCUMULATE_SPV: &[u32] = inline_spirv!(
    r#"
#version 460
#extension GL_GOOGLE_include_directive : enable
#include "engine/shaders/gi/accumulate.comp"
"#,
    glsl,
    comp,
    vulkan1_2,
    entry = "main"
);

const SURFEL_UPDATE_RGEN_SPV: &[u32] = inline_spirv!(
    r#"
#version 460
#extension GL_GOOGLE_include_directive : enable
#include "engine/shaders/gi/surfel_update.rgen"
"#,
    glsl,
    rgen,
    vulkan1_2,
    entry = "main"
);

const RESTIR_SHADE_RGEN_SPV: &[u32] = inline_spirv!(
    r#"
#version 460
#extension GL_GOOGLE_include_directive : enable
#include "engine/shaders/gi/restir_shade.rgen"
"#,
    glsl,
    rgen,
    vulkan1_2,
    entry = "main"
);

const REFERENCE_RGEN_SPV: &[u32] = inline_spirv!(
    r#"
#version 460
#extension GL_GOOGLE_include_directive : enable
#include "engine/shaders/gi/reference.rgen"
"#,
    glsl,
    rgen,
    vulkan1_2,
    entry = "main"
);

const HIT_RMISS_SPV: &[u32] = inline_spirv!(
    r#"
#version 460
#extension GL_GOOGLE_include_directive : enable
#include "engine/shaders/gi/hit.rmiss"
"#,
    glsl,
    rmiss,
    vulkan1_2,
    entry = "main"
);

const HIT_RCHIT_SPV: &[u32] = inline_spirv!(
    r#"
#version 460
#extension GL_GOOGLE_include_directive : enable
#include "engine/shaders/gi/hit.rchit"
"#,
    glsl,
    rchit,
    vulkan1_2,
    entry = "main"
);

/// Status flags (`GI_FLAG_*` in config.glsl).
pub const GI_FLAG_MULTIBOUNCE: u32 = 1 << 0;
pub const GI_FLAG_REFERENCE: u32 = 1 << 1;
pub const GI_FLAG_TEMPORAL: u32 = 1 << 2;
pub const GI_FLAG_SPATIAL: u32 = 1 << 3;
pub const GI_FLAG_CAPTURE: u32 = 1 << 4;
pub const GI_FLAG_RESET: u32 = 1 << 5;
pub const GI_FLAG_SPAWN: u32 = 1 << 6;
pub const GI_FLAG_ACCUMULATE: u32 = 1 << 7;
pub const GI_FLAG_CONVERGE: u32 = 1 << 8;

/// Storage images of the GI set (`IMG_*` in gi/surfels.glsl).
pub const IMG_EIND: usize = 0;
pub const IMG_EDIR: usize = 1;
pub const IMG_EREF: usize = 2;
pub const IMG_EIND_RAW: usize = 3;
pub const IMG_EIND_HISTORY: usize = 4;
pub const IMG_COVERAGE: usize = 5;
pub const IMG_ALBEDO: usize = 6;
pub const GI_IMAGE_COUNT: usize = 7;
const GI_IMAGE_NAMES: [&str; GI_IMAGE_COUNT] = [
    "gi_e_ind",
    "gi_e_dir",
    "gi_e_ref",
    "gi_e_ind_raw",
    "gi_e_ind_history",
    "gi_coverage",
    "gi_albedo",
];

/// Images sampled by the final composite, in `gibuffer[]` order.
const GI_SAMPLED: [usize; 4] = [IMG_EIND, IMG_EDIR, IMG_EREF, IMG_COVERAGE];

const NODE_CAPACITY: usize = 4096;
const MAX_PALETTE_BONES: usize = 4096;
const PREVIEW_WIDTH: u32 = 640;
const PREVIEW_HEIGHT: u32 = 360;
const SKIN_CHANNELS_GROUP_SIZE: u32 = 32;
const SKIN_PALETTE_GROUP_SIZE: u32 = 64;

/// Timestamps written per frame slot (one before the first pass, one after each).
pub const GI_TIMESTAMP_LABELS: [&str; 11] = [
    "maintain",
    "grid",
    "surfel update",
    "coverage+spawn",
    "light cdf",
    "restir temporal",
    "restir spatial",
    "shade",
    "accumulate",
    "reference",
    "copies",
];
const TIMESTAMPS_PER_FRAME: u32 = GI_TIMESTAMP_LABELS.len() as u32 + 1;

/// What a readback captured, in buffer order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReadbackItem {
    EDir,
    EInd,
    ERef,
    Albedo,
    Coverage,
    Overlay,
    State,
    Surfels,
    FreeIds,
    AliveIds,
    Geo,
    Normals,
    Light,
    GridStart,
    GridItems,
    FluxCdf,
    LightTiles,
    LightRecords,
}

/// Host copy of GPU state taken at the end of a frame.
pub struct GiReadback {
    pub width: u32,
    pub height: u32,
    pub frame_index: u32,
    items: Vec<(ReadbackItem, Vec<u8>)>,
}

impl GiReadback {
    pub fn bytes(&self, item: ReadbackItem) -> &[u8] {
        self.items
            .iter()
            .find(|(i, _)| *i == item)
            .map(|(_, b)| b.as_slice())
            .unwrap_or(&[])
    }

    pub fn u32s(&self, item: ReadbackItem) -> Vec<u32> {
        self.bytes(item)
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    pub fn f32s(&self, item: ReadbackItem) -> Vec<f32> {
        self.bytes(item)
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }
}

struct ReadbackRegion {
    item: ReadbackItem,
    offset: u64,
    size: u64,
}

struct GiBuffers {
    state: Arc<AllocatedBuffer>,
    surfels: Arc<AllocatedBuffer>,
    free_ids: Arc<AllocatedBuffer>,
    alive_ids: Arc<AllocatedBuffer>,
    geo: Arc<AllocatedBuffer>,
    normals: Arc<AllocatedBuffer>,
    light: Arc<AllocatedBuffer>,
    grid_count: Arc<AllocatedBuffer>,
    grid_start: Arc<AllocatedBuffer>,
    grid_cursor: Arc<AllocatedBuffer>,
    grid_items: Arc<AllocatedBuffer>,
    flux_cdf: Arc<AllocatedBuffer>,
    scan_blocks: Arc<AllocatedBuffer>,
    light_tiles: Arc<AllocatedBuffer>,
    reservoirs: Arc<AllocatedBuffer>,
    /// Bound at LIGHT_RECORD_BINDING.
    light_records: Arc<AllocatedBuffer>,
}

impl GiBuffers {
    /// Buffers bound at bindings 0..=14 of the GI set, in binding order.
    fn bound(&self) -> [Arc<AllocatedBuffer>; 15] {
        [
            self.state.clone(),
            self.surfels.clone(),
            self.free_ids.clone(),
            self.alive_ids.clone(),
            self.geo.clone(),
            self.normals.clone(),
            self.light.clone(),
            self.grid_count.clone(),
            self.grid_start.clone(),
            self.grid_cursor.clone(),
            self.grid_items.clone(),
            self.flux_cdf.clone(),
            self.scan_blocks.clone(),
            self.light_tiles.clone(),
            self.reservoirs.clone(),
        ]
    }

    fn all(&self) -> Vec<Arc<AllocatedBuffer>> {
        let mut all = self.bound().to_vec();
        all.push(self.light_records.clone());
        all
    }
}

/// Bindings 20..20 + GI_IMAGE_COUNT hold the image array: the framework tracks
/// array elements as consecutive binding slots, so this comes after them.
const LIGHT_RECORD_BINDING: u32 = 20 + GI_IMAGE_COUNT as u32;

pub struct GILighting {
    queue_family: Arc<QueueFamily>,

    /// Timeline semaphore used to chain consecutive frames: the submission of
    /// the frame with counter N waits on the payload value N (signaled when
    /// frame N-1 terminated) before touching the global illumination data of
    /// the previous frame, and signals the payload value N+1 when its own
    /// commands are done.
    raytracing_semaphore: Arc<Semaphore>,

    pool_reset_pipeline: Arc<ComputePipeline>,
    frame_begin_pipeline: Arc<ComputePipeline>,
    maintain_pipeline: Arc<ComputePipeline>,
    grid_build_pipeline: Arc<ComputePipeline>,
    grid_scan_pipeline: Arc<ComputePipeline>,
    coverage_pipeline: Arc<ComputePipeline>,
    publish_pipeline: Arc<ComputePipeline>,
    presample_pipeline: Arc<ComputePipeline>,
    temporal_pipeline: Arc<ComputePipeline>,
    spatial_pipeline: Arc<ComputePipeline>,
    accumulate_pipeline: Arc<ComputePipeline>,
    surfel_update_pipeline: Arc<RaytracingPipeline>,
    shade_pipeline: Arc<RaytracingPipeline>,
    reference_pipeline: Arc<RaytracingPipeline>,
    surfel_update_sbt: Arc<RaytracingBindingTables>,
    shade_sbt: Arc<RaytracingBindingTables>,
    reference_sbt: Arc<RaytracingBindingTables>,

    skin_animate_channels_pipeline: Arc<ComputePipeline>,
    skin_bind_pose_pipeline: Arc<ComputePipeline>,
    skin_animate_pipeline: Arc<ComputePipeline>,
    skin_deform_pipeline: Arc<ComputePipeline>,

    buffers: GiBuffers,
    node_world: Arc<AllocatedBuffer>,
    bone_palette: Arc<AllocatedBuffer>,
    instance_generation: Arc<AllocatedBuffer>,
    instance_bone_base: Arc<AllocatedBuffer>,
    node_matrices: Mutex<Vec<[f32; 16]>>,
    bone_matrices: Mutex<Vec<[f32; 16]>>,
    instance_generations: Mutex<Vec<u32>>,
    instance_bone_bases: Mutex<Vec<u32>>,
    upload_nodes: Mutex<bool>,
    upload_bones: Mutex<bool>,
    upload_instance_meta: Mutex<bool>,

    preview_image: Arc<ImageView>,
    preview_buffer: Arc<AllocatedBuffer>,
    preview_armed: Mutex<bool>,
    preview_copies: Mutex<u32>,

    overlay: Arc<ImageView>,
    gi_images: Vec<Arc<ImageView>>,

    gi_descriptor_set: Arc<DescriptorSet>,
    gibuffer_descriptor_set_layout: Arc<DescriptorSetLayout>,
    gibuffer_descriptor_set: Arc<DescriptorSet>,

    timestamps: Option<Arc<TimestampQueryPool>>,
    timestamps_written: Mutex<Vec<bool>>,

    readback_buffer: Option<Arc<AllocatedBuffer>>,
    readback_regions: Vec<ReadbackRegion>,
    readback_requested: Mutex<bool>,
    /// Frame slot and frame index whose commands copy into the readback buffer.
    readback_pending: Mutex<Option<(usize, u32)>>,

    renderarea_width: u32,
    renderarea_height: u32,
}

fn identity_matrix() -> [f32; 16] {
    [
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ]
}

fn identity_matrices(count: usize) -> Vec<[f32; 16]> {
    vec![identity_matrix(); count]
}

fn storage_buffer(
    device: &Arc<Device>,
    size: u64,
    extra: &[BufferUseAs],
    name: &str,
) -> RenderingResult<UnallocatedResource> {
    let mut usage = vec![
        BufferUseAs::StorageBuffer,
        BufferUseAs::TransferDst,
        BufferUseAs::TransferSrc,
    ];
    usage.extend_from_slice(extra);
    Ok(Buffer::new(
        device.clone(),
        ConcreteBufferDescriptor::new(usage.as_slice().into(), size),
        None,
        Some(name),
    )?
    .into())
}

fn storage_image(
    device: &Arc<Device>,
    render_area: &RenderingDimensions,
    format: CommonImageFormat,
    name: &str,
) -> RenderingResult<UnallocatedResource> {
    Ok(Image::new(
        device.clone(),
        ConcreteImageDescriptor::new(
            render_area.into(),
            [
                ImageUseAs::TransferSrc,
                ImageUseAs::TransferDst,
                ImageUseAs::Storage,
                ImageUseAs::Sampled,
            ]
            .as_slice()
            .into(),
            ImageMultisampling::SamplesPerPixel1,
            1,
            1,
            ImageFormat::from(format),
            ImageFlags::empty(),
            ImageTiling::Optimal,
        ),
        None,
        Some(name),
    )?
    .into())
}

fn full_view(image: Arc<dyn ImageTrait>, name: &str) -> RenderingResult<Arc<ImageView>> {
    Ok(ImageView::new(
        image, None, None, None, None, None, None, None, None,
        Some(name),
    )?)
}

fn gi_stages() -> PipelineStages {
    [
        PipelineStage::ComputeShader,
        PipelineStage::RayTracingPipelineKHR(PipelineStageRayTracingPipelineKHR::RayTracingShader),
    ]
    .as_slice()
    .into()
}

fn shader_rw() -> MemoryAccess {
    [MemoryAccessAs::ShaderRead, MemoryAccessAs::ShaderWrite]
        .as_slice()
        .into()
}

impl GILighting {
    /// Returns the descriptor set layout of the images sampled by the final composite.
    #[inline(always)]
    pub fn descriptor_set_layout(&self) -> Arc<DescriptorSetLayout> {
        self.gibuffer_descriptor_set_layout.clone()
    }

    /// Returns the timeline semaphore used to reuse the global illumination
    /// data computed by the previous frame: the submission of the frame with
    /// counter N waits on payload N at the stages that consume the shared
    /// gbuffer, GI, and HDR images, then signals payload N+1 (see
    /// `Queue::submit_mixed`). Final rendering still samples the GI images.
    pub fn reuse_timeline(&self) -> Arc<Semaphore> {
        self.raytracing_semaphore.clone()
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new(
        queue_family: Arc<QueueFamily>,
        render_area: &RenderingDimensions,
        memory_manager: Arc<Mutex<dyn MemoryManagerTrait>>,
        rt_descriptor_set_layout: Arc<DescriptorSetLayout>,
        gbuffer_descriptor_set_layout: Arc<DescriptorSetLayout>,
        status_descriptor_set_layout: Arc<DescriptorSetLayout>,
        textures_descriptor_set_layout: Arc<DescriptorSetLayout>,
        materials_descriptor_set_layout: Arc<DescriptorSetLayout>,
        frames_in_flight: u32,
        enable_readback: bool,
    ) -> RenderingResult<Self> {
        let device = queue_family.get_parent_device();
        let width = render_area.width();
        let height = render_area.height();
        let pixels = (width as u64) * (height as u64);

        let gi_access = || -> vulkan_framework::shader_stage_access::ShaderStagesAccess {
            [
                ShaderStageAccessIn::Compute,
                ShaderStageAccessIn::RayTracing(ShaderStageAccessInRayTracingKHR::RayGen),
            ]
            .as_slice()
            .into()
        };

        // ---- the GI descriptor set: pool, grid, lights, reservoirs, images ----
        let mut gi_bindings: Vec<Arc<BindingDescriptor>> = (0u32..=18)
            .chain([LIGHT_RECORD_BINDING])
            .map(|binding| {
                BindingDescriptor::new(
                    gi_access(),
                    BindingType::Native(NativeBindingType::StorageBuffer),
                    binding,
                    1,
                )
            })
            .collect();
        gi_bindings.push(BindingDescriptor::new(
            gi_access(),
            BindingType::Native(NativeBindingType::StorageImage),
            19,
            1,
        ));
        gi_bindings.push(BindingDescriptor::new(
            gi_access(),
            BindingType::Native(NativeBindingType::StorageImage),
            20,
            GI_IMAGE_COUNT as u32,
        ));
        let gi_set_layout = DescriptorSetLayout::new(device.clone(), gi_bindings.as_slice())?;

        let push_range = PushConstanRange::new(0, 16, [ShaderStageAccessIn::Compute].as_slice().into());
        let compute_layout = PipelineLayout::new(
            device.clone(),
            [
                status_descriptor_set_layout.clone(),
                gbuffer_descriptor_set_layout.clone(),
                gi_set_layout.clone(),
            ]
            .as_slice(),
            [push_range].as_slice(),
            Some("gi_compute_layout"),
        )?;
        let rt_layout = PipelineLayout::new(
            device.clone(),
            [
                rt_descriptor_set_layout.clone(),
                status_descriptor_set_layout.clone(),
                gbuffer_descriptor_set_layout.clone(),
                gi_set_layout.clone(),
                textures_descriptor_set_layout.clone(),
                materials_descriptor_set_layout.clone(),
            ]
            .as_slice(),
            [].as_slice(),
            Some("gi_rt_layout"),
        )?;

        let compute = |spv: &[u32], name: &str| -> RenderingResult<Arc<ComputePipeline>> {
            Ok(ComputePipeline::new(
                None,
                compute_layout.clone(),
                (ComputeShader::new(device.clone(), spv)?, None),
                Some(name),
            )?)
        };
        let pool_reset_pipeline = compute(POOL_RESET_SPV, "gi_pool_reset")?;
        let frame_begin_pipeline = compute(FRAME_BEGIN_SPV, "gi_frame_begin")?;
        let maintain_pipeline = compute(MAINTAIN_SPV, "gi_maintain")?;
        let grid_build_pipeline = compute(GRID_BUILD_SPV, "gi_grid_build")?;
        let grid_scan_pipeline = compute(GRID_SCAN_SPV, "gi_grid_scan")?;
        let coverage_pipeline = compute(COVERAGE_SPV, "gi_coverage")?;
        let publish_pipeline = compute(PUBLISH_SPV, "gi_publish")?;
        let presample_pipeline = compute(PRESAMPLE_SPV, "gi_presample")?;
        let temporal_pipeline = compute(RESTIR_TEMPORAL_SPV, "gi_restir_temporal")?;
        let spatial_pipeline = compute(RESTIR_SPATIAL_SPV, "gi_restir_spatial")?;
        let accumulate_pipeline = compute(ACCUMULATE_SPV, "gi_accumulate")?;

        let raytracing = |rgen: &[u32], name: &str| -> RenderingResult<Arc<RaytracingPipeline>> {
            Ok(RaytracingPipeline::new(
                rt_layout.clone(),
                1,
                RaygenShader::new(device.clone(), rgen)?,
                None,
                MissShader::new(device.clone(), HIT_RMISS_SPV)?,
                None,
                ClosestHitShader::new(device.clone(), HIT_RCHIT_SPV)?,
                None,
                Some(name),
            )?)
        };
        let surfel_update_pipeline = raytracing(SURFEL_UPDATE_RGEN_SPV, "gi_surfel_update")?;
        let shade_pipeline = raytracing(RESTIR_SHADE_RGEN_SPV, "gi_restir_shade")?;
        let reference_pipeline = raytracing(REFERENCE_RGEN_SPV, "gi_reference")?;

        // ---- skinning pipelines ----
        let skin_storage_bindings = |count: u32| {
            (0..count)
                .map(|binding| {
                    BindingDescriptor::new(
                        [ShaderStageAccessIn::Compute].as_slice().into(),
                        BindingType::Native(NativeBindingType::StorageBuffer),
                        binding,
                        1,
                    )
                })
                .collect::<Vec<_>>()
        };
        let skin_pipeline = |spv: &[u32], bindings: u32, push: u32, name: &str| -> RenderingResult<Arc<ComputePipeline>> {
            let layout = DescriptorSetLayout::new(device.clone(), skin_storage_bindings(bindings).as_slice())?;
            let ranges: Vec<Arc<PushConstanRange>> = if push > 0 {
                vec![PushConstanRange::new(0, push, [ShaderStageAccessIn::Compute].as_slice().into())]
            } else {
                vec![]
            };
            Ok(ComputePipeline::new(
                None,
                PipelineLayout::new(
                    device.clone(),
                    [layout].as_slice(),
                    ranges.as_slice(),
                    Some(format!("{name}_layout").as_str()),
                )?,
                (ComputeShader::new(device.clone(), spv)?, None),
                Some(name),
            )?)
        };
        let skin_animate_channels_pipeline =
            skin_pipeline(SKIN_ANIMATE_CHANNELS_SPV, 4, 8, "skin_animate_channels_pipeline")?;
        let skin_bind_pose_pipeline =
            skin_pipeline(SKIN_ANIMATE_BIND_POSE_SPV, 3, 0, "skin_bind_pose_pipeline")?;
        let skin_animate_pipeline = skin_pipeline(SKIN_ANIMATE_SPV, 2, 0, "skin_animate_pipeline")?;
        let skin_deform_pipeline = skin_pipeline(SKIN_DEFORM_SPV, 3, 0, "skin_deform_pipeline")?;

        // ---- resources ----
        let max = MAX_SURFELS as u64;
        let mut unallocated = vec![
            storage_buffer(&device, (STATE_WORDS * 4) as u64, &[], "gi_state")?,
            storage_buffer(&device, max * SURFEL_STRIDE as u64, &[], "gi_surfels")?,
            storage_buffer(&device, max * 4, &[], "gi_free_ids")?,
            storage_buffer(&device, 2 * max * 4, &[], "gi_alive_ids")?,
            storage_buffer(&device, max * 16, &[], "gi_geo")?,
            storage_buffer(&device, max * 4, &[], "gi_normals")?,
            storage_buffer(&device, max * 16, &[], "gi_light")?,
            storage_buffer(&device, GRID_BUCKETS as u64 * 4, &[], "gi_grid_count")?,
            storage_buffer(&device, (GRID_BUCKETS as u64 + 1) * 4, &[], "gi_grid_start")?,
            storage_buffer(&device, GRID_BUCKETS as u64 * 4, &[], "gi_grid_cursor")?,
            storage_buffer(&device, GRID_MAX_ITEMS as u64 * 4, &[], "gi_grid_items")?,
            storage_buffer(&device, max * 4, &[], "gi_flux_cdf")?,
            storage_buffer(&device, 1024 * 4, &[], "gi_scan_blocks")?,
            storage_buffer(
                &device,
                (LIGHT_TILES * LIGHT_TILE_SIZE) as u64 * TILE_ENTRY_STRIDE as u64,
                &[],
                "gi_light_tiles",
            )?,
            storage_buffer(&device, 3 * pixels * RESERVOIR_STRIDE as u64, &[], "gi_reservoirs")?,
            storage_buffer(&device, max * LIGHT_RECORD_STRIDE as u64, &[], "gi_light_records")?,
            storage_image(&device, render_area, CommonImageFormat::r32_uint, "gi_surfel_overlay")?,
        ];
        for name in GI_IMAGE_NAMES {
            unallocated.push(storage_image(
                &device,
                render_area,
                CommonImageFormat::r32g32b32a32_sfloat,
                name,
            )?);
        }

        let (buffers, overlay, gi_images, surfel_update_sbt, shade_sbt, reference_sbt) = {
            let mut mem_manager = memory_manager.lock().unwrap();
            let allocated = mem_manager.allocate_resources(
                &MemoryType::device_local(),
                &MemoryPoolFeatures::new(false),
                unallocated,
                MemoryManagementTags::default()
                    .with_name("gi_lighting".to_string())
                    .with_size(MemoryManagementTagSize::MediumSmall),
            )?;
            let buffers = GiBuffers {
                state: allocated[0].buffer(),
                surfels: allocated[1].buffer(),
                free_ids: allocated[2].buffer(),
                alive_ids: allocated[3].buffer(),
                geo: allocated[4].buffer(),
                normals: allocated[5].buffer(),
                light: allocated[6].buffer(),
                grid_count: allocated[7].buffer(),
                grid_start: allocated[8].buffer(),
                grid_cursor: allocated[9].buffer(),
                grid_items: allocated[10].buffer(),
                flux_cdf: allocated[11].buffer(),
                scan_blocks: allocated[12].buffer(),
                light_tiles: allocated[13].buffer(),
                reservoirs: allocated[14].buffer(),
                light_records: allocated[15].buffer(),
            };
            let overlay = full_view(allocated[16].image(), "gi_surfel_overlay_view")?;
            let mut gi_images = Vec::with_capacity(GI_IMAGE_COUNT);
            for (index, name) in GI_IMAGE_NAMES.iter().enumerate() {
                gi_images.push(full_view(allocated[17 + index].image(), format!("{name}_view").as_str())?);
            }
            let sbt = |pipeline: &Arc<RaytracingPipeline>, mem: &mut dyn MemoryManagerTrait| {
                RaytracingBindingTables::new(
                    pipeline.clone(),
                    mem,
                    MemoryManagementTags::default()
                        .with_name("gi_lighting_sbt".to_string())
                        .with_size(MemoryManagementTagSize::Small),
                )
            };
            let surfel_update_sbt = sbt(&surfel_update_pipeline, mem_manager.deref_mut())?;
            let shade_sbt = sbt(&shade_pipeline, mem_manager.deref_mut())?;
            let reference_sbt = sbt(&reference_pipeline, mem_manager.deref_mut())?;
            (buffers, overlay, gi_images, surfel_update_sbt, shade_sbt, reference_sbt)
        };

        let (node_world, bone_palette, instance_generation, instance_bone_base, preview_image, preview_buffer) = {
            let mut mem_manager = memory_manager.lock().unwrap();
            let matrix_bytes = (NODE_CAPACITY as u64) * 64;
            let bone_bytes = (MAX_PALETTE_BONES as u64) * 64;
            let instance_meta_bytes = (NODE_CAPACITY as u64) * 4;
            let allocated = mem_manager.allocate_resources(
                &MemoryType::device_local(),
                &MemoryPoolFeatures::new(false),
                vec![
                    storage_buffer(&device, matrix_bytes, &[], "node_world")?,
                    storage_buffer(&device, bone_bytes, &[], "bone_palette")?,
                    storage_buffer(&device, instance_meta_bytes, &[], "instance_generation")?,
                    storage_buffer(&device, instance_meta_bytes, &[], "instance_bone_base")?,
                    Image::new(
                        device.clone(),
                        ConcreteImageDescriptor::new(
                            Image2DDimensions::new(PREVIEW_WIDTH, PREVIEW_HEIGHT).into(),
                            [
                                ImageUseAs::TransferSrc,
                                ImageUseAs::TransferDst,
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
                        Some("preview_image"),
                    )?
                    .into(),
                ],
                MemoryManagementTags::default()
                    .with_name("node_and_preview".to_string())
                    .with_size(MemoryManagementTagSize::MediumSmall),
            )?;
            let preview_readback = mem_manager.allocate_resources(
                &MemoryType::device_local_and_host_visible(),
                &MemoryPoolFeatures::default(),
                vec![Buffer::new(
                    device.clone(),
                    ConcreteBufferDescriptor::new(
                        [BufferUseAs::TransferDst].as_slice().into(),
                        (PREVIEW_WIDTH as u64) * (PREVIEW_HEIGHT as u64) * 16,
                    ),
                    None,
                    Some("preview_readback"),
                )?
                .into()],
                MemoryManagementTags::default()
                    .with_name("preview_readback".to_string())
                    .with_size(MemoryManagementTagSize::MediumSmall),
            )?;
            let preview_image = full_view(allocated[4].image(), "preview_image_view")?;
            (
                allocated[0].buffer(),
                allocated[1].buffer(),
                allocated[2].buffer(),
                allocated[3].buffer(),
                preview_image,
                preview_readback[0].buffer(),
            )
        };

        // ---- readback (captures and self-test only) ----
        let image_bytes = pixels * 16;
        let regions_spec: Vec<(ReadbackItem, u64)> = vec![
            (ReadbackItem::EDir, image_bytes),
            (ReadbackItem::EInd, image_bytes),
            (ReadbackItem::ERef, image_bytes),
            (ReadbackItem::Albedo, image_bytes),
            (ReadbackItem::Coverage, image_bytes),
            (ReadbackItem::Overlay, pixels * 4),
            (ReadbackItem::State, buffers.state.size()),
            (ReadbackItem::Surfels, buffers.surfels.size()),
            (ReadbackItem::FreeIds, buffers.free_ids.size()),
            (ReadbackItem::AliveIds, buffers.alive_ids.size()),
            (ReadbackItem::Geo, buffers.geo.size()),
            (ReadbackItem::Normals, buffers.normals.size()),
            (ReadbackItem::Light, buffers.light.size()),
            (ReadbackItem::GridStart, buffers.grid_start.size()),
            (ReadbackItem::GridItems, buffers.grid_items.size()),
            (ReadbackItem::FluxCdf, buffers.flux_cdf.size()),
            (ReadbackItem::LightTiles, buffers.light_tiles.size()),
            (ReadbackItem::LightRecords, buffers.light_records.size()),
        ];
        let mut readback_regions = Vec::new();
        let mut offset = 0u64;
        for (item, size) in regions_spec {
            readback_regions.push(ReadbackRegion { item, offset, size });
            offset += size.div_ceil(256) * 256;
        }
        let readback_buffer = if enable_readback {
            let mut mem_manager = memory_manager.lock().unwrap();
            let allocated = mem_manager.allocate_resources(
                &MemoryType::host_visible_and_coherent(),
                &MemoryPoolFeatures::default(),
                vec![Buffer::new(
                    device.clone(),
                    ConcreteBufferDescriptor::new(
                        [BufferUseAs::TransferDst].as_slice().into(),
                        offset,
                    ),
                    None,
                    Some("gi_readback"),
                )?
                .into()],
                MemoryManagementTags::default()
                    .with_name("gi_readback".to_string())
                    .with_size(MemoryManagementTagSize::Small),
            )?;
            Some(allocated[0].buffer())
        } else {
            None
        };

        // ---- descriptor sets ----
        let gi_descriptor_pool = DescriptorPool::new(
            device.clone(),
            DescriptorPoolConcreteDescriptor::new(
                DescriptorPoolSizesConcreteDescriptor::new(
                    0,
                    0,
                    0,
                    1 + GI_IMAGE_COUNT as u32,
                    0,
                    0,
                    20,
                    0,
                    0,
                    None,
                ),
                1,
            ),
            Some("gi_descriptor_pool"),
        )?;
        let gi_descriptor_set = DescriptorSet::new(gi_descriptor_pool, gi_set_layout.clone())?;
        {
            let mut bound: Vec<Arc<AllocatedBuffer>> = buffers.bound().to_vec();
            bound.extend([
                node_world.clone(),
                bone_palette.clone(),
                instance_generation.clone(),
                instance_bone_base.clone(),
            ]);
            let light_records = buffers.light_records.clone();
            let overlay = overlay.clone();
            let images = gi_images.clone();
            gi_descriptor_set.bind_resources(|binder| {
                for (binding, buffer) in bound.iter().enumerate() {
                    binder
                        .bind_storage_buffers(
                            binding as u32,
                            [(buffer.clone() as Arc<dyn BufferTrait>, None, None)].as_slice(),
                        )
                        .unwrap();
                }
                binder
                    .bind_storage_buffers(
                        LIGHT_RECORD_BINDING,
                        [(light_records as Arc<dyn BufferTrait>, None, None)].as_slice(),
                    )
                    .unwrap();
                binder
                    .bind_storage_images_with_same_layout(19, ImageLayout::General, [overlay].as_slice())
                    .unwrap();
                binder
                    .bind_storage_images_with_same_layout(20, ImageLayout::General, images.as_slice())
                    .unwrap();
            })?;
        }

        // Read by the final composite.
        let gibuffer_descriptor_set_layout = DescriptorSetLayout::new(
            device.clone(),
            [
                BindingDescriptor::new(
                    [ShaderStageAccessIn::Fragment].as_slice().into(),
                    BindingType::Native(NativeBindingType::CombinedImageSampler),
                    0,
                    1,
                ),
                BindingDescriptor::new(
                    [ShaderStageAccessIn::Fragment].as_slice().into(),
                    BindingType::Native(NativeBindingType::CombinedImageSampler),
                    1,
                    GI_SAMPLED.len() as u32,
                ),
            ]
            .as_slice(),
        )?;
        let gibuffer_descriptor_pool = DescriptorPool::new(
            device.clone(),
            DescriptorPoolConcreteDescriptor::new(
                DescriptorPoolSizesConcreteDescriptor::new(
                    0,
                    1 + GI_SAMPLED.len() as u32,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    None,
                ),
                1,
            ),
            Some("global_illumination_buffer_descriptor_pool"),
        )?;
        let sampler = Sampler::new(
            device.clone(),
            Filtering::Nearest,
            Filtering::Nearest,
            MipmapMode::ModeNearest,
            0.0,
        )?;
        let gibuffer_descriptor_set =
            DescriptorSet::new(gibuffer_descriptor_pool, gibuffer_descriptor_set_layout.clone())?;
        {
            let overlay = overlay.clone();
            let sampled: Vec<Arc<ImageView>> = GI_SAMPLED.iter().map(|&i| gi_images[i].clone()).collect();
            gibuffer_descriptor_set.bind_resources(|binder| {
                binder
                    .bind_combined_images_samplers_with_same_layout_and_sampler(
                        0,
                        ImageLayout::ShaderReadOnlyOptimal,
                        sampler.clone(),
                        [overlay].as_slice(),
                    )
                    .unwrap();
                binder
                    .bind_combined_images_samplers_with_same_layout_and_sampler(
                        1,
                        ImageLayout::ShaderReadOnlyOptimal,
                        sampler.clone(),
                        sampled.as_slice(),
                    )
                    .unwrap();
            })?;
        }

        let timestamps = match TimestampQueryPool::new(device.clone(), frames_in_flight.max(1) * TIMESTAMPS_PER_FRAME) {
            Ok(pool) => Some(pool),
            Err(err) => {
                eprintln!("GI timestamps disabled: {err:?}");
                None
            }
        };

        let raytracing_semaphore =
            Semaphore::new_timeline(device.clone(), 0, Some("gi_reuse_timeline"))?;

        Ok(Self {
            queue_family,
            raytracing_semaphore,
            pool_reset_pipeline,
            frame_begin_pipeline,
            maintain_pipeline,
            grid_build_pipeline,
            grid_scan_pipeline,
            coverage_pipeline,
            publish_pipeline,
            presample_pipeline,
            temporal_pipeline,
            spatial_pipeline,
            accumulate_pipeline,
            surfel_update_pipeline,
            shade_pipeline,
            reference_pipeline,
            surfel_update_sbt,
            shade_sbt,
            reference_sbt,
            skin_animate_channels_pipeline,
            skin_bind_pose_pipeline,
            skin_animate_pipeline,
            skin_deform_pipeline,
            buffers,
            node_world,
            bone_palette,
            instance_generation,
            instance_bone_base,
            node_matrices: Mutex::new(identity_matrices(NODE_CAPACITY)),
            bone_matrices: Mutex::new(identity_matrices(MAX_PALETTE_BONES)),
            instance_generations: Mutex::new(vec![0u32; NODE_CAPACITY]),
            instance_bone_bases: Mutex::new(vec![0u32; NODE_CAPACITY]),
            upload_nodes: Mutex::new(true),
            upload_bones: Mutex::new(true),
            upload_instance_meta: Mutex::new(true),
            preview_image,
            preview_buffer,
            preview_armed: Mutex::new(false),
            preview_copies: Mutex::new(0),
            overlay,
            gi_images,
            gi_descriptor_set,
            gibuffer_descriptor_set_layout,
            gibuffer_descriptor_set,
            timestamps,
            timestamps_written: Mutex::new(vec![false; frames_in_flight.max(1) as usize]),
            readback_buffer,
            readback_regions,
            readback_requested: Mutex::new(false),
            readback_pending: Mutex::new(None),
            renderarea_width: width,
            renderarea_height: height,
        })
    }

    fn all_images(&self) -> Vec<Arc<ImageView>> {
        let mut images = vec![self.overlay.clone()];
        images.extend(self.gi_images.iter().cloned());
        images
    }

    fn image_barrier(
        &self,
        view: &Arc<ImageView>,
        src_stages: PipelineStages,
        src_access: MemoryAccess,
        dst_stages: PipelineStages,
        dst_access: MemoryAccess,
        old: ImageLayout,
        new: ImageLayout,
    ) -> vulkan_framework::memory_barriers::PipelineBarrier {
        ImageMemoryBarrier::new(
            src_stages,
            src_access,
            dst_stages,
            dst_access,
            view.image().into(),
            old,
            new,
            self.queue_family.clone(),
            self.queue_family.clone(),
        )
        .into()
    }

    /// Makes every write of a GI pass visible to the next one.
    fn barrier(recorder: &mut CommandBufferRecorder) {
        recorder.pipeline_barriers([MemoryBarrier::new(
            [
                PipelineStage::ComputeShader,
                PipelineStage::RayTracingPipelineKHR(
                    PipelineStageRayTracingPipelineKHR::RayTracingShader,
                ),
                PipelineStage::Transfer,
            ]
            .as_slice()
            .into(),
            [
                MemoryAccessAs::ShaderWrite,
                MemoryAccessAs::ShaderRead,
                MemoryAccessAs::TransferWrite,
            ]
            .as_slice()
            .into(),
            gi_stages(),
            shader_rw(),
        )
        .into()]);
    }

    fn timestamp(&self, recorder: &mut CommandBufferRecorder, slot: usize, index: u32) {
        if let Some(pool) = &self.timestamps {
            recorder.write_timestamp(
                pool,
                PipelineStage::AllCommands,
                slot as u32 * TIMESTAMPS_PER_FRAME + index,
            );
        }
    }

    /// GPU time of each GI pass recorded in `slot`, in milliseconds. Call after
    /// the frame's fence has been waited.
    pub fn render_width(&self) -> u32 {
        self.renderarea_width
    }

    pub fn render_height(&self) -> u32 {
        self.renderarea_height
    }

    pub fn read_timings(&self, slot: usize) -> Option<Vec<(&'static str, f64)>> {
        let pool = self.timestamps.as_ref()?;
        let mut written = self.timestamps_written.lock().unwrap();
        if !written.get(slot).copied().unwrap_or(false) {
            return None;
        }
        let ticks = pool
            .read(slot as u32 * TIMESTAMPS_PER_FRAME, TIMESTAMPS_PER_FRAME)
            .ok()??;
        written[slot] = false;
        let period = pool.period_ns();
        Some(
            GI_TIMESTAMP_LABELS
                .iter()
                .enumerate()
                .map(|(i, label)| {
                    let delta = ticks[i + 1].wrapping_sub(ticks[i]) as f64;
                    (*label, delta * period / 1.0e6)
                })
                .collect(),
        )
    }

    fn dispatch(
        &self,
        recorder: &mut CommandBufferRecorder,
        pipeline: &Arc<ComputePipeline>,
        sets: &[Arc<DescriptorSet>; 3],
        phase: u32,
        groups: (u32, u32),
    ) {
        let layout = pipeline.get_parent_pipeline_layout();
        recorder.bind_compute_pipeline(pipeline.clone());
        recorder.bind_descriptor_sets_for_compute_pipeline(layout.clone(), 0, sets.as_slice());
        let push = [phase, 0u32, 0u32, 0u32];
        recorder.push_constant(
            layout,
            [ShaderStageAccessIn::Compute].as_slice().into(),
            0,
            unsafe { std::slice::from_raw_parts(push.as_ptr() as *const u8, 16) },
        );
        recorder.dispatch(groups.0, groups.1, 1);
    }

    fn trace(
        &self,
        recorder: &mut CommandBufferRecorder,
        pipeline: &Arc<RaytracingPipeline>,
        sbt: &Arc<RaytracingBindingTables>,
        sets: &[Arc<DescriptorSet>; 6],
        width: u32,
        height: u32,
    ) {
        recorder.bind_ray_tracing_pipeline(pipeline.clone());
        recorder.bind_descriptor_sets_for_ray_tracing_pipeline(
            pipeline.get_parent_pipeline_layout(),
            0,
            sets.as_slice(),
        );
        recorder.trace_rays(sbt.clone(), Image3DDimensions::new(width, height, 1));
    }

    pub fn record_init_commands(&self, recorder: &mut CommandBufferRecorder) {
        let buffers = self.buffers.all();
        recorder.pipeline_barriers(buffers.iter().map(|buffer| {
            BufferMemoryBarrier::new(
                [PipelineStage::TopOfPipe].as_slice().into(),
                [].as_slice().into(),
                [PipelineStage::Transfer].as_slice().into(),
                [MemoryAccessAs::TransferWrite].as_slice().into(),
                BufferSubresourceRange::new(buffer.clone(), 0, buffer.size()),
                self.queue_family.clone(),
                self.queue_family.clone(),
            )
            .into()
        }));
        // Zero is the empty state of every GI buffer; the first frame resets the pool.
        for buffer in buffers.iter() {
            recorder.fill_buffer(buffer.clone(), 0, buffer.size(), 0);
        }
        self.record_transform_upload(recorder);
        recorder.pipeline_barriers([MemoryBarrier::new(
            [PipelineStage::Transfer].as_slice().into(),
            [MemoryAccessAs::TransferWrite].as_slice().into(),
            gi_stages(),
            shader_rw(),
        )
        .into()]);
    }

    /// Requests a host copy of the GI state at the end of the next recorded frame.
    pub fn request_readback(&self) -> bool {
        if self.readback_buffer.is_none() {
            return false;
        }
        *self.readback_requested.lock().unwrap() = true;
        true
    }

    /// The readback recorded in `slot`, once its frame completed.
    pub fn take_readback(&self, slot: usize) -> Option<GiReadback> {
        let (pending_slot, frame_index) = (*self.readback_pending.lock().unwrap())?;
        if pending_slot != slot {
            return None;
        }
        *self.readback_pending.lock().unwrap() = None;
        let buffer = self.readback_buffer.as_ref()?;
        let map = MemoryMap::new(buffer.get_backing_memory_pool()).ok()?;
        let range = map
            .range::<u8>(buffer.clone() as Arc<dyn MemoryPoolBacked>)
            .ok()?;
        let all = range.as_slice();
        let items = self
            .readback_regions
            .iter()
            .map(|r| (r.item, all[r.offset as usize..(r.offset + r.size) as usize].to_vec()))
            .collect();
        Some(GiReadback {
            width: self.renderarea_width,
            height: self.renderarea_height,
            frame_index,
            items,
        })
    }

    fn record_readback(&self, recorder: &mut CommandBufferRecorder, slot: usize, frame_index: u32) {
        let requested = std::mem::take(&mut *self.readback_requested.lock().unwrap());
        let Some(dst) = self.readback_buffer.clone() else {
            return;
        };
        if !requested {
            return;
        }
        recorder.pipeline_barriers([MemoryBarrier::new(
            [
                PipelineStage::ComputeShader,
                PipelineStage::RayTracingPipelineKHR(
                    PipelineStageRayTracingPipelineKHR::RayTracingShader,
                ),
                PipelineStage::Transfer,
            ]
            .as_slice()
            .into(),
            [MemoryAccessAs::ShaderWrite, MemoryAccessAs::TransferWrite]
                .as_slice()
                .into(),
            [PipelineStage::Transfer].as_slice().into(),
            [MemoryAccessAs::TransferRead].as_slice().into(),
        )
        .into()]);
        let extent = ImageDimensions::from(Image2DDimensions::new(
            self.renderarea_width,
            self.renderarea_height,
        ));
        for region in self.readback_regions.iter() {
            let image = match region.item {
                ReadbackItem::EDir => Some(self.gi_images[IMG_EDIR].clone()),
                ReadbackItem::EInd => Some(self.gi_images[IMG_EIND].clone()),
                ReadbackItem::ERef => Some(self.gi_images[IMG_EREF].clone()),
                ReadbackItem::Albedo => Some(self.gi_images[IMG_ALBEDO].clone()),
                ReadbackItem::Coverage => Some(self.gi_images[IMG_COVERAGE].clone()),
                ReadbackItem::Overlay => Some(self.overlay.clone()),
                _ => None,
            };
            if let Some(view) = image {
                recorder.copy_image_to_buffer(
                    view.image(),
                    ImageLayout::General,
                    ImageSubresourceLayers::new(
                        ImageAspects::from([ImageAspect::Color].as_ref()),
                        0,
                        0,
                        1,
                    ),
                    dst.clone() as Arc<dyn BufferTrait>,
                    region.offset,
                    extent,
                );
                continue;
            }
            let src = match region.item {
                ReadbackItem::State => &self.buffers.state,
                ReadbackItem::Surfels => &self.buffers.surfels,
                ReadbackItem::FreeIds => &self.buffers.free_ids,
                ReadbackItem::AliveIds => &self.buffers.alive_ids,
                ReadbackItem::Geo => &self.buffers.geo,
                ReadbackItem::Normals => &self.buffers.normals,
                ReadbackItem::Light => &self.buffers.light,
                ReadbackItem::GridStart => &self.buffers.grid_start,
                ReadbackItem::GridItems => &self.buffers.grid_items,
                ReadbackItem::FluxCdf => &self.buffers.flux_cdf,
                ReadbackItem::LightTiles => &self.buffers.light_tiles,
                ReadbackItem::LightRecords => &self.buffers.light_records,
                _ => unreachable!(),
            };
            recorder.copy_buffer(
                src.clone() as Arc<dyn BufferTrait>,
                dst.clone() as Arc<dyn BufferTrait>,
                [(0u64, region.offset, region.size)].as_slice(),
            );
        }
        recorder.pipeline_barriers([BufferMemoryBarrier::new(
            [PipelineStage::Transfer].as_slice().into(),
            [MemoryAccessAs::TransferWrite].as_slice().into(),
            [PipelineStage::Host].as_slice().into(),
            [MemoryAccessAs::HostRead].as_slice().into(),
            BufferSubresourceRange::new(dst.clone(), 0, dst.size()),
            self.queue_family.clone(),
            self.queue_family.clone(),
        )
        .into()]);
        *self.readback_pending.lock().unwrap() = Some((slot, frame_index));
    }

    /// Records the whole GI frame. `gi_flags` must match the flags written in
    /// the status buffer; `frame_slot` selects the per-frame timestamp range.
    #[allow(clippy::too_many_arguments)]
    pub fn record_rendering_commands(
        &self,
        reuse_previous_frame: u32,
        gi_flags: u32,
        frame_index: u32,
        frame_slot: usize,
        raytracing_descriptor_set: Arc<DescriptorSet>,
        gbuffer_descriptor_set: Arc<DescriptorSet>,
        status_descriptor_set: Arc<DescriptorSet>,
        textures_descriptor_set: Arc<DescriptorSet>,
        materials_descriptor_set: Arc<DescriptorSet>,
        gibuffer_stages: PipelineStages,
        gibuffer_access: MemoryAccess,
        recorder: &mut CommandBufferRecorder,
    ) -> Arc<DescriptorSet> {
        let width = self.renderarea_width;
        let height = self.renderarea_height;
        let images = self.all_images();
        let transfer_write: MemoryAccess = [MemoryAccessAs::TransferWrite].as_slice().into();
        let transfer: PipelineStages = [PipelineStage::Transfer].as_slice().into();

        // ---- image layouts: cleared on a history reset, otherwise kept ----
        if reuse_previous_frame == 0 {
            recorder.pipeline_barriers(images.iter().map(|view| {
                self.image_barrier(
                    view,
                    [PipelineStage::TopOfPipe].as_slice().into(),
                    [].as_slice().into(),
                    transfer,
                    transfer_write,
                    ImageLayout::Undefined,
                    ImageLayout::TransferDstOptimal,
                )
            }));
            recorder.clear_color_image(
                ColorClearValues::UVec4(u32::MAX, u32::MAX, u32::MAX, u32::MAX),
                self.overlay.image().into(),
            );
            for view in self.gi_images.iter() {
                recorder.clear_color_image(ColorClearValues::Vec4(0.0, 0.0, 0.0, 0.0), view.image().into());
            }
            recorder.pipeline_barriers(images.iter().map(|view| {
                self.image_barrier(
                    view,
                    transfer,
                    transfer_write,
                    gi_stages(),
                    shader_rw(),
                    ImageLayout::TransferDstOptimal,
                    ImageLayout::General,
                )
            }));
        } else {
            recorder.pipeline_barriers(images.iter().map(|view| {
                self.image_barrier(
                    view,
                    [PipelineStage::FragmentShader].as_slice().into(),
                    [MemoryAccessAs::ShaderRead].as_slice().into(),
                    gi_stages(),
                    shader_rw(),
                    ImageLayout::ShaderReadOnlyOptimal,
                    ImageLayout::General,
                )
            }));
        }

        // G-buffer, lights and status writes of this frame.
        recorder.pipeline_barriers([MemoryBarrier::new(
            [PipelineStage::AllCommands].as_slice().into(),
            [MemoryAccessAs::MemoryWrite].as_slice().into(),
            gi_stages(),
            shader_rw(),
        )
        .into()]);
        self.record_transform_upload(recorder);

        if let Some(pool) = &self.timestamps {
            recorder.reset_query_pool(pool, frame_slot as u32 * TIMESTAMPS_PER_FRAME, TIMESTAMPS_PER_FRAME);
            self.timestamps_written.lock().unwrap()[frame_slot] = true;
        }
        self.timestamp(recorder, frame_slot, 0);

        let csets = [
            status_descriptor_set.clone(),
            gbuffer_descriptor_set.clone(),
            self.gi_descriptor_set.clone(),
        ];
        let rsets = [
            raytracing_descriptor_set,
            status_descriptor_set,
            gbuffer_descriptor_set,
            self.gi_descriptor_set.clone(),
            textures_descriptor_set,
            materials_descriptor_set,
        ];
        let surfel_groups = MAX_SURFELS / 256;
        let tiles8 = (width.div_ceil(8), height.div_ceil(8));

        // ---- pool maintenance ----
        if reuse_previous_frame == 0 || gi_flags & GI_FLAG_RESET != 0 {
            self.dispatch(recorder, &self.pool_reset_pipeline, &csets, 0, (surfel_groups, 1));
            Self::barrier(recorder);
        }
        self.dispatch(recorder, &self.frame_begin_pipeline, &csets, 0, (1, 1));
        Self::barrier(recorder);
        self.dispatch(recorder, &self.maintain_pipeline, &csets, 0, (surfel_groups, 1));
        Self::barrier(recorder);
        self.timestamp(recorder, frame_slot, 1);

        // ---- hashed grid ----
        let scan_blocks = GRID_BUCKETS / GRID_SCAN_BLOCK;
        self.dispatch(recorder, &self.grid_build_pipeline, &csets, 0, (GRID_BUCKETS / 256, 1));
        Self::barrier(recorder);
        self.dispatch(recorder, &self.grid_build_pipeline, &csets, 1, (surfel_groups, 1));
        Self::barrier(recorder);
        for phase in 0..3 {
            let groups = if phase == 1 { 1 } else { scan_blocks };
            self.dispatch(recorder, &self.grid_scan_pipeline, &csets, phase, (groups, 1));
            Self::barrier(recorder);
        }
        self.dispatch(recorder, &self.grid_build_pipeline, &csets, 2, (surfel_groups, 1));
        Self::barrier(recorder);
        self.timestamp(recorder, frame_slot, 2);

        // ---- surfel lighting ----
        self.trace(recorder, &self.surfel_update_pipeline, &self.surfel_update_sbt, &rsets, MAX_SURFELS, 1);
        Self::barrier(recorder);
        self.timestamp(recorder, frame_slot, 3);

        self.dispatch(
            recorder,
            &self.coverage_pipeline,
            &csets,
            0,
            (width.div_ceil(16), height.div_ceil(16)),
        );
        Self::barrier(recorder);
        self.timestamp(recorder, frame_slot, 4);

        // ---- light sampling ----
        let flux_blocks = MAX_SURFELS / FLUX_SCAN_BLOCK;
        for phase in 0..3 {
            let groups = if phase == 1 { 1 } else { flux_blocks };
            self.dispatch(recorder, &self.publish_pipeline, &csets, phase, (groups, 1));
            Self::barrier(recorder);
        }
        self.dispatch(
            recorder,
            &self.presample_pipeline,
            &csets,
            0,
            ((LIGHT_TILES * LIGHT_TILE_SIZE) / 256, 1),
        );
        Self::barrier(recorder);
        self.timestamp(recorder, frame_slot, 5);

        // ---- ReSTIR ----
        self.dispatch(recorder, &self.temporal_pipeline, &csets, 0, tiles8);
        Self::barrier(recorder);
        self.timestamp(recorder, frame_slot, 6);
        self.dispatch(recorder, &self.spatial_pipeline, &csets, 0, tiles8);
        Self::barrier(recorder);
        self.timestamp(recorder, frame_slot, 7);
        self.trace(recorder, &self.shade_pipeline, &self.shade_sbt, &rsets, width, height);
        Self::barrier(recorder);
        self.timestamp(recorder, frame_slot, 8);
        self.dispatch(recorder, &self.accumulate_pipeline, &csets, 0, tiles8);
        Self::barrier(recorder);
        self.timestamp(recorder, frame_slot, 9);

        if gi_flags & GI_FLAG_REFERENCE != 0 {
            self.trace(recorder, &self.reference_pipeline, &self.reference_sbt, &rsets, width, height);
            Self::barrier(recorder);
        }
        self.timestamp(recorder, frame_slot, 10);

        // ---- history copy, preview, readback ----
        recorder.pipeline_barriers([MemoryBarrier::new(
            gi_stages(),
            shader_rw(),
            transfer,
            [MemoryAccessAs::TransferRead, MemoryAccessAs::TransferWrite]
                .as_slice()
                .into(),
        )
        .into()]);
        let color_layers = ImageSubresourceLayers::new(ImageAspects::from([ImageAspect::Color].as_ref()), 0, 0, 1);
        recorder.copy_image(
            ImageLayout::General,
            color_layers.clone(),
            self.gi_images[IMG_EIND].image(),
            ImageLayout::General,
            color_layers,
            self.gi_images[IMG_EIND_HISTORY].image(),
            ImageDimensions::from(Image2DDimensions::new(width, height)),
        );
        self.record_preview(recorder);
        self.record_readback(recorder, frame_slot, frame_index);
        self.timestamp(recorder, frame_slot, 11);

        recorder.pipeline_barriers(images.iter().map(|view| {
            self.image_barrier(
                view,
                [
                    PipelineStage::ComputeShader,
                    PipelineStage::RayTracingPipelineKHR(
                        PipelineStageRayTracingPipelineKHR::RayTracingShader,
                    ),
                    PipelineStage::Transfer,
                ]
                .as_slice()
                .into(),
                [
                    MemoryAccessAs::ShaderWrite,
                    MemoryAccessAs::ShaderRead,
                    MemoryAccessAs::TransferWrite,
                    MemoryAccessAs::TransferRead,
                ]
                .as_slice()
                .into(),
                gibuffer_stages,
                gibuffer_access,
                ImageLayout::General,
                ImageLayout::ShaderReadOnlyOptimal,
            )
        }));

        self.gibuffer_descriptor_set.clone()
    }

    fn upload_mat4s(
        recorder: &mut CommandBufferRecorder,
        buffer: Arc<AllocatedBuffer>,
        matrices: &[[f32; 16]],
    ) {
        const CHUNK: usize = 1024;
        for (index, chunk) in matrices.chunks(CHUNK).enumerate() {
            recorder.update_buffer(
                buffer.clone() as Arc<dyn BufferTrait>,
                (index * CHUNK * 64) as u64,
                chunk,
            );
        }
    }

    fn upload_u32s(
        recorder: &mut CommandBufferRecorder,
        buffer: Arc<AllocatedBuffer>,
        words: &[u32],
    ) {
        const CHUNK: usize = 4096;
        for (index, chunk) in words.chunks(CHUNK).enumerate() {
            recorder.update_buffer(
                buffer.clone() as Arc<dyn BufferTrait>,
                (index * CHUNK * 4) as u64,
                chunk,
            );
        }
    }

    fn record_transform_upload(&self, recorder: &mut CommandBufferRecorder) {
        let upload_nodes = {
            let mut flag = self.upload_nodes.lock().unwrap();
            let upload = *flag;
            *flag = false;
            upload
        };
        let upload_bones = {
            let mut flag = self.upload_bones.lock().unwrap();
            let upload = *flag;
            *flag = false;
            upload
        };
        let upload_instance_meta = {
            let mut flag = self.upload_instance_meta.lock().unwrap();
            let upload = *flag;
            *flag = false;
            upload
        };
        if upload_nodes {
            let matrices = self.node_matrices.lock().unwrap().clone();
            Self::upload_mat4s(recorder, self.node_world.clone(), &matrices);
        }
        if upload_bones {
            let matrices = self.bone_matrices.lock().unwrap().clone();
            Self::upload_mat4s(recorder, self.bone_palette.clone(), &matrices);
        }
        if upload_instance_meta {
            let generations = self.instance_generations.lock().unwrap().clone();
            let bone_bases = self.instance_bone_bases.lock().unwrap().clone();
            Self::upload_u32s(recorder, self.instance_generation.clone(), &generations);
            Self::upload_u32s(recorder, self.instance_bone_base.clone(), &bone_bases);
        }
        if upload_nodes || upload_bones || upload_instance_meta {
            let mut barriers = Vec::new();
            if upload_nodes {
                barriers.push(
                    BufferMemoryBarrier::new(
                        [PipelineStage::Transfer].as_slice().into(),
                        [MemoryAccessAs::TransferWrite].as_slice().into(),
                        gi_stages(),
                        [MemoryAccessAs::ShaderRead].as_slice().into(),
                        BufferSubresourceRange::new(
                            self.node_world.clone(),
                            0,
                            self.node_world.size(),
                        ),
                        self.queue_family.clone(),
                        self.queue_family.clone(),
                    )
                    .into(),
                );
            }
            if upload_bones {
                barriers.push(
                    BufferMemoryBarrier::new(
                        [PipelineStage::Transfer].as_slice().into(),
                        [MemoryAccessAs::TransferWrite].as_slice().into(),
                        gi_stages(),
                        [MemoryAccessAs::ShaderRead].as_slice().into(),
                        BufferSubresourceRange::new(
                            self.bone_palette.clone(),
                            0,
                            self.bone_palette.size(),
                        ),
                        self.queue_family.clone(),
                        self.queue_family.clone(),
                    )
                    .into(),
                );
            }
            if upload_instance_meta {
                barriers.push(
                    BufferMemoryBarrier::new(
                        [PipelineStage::Transfer].as_slice().into(),
                        [MemoryAccessAs::TransferWrite].as_slice().into(),
                        gi_stages(),
                        [MemoryAccessAs::ShaderRead].as_slice().into(),
                        BufferSubresourceRange::new(
                            self.instance_generation.clone(),
                            0,
                            self.instance_generation.size(),
                        ),
                        self.queue_family.clone(),
                        self.queue_family.clone(),
                    )
                    .into(),
                );
                barriers.push(
                    BufferMemoryBarrier::new(
                        [PipelineStage::Transfer].as_slice().into(),
                        [MemoryAccessAs::TransferWrite].as_slice().into(),
                        gi_stages(),
                        [MemoryAccessAs::ShaderRead].as_slice().into(),
                        BufferSubresourceRange::new(
                            self.instance_bone_base.clone(),
                            0,
                            self.instance_bone_base.size(),
                        ),
                        self.queue_family.clone(),
                        self.queue_family.clone(),
                    )
                    .into(),
                );
            }
            recorder.pipeline_barriers(barriers);
        }
    }

    pub fn dispatch_skinned_asset(
        &self,
        recorder: &mut CommandBufferRecorder,
        asset: &mut SkinnedAsset,
    ) -> RenderingResult<()> {
        asset.bind_palette(self.bone_palette.clone())?;
        let view = asset.dispatch_view();
        if view.bone_count == 0 {
            return Ok(());
        }
        let bone_groups =
            (view.bone_count + SKIN_CHANNELS_GROUP_SIZE - 1) / SKIN_CHANNELS_GROUP_SIZE;

        if view.animated {
            let Some(channels_set) = view.channels_set else {
                return Ok(());
            };
            recorder.bind_compute_pipeline(self.skin_animate_channels_pipeline.clone());
            recorder.bind_descriptor_sets_for_compute_pipeline(
                self.skin_animate_channels_pipeline.get_parent_pipeline_layout(),
                0,
                [channels_set].as_slice(),
            );
            let mut push = [0u8; 8];
            push[0..4].copy_from_slice(&view.time_ticks.to_le_bytes());
            push[4..8].copy_from_slice(&view.channel_count.to_le_bytes());
            recorder.push_constant(
                self.skin_animate_channels_pipeline.get_parent_pipeline_layout(),
                [ShaderStageAccessIn::Compute].as_slice().into(),
                0,
                &push,
            );
            recorder.dispatch(bone_groups.max(1), 1, 1);
        } else {
            recorder.bind_compute_pipeline(self.skin_bind_pose_pipeline.clone());
            recorder.bind_descriptor_sets_for_compute_pipeline(
                self.skin_bind_pose_pipeline.get_parent_pipeline_layout(),
                0,
                [view.bind_pose_set].as_slice(),
            );
            recorder.dispatch(bone_groups.max(1), 1, 1);
        }

        recorder.pipeline_barriers([BufferMemoryBarrier::new(
            [PipelineStage::ComputeShader].as_slice().into(),
            [MemoryAccessAs::ShaderWrite].as_slice().into(),
            [PipelineStage::ComputeShader].as_slice().into(),
            [MemoryAccessAs::ShaderRead].as_slice().into(),
            BufferSubresourceRange::new(view.per_frame.clone(), 0, view.per_frame.size()),
            self.queue_family.clone(),
            self.queue_family.clone(),
        )
        .into()]);

        recorder.bind_compute_pipeline(self.skin_animate_pipeline.clone());
        recorder.bind_descriptor_sets_for_compute_pipeline(
            self.skin_animate_pipeline.get_parent_pipeline_layout(),
            0,
            [view.palette_set].as_slice(),
        );
        let palette_groups =
            (view.bone_count + SKIN_PALETTE_GROUP_SIZE - 1) / SKIN_PALETTE_GROUP_SIZE;
        recorder.dispatch(palette_groups.max(1), 1, 1);

        recorder.pipeline_barriers([
            BufferMemoryBarrier::new(
                [PipelineStage::ComputeShader].as_slice().into(),
                [MemoryAccessAs::ShaderWrite].as_slice().into(),
                [PipelineStage::ComputeShader].as_slice().into(),
                [MemoryAccessAs::ShaderRead].as_slice().into(),
                BufferSubresourceRange::new(self.bone_palette.clone(), 0, self.bone_palette.size()),
                self.queue_family.clone(),
                self.queue_family.clone(),
            )
            .into(),
        ]);

        recorder.bind_compute_pipeline(self.skin_deform_pipeline.clone());
        recorder.bind_descriptor_sets_for_compute_pipeline(
            self.skin_deform_pipeline.get_parent_pipeline_layout(),
            0,
            [view.deform_set].as_slice(),
        );
        let deform_groups =
            (view.vertex_count + SKIN_PALETTE_GROUP_SIZE - 1) / SKIN_PALETTE_GROUP_SIZE;
        recorder.dispatch(deform_groups.max(1), 1, 1);

        Ok(())
    }

    pub fn set_node_matrices(&self, matrices: &[[f32; 16]]) {
        let mut stored = self.node_matrices.lock().unwrap();
        for (slot, matrix) in matrices.iter().take(NODE_CAPACITY).enumerate() {
            stored[slot] = *matrix;
        }
        *self.upload_nodes.lock().unwrap() = true;
    }

    pub fn set_instance_generations(&self, generations: &[u32]) {
        let mut stored = self.instance_generations.lock().unwrap();
        stored.resize(NODE_CAPACITY, 0);
        for (slot, generation) in generations.iter().take(NODE_CAPACITY).enumerate() {
            stored[slot] = *generation;
        }
        *self.upload_instance_meta.lock().unwrap() = true;
    }

    pub fn set_bone_palette(&self, matrices: &[[f32; 16]]) {
        let mut stored = self.bone_matrices.lock().unwrap();
        for (slot, matrix) in matrices.iter().take(MAX_PALETTE_BONES).enumerate() {
            stored[slot] = *matrix;
        }
        *self.upload_bones.lock().unwrap() = true;
    }

    pub fn mark_surfel_index_dirty(&self) {
        *self.upload_instance_meta.lock().unwrap() = true;
    }

    pub fn arm_preview(&self) {
        *self.preview_armed.lock().unwrap() = true;
    }

    pub fn preview_copies(&self) -> u32 {
        *self.preview_copies.lock().unwrap()
    }

    pub fn read_preview_rgba(&self) -> Option<Vec<u8>> {
        let map = MemoryMap::new(self.preview_buffer.get_backing_memory_pool()).ok()?;
        let range = map
            .range::<u8>(self.preview_buffer.clone() as Arc<dyn MemoryPoolBacked>)
            .ok()?;
        Some(range.as_slice().to_vec())
    }

    fn record_preview(&self, recorder: &mut CommandBufferRecorder) {
        let armed = {
            let mut flag = self.preview_armed.lock().unwrap();
            let armed = *flag;
            *flag = false;
            armed
        };
        if !armed {
            return;
        }
        let src = self.gi_images[IMG_EIND].image();
        let dst = self.preview_image.image();
        let src_range = ImageSubresourceRange::from(src.clone() as Arc<dyn vulkan_framework::image::ImageTrait>);
        let dst_range = ImageSubresourceRange::from(dst.clone() as Arc<dyn vulkan_framework::image::ImageTrait>);
        recorder.pipeline_barriers([
            ImageMemoryBarrier::new(
                [
                    PipelineStage::ComputeShader,
                    PipelineStage::RayTracingPipelineKHR(
                        PipelineStageRayTracingPipelineKHR::RayTracingShader,
                    ),
                ]
                .as_slice()
                .into(),
                [MemoryAccessAs::ShaderWrite, MemoryAccessAs::ShaderRead]
                    .as_slice()
                    .into(),
                [PipelineStage::Transfer].as_slice().into(),
                [MemoryAccessAs::TransferRead].as_slice().into(),
                src_range,
                ImageLayout::General,
                ImageLayout::General,
                self.queue_family.clone(),
                self.queue_family.clone(),
            )
            .into(),
            ImageMemoryBarrier::new(
                [PipelineStage::TopOfPipe].as_slice().into(),
                [].as_slice().into(),
                [PipelineStage::Transfer].as_slice().into(),
                [MemoryAccessAs::TransferWrite].as_slice().into(),
                dst_range.clone(),
                ImageLayout::Undefined,
                ImageLayout::TransferDstOptimal,
                self.queue_family.clone(),
                self.queue_family.clone(),
            )
            .into(),
        ]);
        recorder.blit_image(
            src.clone(),
            ImageLayout::General,
            ImageDimensions::from(Image2DDimensions::new(
                self.renderarea_width,
                self.renderarea_height,
            )),
            dst.clone(),
            ImageLayout::TransferDstOptimal,
            ImageDimensions::from(Image2DDimensions::new(PREVIEW_WIDTH, PREVIEW_HEIGHT)),
        );
        recorder.pipeline_barriers([ImageMemoryBarrier::new(
            [PipelineStage::Transfer].as_slice().into(),
            [MemoryAccessAs::TransferWrite].as_slice().into(),
            [PipelineStage::Transfer].as_slice().into(),
            [MemoryAccessAs::TransferRead].as_slice().into(),
            dst_range,
            ImageLayout::TransferDstOptimal,
            ImageLayout::TransferSrcOptimal,
            self.queue_family.clone(),
            self.queue_family.clone(),
        )
        .into()]);
        recorder.copy_image_to_buffer(
            dst,
            ImageLayout::TransferSrcOptimal,
            ImageSubresourceLayers::new(
                ImageAspects::from([ImageAspect::Color].as_ref()),
                0,
                0,
                1,
            ),
            self.preview_buffer.clone() as Arc<dyn BufferTrait>,
            0,
            ImageDimensions::from(Image2DDimensions::new(PREVIEW_WIDTH, PREVIEW_HEIGHT)),
        );
        recorder.pipeline_barriers([BufferMemoryBarrier::new(
            [PipelineStage::Transfer].as_slice().into(),
            [MemoryAccessAs::TransferWrite].as_slice().into(),
            [PipelineStage::Host].as_slice().into(),
            [MemoryAccessAs::HostRead].as_slice().into(),
            BufferSubresourceRange::new(self.preview_buffer.clone(), 0, self.preview_buffer.size()),
            self.queue_family.clone(),
            self.queue_family.clone(),
        )
        .into()]);
        *self.preview_copies.lock().unwrap() += 1;
    }
}

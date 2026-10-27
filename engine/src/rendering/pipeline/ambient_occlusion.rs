//! Screen-space and ray-traced ambient occlusion.
//!
//! Both algorithms write the same full-resolution `r16f` image: 1 is open
//! hemisphere, 0 is fully occluded. The final composite samples that image and
//! multiplies indirect light by it only when the apply flag is set.
//!
//! Shader sources are embedded at compile time: touch this file after editing
//! an included `.glsl` so the macro expands again.

use std::{
    ops::DerefMut,
    path::PathBuf,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
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
        CommonImageFormat, ConcreteImageDescriptor, Image, Image2DDimensions, ImageAspect,
        ImageAspects, ImageDimensions, ImageFlags, ImageFormat, ImageLayout, ImageMultisampling,
        ImageSubresourceLayers, ImageTiling, ImageTrait, ImageUseAs,
    },
    image_view::ImageView,
    memory_barriers::{BufferMemoryBarrier, ImageMemoryBarrier, MemoryAccess, MemoryAccessAs},
    memory_heap::MemoryType,
    memory_management::{
        MemoryManagementTagSize, MemoryManagementTags, MemoryManagerTrait, UnallocatedResource,
    },
    memory_pool::{MemoryMap, MemoryPoolBacked, MemoryPoolFeatures},
    pipeline_layout::{PipelineLayout, PipelineLayoutDependant},
    pipeline_stage::{PipelineStage, PipelineStageRayTracingPipelineKHR, PipelineStages},
    queue_family::QueueFamily,
    raytracing_pipeline::RaytracingPipeline,
    sampler::{Filtering, MipmapMode, Sampler},
    shader_layout_binding::{BindingDescriptor, BindingType, NativeBindingType},
    shader_stage_access::{ShaderStageAccessIn, ShaderStageAccessInRayTracingKHR},
    shaders::{
        closest_hit_shader::ClosestHitShader, compute_shader::ComputeShader,
        miss_shader::MissShader, raygen_shader::RaygenShader,
    },
};

use crate::rendering::{
    RenderingResult, debug_state::AoAlgorithm, rendering_dimensions::RenderingDimensions,
};

const SSAO_SPV: &[u32] = inline_spirv!(
    r#"
#version 460
#extension GL_GOOGLE_include_directive : enable
#include "engine/shaders/ao/ssao.comp"
"#,
    glsl,
    comp,
    vulkan1_2,
    entry = "main"
);

const BLUR_SPV: &[u32] = inline_spirv!(
    r#"
#version 460
#extension GL_GOOGLE_include_directive : enable
#include "engine/shaders/ao/ao_blur.comp"
"#,
    glsl,
    comp,
    vulkan1_2,
    entry = "main"
);

const RTAO_RGEN_SPV: &[u32] = inline_spirv!(
    r#"
#version 460
#extension GL_GOOGLE_include_directive : enable
#include "engine/shaders/ao/rtao.rgen"
"#,
    glsl,
    rgen,
    vulkan1_2,
    entry = "main"
);

const RTAO_RMISS_SPV: &[u32] = inline_spirv!(
    r#"
#version 460
#extension GL_GOOGLE_include_directive : enable
#include "engine/shaders/ao/rtao.rmiss"
"#,
    glsl,
    rmiss,
    vulkan1_2,
    entry = "main"
);

const RTAO_RCHIT_SPV: &[u32] = inline_spirv!(
    r#"
#version 460
#extension GL_GOOGLE_include_directive : enable
#include "engine/shaders/ao/rtao.rchit"
"#,
    glsl,
    rchit,
    vulkan1_2,
    entry = "main"
);

/// The ambient-occlusion target and the two passes that fill it.
pub struct AmbientOcclusion {
    queue_family: Arc<QueueFamily>,
    ssao_pipeline: Arc<ComputePipeline>,
    blur_pipeline: Arc<ComputePipeline>,
    rtao_pipeline: Arc<RaytracingPipeline>,
    rtao_sbt: Arc<RaytracingBindingTables>,
    raw: Arc<ImageView>,
    image: Arc<ImageView>,
    storage_set: Arc<DescriptorSet>,
    sampled_layout: Arc<DescriptorSetLayout>,
    sampled_set: Arc<DescriptorSet>,
    width: u32,
    height: u32,
    /// Host copy of the AO image, written once when `ART_RTIC_AO_DUMP` is set.
    dump: Option<AoDump>,
}

struct AoDump {
    path: PathBuf,
    buffer: Arc<AllocatedBuffer>,
    copied: AtomicBool,
}

fn storage_image(
    device: &Arc<vulkan_framework::device::Device>,
    render_area: &RenderingDimensions,
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
            ImageFormat::from(CommonImageFormat::r16_sfloat),
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
        image,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        Some(name),
    )?)
}

fn ao_stages() -> PipelineStages {
    [
        PipelineStage::ComputeShader,
        PipelineStage::RayTracingPipelineKHR(PipelineStageRayTracingPipelineKHR::RayTracingShader),
    ]
    .as_slice()
    .into()
}

impl AmbientOcclusion {
    /// Layout of the image sampled by the final composite.
    pub fn descriptor_set_layout(&self) -> Arc<DescriptorSetLayout> {
        self.sampled_layout.clone()
    }

    pub fn descriptor_set(&self) -> Arc<DescriptorSet> {
        self.sampled_set.clone()
    }

    /// True when `ART_RTIC_AO_DUMP` asked for a copy, so the passes must run.
    pub fn dump_armed(&self) -> bool {
        self.dump.is_some()
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new(
        queue_family: Arc<QueueFamily>,
        render_area: &RenderingDimensions,
        memory_manager: Arc<Mutex<dyn MemoryManagerTrait>>,
        rt_descriptor_set_layout: Arc<DescriptorSetLayout>,
        gbuffer_descriptor_set_layout: Arc<DescriptorSetLayout>,
        status_descriptor_set_layout: Arc<DescriptorSetLayout>,
    ) -> RenderingResult<Self> {
        let device = queue_family.get_parent_device();
        let storage_access = [
            ShaderStageAccessIn::Compute,
            ShaderStageAccessIn::RayTracing(ShaderStageAccessInRayTracingKHR::RayGen),
        ]
        .as_slice()
        .into();
        let storage_layout = DescriptorSetLayout::new(
            device.clone(),
            [
                BindingDescriptor::new(
                    storage_access,
                    BindingType::Native(NativeBindingType::StorageImage),
                    0,
                    1,
                ),
                BindingDescriptor::new(
                    [ShaderStageAccessIn::Compute].as_slice().into(),
                    BindingType::Native(NativeBindingType::StorageImage),
                    1,
                    1,
                ),
            ]
            .as_slice(),
        )?;

        let compute_layout = PipelineLayout::new(
            device.clone(),
            [
                status_descriptor_set_layout.clone(),
                gbuffer_descriptor_set_layout.clone(),
                storage_layout.clone(),
            ]
            .as_slice(),
            &[],
            Some("ao_compute_layout"),
        )?;
        let rt_layout = PipelineLayout::new(
            device.clone(),
            [
                rt_descriptor_set_layout,
                status_descriptor_set_layout,
                gbuffer_descriptor_set_layout,
                storage_layout.clone(),
            ]
            .as_slice(),
            &[],
            Some("ao_rt_layout"),
        )?;

        let ssao_pipeline = ComputePipeline::new(
            None,
            compute_layout.clone(),
            (ComputeShader::new(device.clone(), SSAO_SPV)?, None),
            Some("ao_ssao"),
        )?;
        let blur_pipeline = ComputePipeline::new(
            None,
            compute_layout,
            (ComputeShader::new(device.clone(), BLUR_SPV)?, None),
            Some("ao_blur"),
        )?;
        let rtao_pipeline = RaytracingPipeline::new(
            rt_layout,
            1,
            RaygenShader::new(device.clone(), RTAO_RGEN_SPV)?,
            None,
            MissShader::new(device.clone(), RTAO_RMISS_SPV)?,
            None,
            ClosestHitShader::new(device.clone(), RTAO_RCHIT_SPV)?,
            None,
            Some("ao_rtao"),
        )?;

        let (raw, image, rtao_sbt) = {
            let mut mem_manager = memory_manager.lock().unwrap();
            let allocated = mem_manager.allocate_resources(
                &MemoryType::device_local(),
                &MemoryPoolFeatures::new(false),
                vec![
                    storage_image(&device, render_area, "ao_raw")?,
                    storage_image(&device, render_area, "ao")?,
                ],
                MemoryManagementTags::default()
                    .with_name("ambient_occlusion".to_string())
                    .with_size(MemoryManagementTagSize::MediumSmall),
            )?;
            let raw = full_view(allocated[0].image(), "ao_raw_view")?;
            let image = full_view(allocated[1].image(), "ao_view")?;
            let rtao_sbt = RaytracingBindingTables::new(
                rtao_pipeline.clone(),
                mem_manager.deref_mut(),
                MemoryManagementTags::default()
                    .with_name("ao_rtao_sbt".to_string())
                    .with_size(MemoryManagementTagSize::Small),
            )?;
            (raw, image, rtao_sbt)
        };

        let storage_pool = DescriptorPool::new(
            device.clone(),
            DescriptorPoolConcreteDescriptor::new(
                DescriptorPoolSizesConcreteDescriptor::new(0, 0, 0, 2, 0, 0, 0, 0, 0, None),
                1,
            ),
            Some("ao_storage_pool"),
        )?;
        let storage_set = DescriptorSet::new(storage_pool, storage_layout)?;
        {
            let raw = raw.clone();
            let image = image.clone();
            storage_set.bind_resources(|binder| {
                binder
                    .bind_storage_images_with_same_layout(0, ImageLayout::General, [raw].as_slice())
                    .unwrap();
                binder
                    .bind_storage_images_with_same_layout(
                        1,
                        ImageLayout::General,
                        [image].as_slice(),
                    )
                    .unwrap();
            })?;
        }

        let sampled_layout = DescriptorSetLayout::new(
            device.clone(),
            [BindingDescriptor::new(
                [ShaderStageAccessIn::Fragment].as_slice().into(),
                BindingType::Native(NativeBindingType::CombinedImageSampler),
                0,
                1,
            )]
            .as_slice(),
        )?;
        let sampled_pool = DescriptorPool::new(
            device.clone(),
            DescriptorPoolConcreteDescriptor::new(
                DescriptorPoolSizesConcreteDescriptor::new(0, 1, 0, 0, 0, 0, 0, 0, 0, None),
                1,
            ),
            Some("ao_sampled_pool"),
        )?;
        let sampler = Sampler::new(
            device.clone(),
            Filtering::Nearest,
            Filtering::Nearest,
            MipmapMode::ModeNearest,
            0.0,
        )?;
        let sampled_set = DescriptorSet::new(sampled_pool, sampled_layout.clone())?;
        {
            let image = image.clone();
            sampled_set.bind_resources(|binder| {
                binder
                    .bind_combined_images_samplers_with_same_layout_and_sampler(
                        0,
                        ImageLayout::ShaderReadOnlyOptimal,
                        sampler,
                        [image].as_slice(),
                    )
                    .unwrap();
            })?;
        }

        let dump = match std::env::var("ART_RTIC_AO_DUMP") {
            Ok(path) if !path.is_empty() && path != "0" => {
                let bytes = (render_area.width() as u64) * (render_area.height() as u64) * 2;
                let mut mem_manager = memory_manager.lock().unwrap();
                let allocated = mem_manager.allocate_resources(
                    &MemoryType::host_visible_and_coherent(),
                    &MemoryPoolFeatures::default(),
                    vec![
                        Buffer::new(
                            device.clone(),
                            ConcreteBufferDescriptor::new(
                                [BufferUseAs::TransferDst].as_slice().into(),
                                bytes,
                            ),
                            None,
                            Some("ao_dump"),
                        )?
                        .into(),
                    ],
                    MemoryManagementTags::default()
                        .with_name("ao_dump".to_string())
                        .with_size(MemoryManagementTagSize::MediumSmall),
                )?;
                Some(AoDump {
                    path: PathBuf::from(path),
                    buffer: allocated[0].buffer(),
                    copied: AtomicBool::new(false),
                })
            }
            _ => None,
        };

        Ok(Self {
            queue_family,
            ssao_pipeline,
            blur_pipeline,
            rtao_pipeline,
            rtao_sbt,
            raw,
            image,
            storage_set,
            sampled_layout,
            sampled_set,
            width: render_area.width(),
            height: render_area.height(),
            dump,
        })
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

    /// Clears both images to 1 (nothing occluded) and leaves the sampled image
    /// in the layout the final composite reads.
    pub fn record_init_commands(&self, recorder: &mut CommandBufferRecorder) {
        let top: PipelineStages = [PipelineStage::TopOfPipe].as_slice().into();
        let none: MemoryAccess = [].as_slice().into();
        let transfer: PipelineStages = [PipelineStage::Transfer].as_slice().into();
        let transfer_write: MemoryAccess = [MemoryAccessAs::TransferWrite].as_slice().into();
        for view in [&self.raw, &self.image] {
            recorder.pipeline_barriers([self.image_barrier(
                view,
                top,
                none,
                transfer,
                transfer_write,
                ImageLayout::Undefined,
                ImageLayout::TransferDstOptimal,
            )]);
            recorder.clear_color_image(
                ColorClearValues::Vec4(1.0, 1.0, 1.0, 1.0),
                view.image().into(),
            );
        }
        let shader_write: MemoryAccess = [MemoryAccessAs::ShaderWrite].as_slice().into();
        let shader_read: MemoryAccess = [MemoryAccessAs::ShaderRead].as_slice().into();
        recorder.pipeline_barriers([
            self.image_barrier(
                &self.raw,
                transfer,
                transfer_write,
                ao_stages(),
                shader_write,
                ImageLayout::TransferDstOptimal,
                ImageLayout::General,
            ),
            self.image_barrier(
                &self.image,
                transfer,
                transfer_write,
                [PipelineStage::FragmentShader].as_slice().into(),
                shader_read,
                ImageLayout::TransferDstOptimal,
                ImageLayout::ShaderReadOnlyOptimal,
            ),
        ]);
    }

    /// Fills the ambient-occlusion image with `algorithm` and returns the set
    /// the final composite samples. The image is left in
    /// `ShaderReadOnlyOptimal`.
    pub fn record_rendering_commands(
        &self,
        algorithm: AoAlgorithm,
        frame_index: u32,
        raytracing_descriptor_set: Arc<DescriptorSet>,
        gbuffer_descriptor_set: Arc<DescriptorSet>,
        status_descriptor_set: Arc<DescriptorSet>,
        execute: bool,
        recorder: &mut CommandBufferRecorder,
    ) -> Arc<DescriptorSet> {
        // Left at 1 (the clear) until AO is applied or inspected, so a frame
        // that does not use it does not pay for the passes.
        if !execute {
            return self.sampled_set.clone();
        }
        let shader_write: MemoryAccess = [MemoryAccessAs::ShaderWrite].as_slice().into();
        let shader_read: MemoryAccess = [MemoryAccessAs::ShaderRead].as_slice().into();
        let shader_rw: MemoryAccess = [MemoryAccessAs::ShaderRead, MemoryAccessAs::ShaderWrite]
            .as_slice()
            .into();

        // `raw` stays in General. `image` comes back from the previous composite.
        recorder.pipeline_barriers([
            self.image_barrier(
                &self.raw,
                ao_stages(),
                shader_read,
                ao_stages(),
                shader_write,
                ImageLayout::General,
                ImageLayout::General,
            ),
            self.image_barrier(
                &self.image,
                [PipelineStage::FragmentShader].as_slice().into(),
                shader_read,
                [PipelineStage::ComputeShader].as_slice().into(),
                shader_write,
                ImageLayout::ShaderReadOnlyOptimal,
                ImageLayout::General,
            ),
        ]);

        let groups = (self.width.div_ceil(8), self.height.div_ceil(8));
        match algorithm {
            AoAlgorithm::Ssao => {
                let sets = [
                    status_descriptor_set.clone(),
                    gbuffer_descriptor_set.clone(),
                    self.storage_set.clone(),
                ];
                recorder.bind_compute_pipeline(self.ssao_pipeline.clone());
                recorder.bind_descriptor_sets_for_compute_pipeline(
                    self.ssao_pipeline.get_parent_pipeline_layout(),
                    0,
                    sets.as_slice(),
                );
                recorder.dispatch(groups.0, groups.1, 1);
            }
            AoAlgorithm::Rtao => {
                let sets = [
                    raytracing_descriptor_set,
                    status_descriptor_set.clone(),
                    gbuffer_descriptor_set.clone(),
                    self.storage_set.clone(),
                ];
                recorder.bind_ray_tracing_pipeline(self.rtao_pipeline.clone());
                recorder.bind_descriptor_sets_for_ray_tracing_pipeline(
                    self.rtao_pipeline.get_parent_pipeline_layout(),
                    0,
                    sets.as_slice(),
                );
                recorder.trace_rays(
                    self.rtao_sbt.clone(),
                    vulkan_framework::image::Image3DDimensions::new(self.width, self.height, 1),
                );
            }
        }

        recorder.pipeline_barriers([self.image_barrier(
            &self.raw,
            ao_stages(),
            shader_write,
            [PipelineStage::ComputeShader].as_slice().into(),
            shader_read,
            ImageLayout::General,
            ImageLayout::General,
        )]);

        let sets = [
            status_descriptor_set,
            gbuffer_descriptor_set,
            self.storage_set.clone(),
        ];
        recorder.bind_compute_pipeline(self.blur_pipeline.clone());
        recorder.bind_descriptor_sets_for_compute_pipeline(
            self.blur_pipeline.get_parent_pipeline_layout(),
            0,
            sets.as_slice(),
        );
        recorder.dispatch(groups.0, groups.1, 1);

        let dump_this_frame = self
            .dump
            .as_ref()
            .is_some_and(|dump| frame_index == 8 && !dump.copied.swap(true, Ordering::Relaxed));
        if dump_this_frame {
            let dump = self.dump.as_ref().unwrap();
            let transfer: PipelineStages = [PipelineStage::Transfer].as_slice().into();
            let transfer_read: MemoryAccess = [MemoryAccessAs::TransferRead].as_slice().into();
            recorder.pipeline_barriers([self.image_barrier(
                &self.image,
                [PipelineStage::ComputeShader].as_slice().into(),
                shader_rw,
                transfer,
                transfer_read,
                ImageLayout::General,
                ImageLayout::TransferSrcOptimal,
            )]);
            let layers = ImageSubresourceLayers::new(
                ImageAspects::from([ImageAspect::Color].as_ref()),
                0,
                0,
                1,
            );
            recorder.copy_image_to_buffer(
                self.image.image(),
                ImageLayout::TransferSrcOptimal,
                layers,
                dump.buffer.clone(),
                0,
                ImageDimensions::from(Image2DDimensions::new(self.width, self.height)),
            );
            recorder.pipeline_barriers([
                self.image_barrier(
                    &self.image,
                    transfer,
                    transfer_read,
                    [PipelineStage::FragmentShader].as_slice().into(),
                    shader_read,
                    ImageLayout::TransferSrcOptimal,
                    ImageLayout::ShaderReadOnlyOptimal,
                ),
                BufferMemoryBarrier::new(
                    transfer,
                    [MemoryAccessAs::TransferWrite].as_slice().into(),
                    [PipelineStage::Host].as_slice().into(),
                    [MemoryAccessAs::HostRead].as_slice().into(),
                    BufferSubresourceRange::new(dump.buffer.clone(), 0, dump.buffer.size()),
                    self.queue_family.clone(),
                    self.queue_family.clone(),
                )
                .into(),
            ]);
        } else {
            recorder.pipeline_barriers([self.image_barrier(
                &self.image,
                [PipelineStage::ComputeShader].as_slice().into(),
                shader_rw,
                [PipelineStage::FragmentShader].as_slice().into(),
                shader_read,
                ImageLayout::General,
                ImageLayout::ShaderReadOnlyOptimal,
            )]);
        }

        self.sampled_set.clone()
    }

    /// Writes the frame-8 copy requested by `ART_RTIC_AO_DUMP`. Call after the
    /// frame's fence has been waited.
    pub fn write_dump(&self) {
        let Some(dump) = &self.dump else {
            return;
        };
        if !dump.copied.load(Ordering::Relaxed) {
            return;
        }
        let Ok(map) = MemoryMap::new(dump.buffer.get_backing_memory_pool()) else {
            eprintln!("AO dump: failed to map the readback buffer");
            return;
        };
        let Ok(range) = map.range::<u8>(dump.buffer.clone() as Arc<dyn MemoryPoolBacked>) else {
            eprintln!("AO dump: failed to read the readback buffer");
            return;
        };
        let pixels = (self.width as usize) * (self.height as usize);
        let bytes = range.as_slice();
        if bytes.len() < pixels * 2 {
            eprintln!("AO dump: readback is shorter than the image");
            return;
        }
        let mut floats = Vec::with_capacity(pixels);
        for px in 0..pixels {
            let h = u16::from_le_bytes([bytes[px * 2], bytes[px * 2 + 1]]);
            floats.push(f16_to_f32(h));
        }
        if let Err(err) = write_pfm(&dump.path, self.width, self.height, &floats) {
            eprintln!("AO dump: {err}");
        } else {
            let mean = floats.iter().sum::<f32>() / floats.len() as f32;
            println!(
                "AO dump {} ({}x{}, mean {mean:.3})",
                dump.path.display(),
                self.width,
                self.height
            );
        }
    }
}

fn f16_to_f32(h: u16) -> f32 {
    let sign = (h >> 15) & 1;
    let exp = (h >> 10) & 0x1f;
    let frac = h & 0x3ff;
    let bits = if exp == 0 {
        if frac == 0 {
            (sign as u32) << 31
        } else {
            let mut e = 127 - 14;
            let mut f = frac;
            while f & 0x400 == 0 {
                f <<= 1;
                e -= 1;
            }
            ((sign as u32) << 31) | (e << 23) | (((f & 0x3ff) as u32) << 13)
        }
    } else if exp == 31 {
        ((sign as u32) << 31) | (0xff << 23) | ((frac as u32) << 13)
    } else {
        ((sign as u32) << 31) | ((exp as u32 + 112) << 23) | ((frac as u32) << 13)
    };
    f32::from_bits(bits)
}

fn write_pfm(
    path: &std::path::Path,
    width: u32,
    height: u32,
    pixels: &[f32],
) -> std::io::Result<()> {
    use std::io::Write;
    let mut file = std::fs::File::create(path)?;
    write!(file, "PF\n{width} {height}\n-1.0\n")?;
    // PFM is bottom-up. The image is top-down, so flip rows.
    let row = width as usize;
    for y in (0..height as usize).rev() {
        for x in 0..row {
            let v = pixels[y * row + x];
            file.write_all(&v.to_le_bytes())?;
            file.write_all(&v.to_le_bytes())?;
            file.write_all(&v.to_le_bytes())?;
        }
    }
    Ok(())
}

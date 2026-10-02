use std::sync::Arc;

use inline_spirv::inline_spirv;

use crate::{
    binding_tables::{RaytracingBindingTables, ShaderBindingTableLayout},
    buffer::{AllocatedBuffer, BufferTrait},
    command_buffer::{PrimaryCommandBuffer, SubmittableCommandBufferTrait},
    command_pool::CommandPool,
    device::{Device, DeviceOwned},
    fence::Fence,
    image::Image3DDimensions,
    instance::Instance,
    memory_heap::MemoryType,
    memory_management::{
        AllocatedResource, DefaultMemoryManager, MemoryManagementTags, MemoryManagerTrait,
        UnallocatedResource,
    },
    memory_pool::{non_coherent_flush_range, MemoryMap, MemoryPoolBacked, MemoryPoolFeatures},
    pipeline_layout::PipelineLayout,
    prelude::{FrameworkError, VulkanError, VulkanResult},
    queue::Queue,
    queue_family::{ConcreteQueueFamilyDescriptor, QueueFamily, QueueFamilySupportedOperationType},
    raytracing_pipeline::RaytracingPipeline,
    shaders::{
        any_hit_shader::AnyHitShader, callable_shader::CallableShader,
        closest_hit_shader::ClosestHitShader, intersection_shader::IntersectionShader,
        miss_shader::MissShader, raygen_shader::RaygenShader,
    },
};

#[test]
fn packed_handles_are_read_without_destination_stride_and_all_hits_are_written() -> VulkanResult<()>
{
    let layout = ShaderBindingTableLayout::new(24, 32, 64, 128)?;
    let packed: Vec<_> = (0..5u8).flat_map(|group| vec![group + 1; 24]).collect();
    let mut records = vec![0xff; 32 * 3];
    layout.write_handles(&packed, 2, 3, &mut records)?;
    for record in 0..3 {
        assert_eq!(
            &records[record * 32..record * 32 + 24],
            &[record as u8 + 3; 24]
        );
        assert_eq!(&records[record * 32 + 24..(record + 1) * 32], &[0; 8]);
    }
    let mut miss = [0xff; 32];
    layout.write_handles(&packed, 1, 1, &mut miss)?;
    assert_eq!(&miss[..24], &[2; 24]);
    assert_eq!(&miss[24..], &[0; 8]);
    Ok(())
}

#[test]
fn sbt_regions_align_actual_addresses_and_fit_allocated_buffer_sizes() -> VulkanResult<()> {
    let layout = ShaderBindingTableLayout::new(24, 32, 128, 128)?;
    for count in [1, 3] {
        let buffer_size = layout.allocation_size(count)?;
        assert_eq!(buffer_size, 32 * u64::from(count) + 127);
        for misalignment in 0..128 {
            let address = 0x1000 + misalignment;
            let (offset, region) = layout.region(address, buffer_size, count)?;
            assert_eq!(region.device_address % 128, 0);
            assert_eq!(region.device_address, address + offset);
            assert_eq!(region.stride, 32);
            assert_eq!(region.size, 32 * u64::from(count));
            assert!(offset + region.size <= buffer_size);
            if count == 1 {
                assert_eq!(region.size, region.stride);
            }
        }
    }
    assert!(layout.region(0x1001, 24, 1).is_err());
    assert!(layout.region(0, 256, 1).is_err());
    assert!(layout.region(u64::MAX - 1, 256, 1).is_err());
    assert!(layout.region(u64::MAX - 127, 256, 5).is_err());
    Ok(())
}

#[test]
fn absent_regions_are_all_zero_without_an_allocation() -> VulkanResult<()> {
    let layout = ShaderBindingTableLayout::new(24, 32, 64, 128)?;
    assert_eq!(layout.allocation_size(0)?, 0);
    let (offset, empty) = layout.region(0, 0, 0)?;
    assert_eq!(
        (offset, empty.device_address, empty.stride, empty.size),
        (0, 0, 0, 0)
    );
    Ok(())
}

#[test]
fn sbt_invalid_layouts_and_truncated_handles_are_rejected() -> VulkanResult<()> {
    for (size, alignment, base, max) in [
        (0, 32, 64, 64),
        (24, 0, 64, 64),
        (24, 3, 64, 64),
        (24, 32, 0, 64),
        (24, 32, 3, 64),
        (65, 32, 64, 64),
    ] {
        assert!(ShaderBindingTableLayout::new(size, alignment, base, max).is_err());
    }
    let layout = ShaderBindingTableLayout::new(24, 32, 64, 128)?;
    let mut destination = [0xff; 32];
    assert!(layout
        .write_handles(&[1; 24], 1, 1, &mut destination)
        .is_err());
    assert_eq!(destination, [0xff; 32]);
    assert!(layout
        .write_handles(&[1; 24], 0, 1, &mut destination[..24])
        .is_err());
    assert!(layout
        .write_handles(&[1; 24], u32::MAX, 1, &mut destination)
        .is_err());
    Ok(())
}

#[test]
fn padded_sbt_flush_ranges_cover_whole_non_coherent_atoms() -> VulkanResult<()> {
    assert_eq!(non_coherent_flush_range(512, 95, 4096, 256)?, (512, 256));
    assert_eq!(non_coherent_flush_range(513, 95, 4096, 256)?, (512, 256));
    assert_eq!(non_coherent_flush_range(255, 2, 4096, 256)?, (0, 512));
    assert_eq!(non_coherent_flush_range(512, 188, 700, 256)?, (512, 188));
    assert!(non_coherent_flush_range(512, 189, 700, 256).is_err());
    assert!(non_coherent_flush_range(0, 32, 1024, 0).is_err());
    assert!(non_coherent_flush_range(u64::MAX, 1, u64::MAX, 256).is_err());
    Ok(())
}

const RAYGEN: &[u32] = inline_spirv!(
    r#"
#version 460
#extension GL_EXT_ray_tracing : require
void main() {}
"#,
    glsl,
    rgen,
    vulkan1_2
);
const MISS: &[u32] = inline_spirv!(
    r#"
#version 460
#extension GL_EXT_ray_tracing : require
void main() {}
"#,
    glsl,
    rmiss,
    vulkan1_2
);
const CLOSEST_HIT: &[u32] = inline_spirv!(
    r#"
#version 460
#extension GL_EXT_ray_tracing : require
void main() {}
"#,
    glsl,
    rchit,
    vulkan1_2
);

const CALLABLE: &[u32] = inline_spirv!(
    r#"
#version 460
#extension GL_EXT_ray_tracing : require
void main() {}
"#,
    glsl,
    rcall,
    vulkan1_2
);

const INTERSECTION: &[u32] = inline_spirv!(
    r#"
#version 460
#extension GL_EXT_ray_tracing : require
hitAttributeEXT vec2 hit;
void main() { hit = vec2(0.0); reportIntersectionEXT(1.0, 0u); }
"#,
    glsl,
    rint,
    vulkan1_2
);
const ANY_HIT: &[u32] = inline_spirv!(
    r#"
#version 460
#extension GL_EXT_ray_tracing : require
void main() {}
"#,
    glsl,
    rahit,
    vulkan1_2
);

fn setup_raytracing_pipeline() -> VulkanResult<Option<Arc<RaytracingPipeline>>> {
    setup_raytracing_pipeline_with_options(false, false, false)
}

fn setup_raytracing_pipeline_with_options(
    intersection: bool,
    any_hit: bool,
    callable: bool,
) -> VulkanResult<Option<Arc<RaytracingPipeline>>> {
    let instance = Instance::new(&[], &[], &"sbt_tests".into(), &"headless".into())?;
    let descriptor =
        ConcreteQueueFamilyDescriptor::new(&[QueueFamilySupportedOperationType::Compute], &[1.0]);
    let extensions = [
        ash::khr::acceleration_structure::NAME,
        ash::khr::ray_tracing_pipeline::NAME,
        ash::khr::deferred_host_operations::NAME,
    ]
    .map(|name| name.to_str().unwrap().to_owned());
    let device = match Device::new(instance, &[descriptor], &extensions, None) {
        Ok(device) => device,
        Err(VulkanError::Framework(FrameworkError::NoSuitableDeviceFound)) => {
            eprintln!(
                "Skipping GPU SBT test: no device supports the required ray-tracing extensions"
            );
            return Ok(None);
        }
        Err(err) => return Err(err),
    };
    let layout = PipelineLayout::new(device.clone(), &[], &[], None)?;
    let callable_shader = if callable {
        Some(CallableShader::new(device.clone(), CALLABLE)?)
    } else {
        None
    };
    RaytracingPipeline::new(
        layout,
        1,
        RaygenShader::new(device.clone(), RAYGEN)?,
        intersection
            .then(|| IntersectionShader::new(device.clone(), INTERSECTION))
            .transpose()?,
        MissShader::new(device.clone(), MISS)?,
        any_hit
            .then(|| AnyHitShader::new(device.clone(), ANY_HIT))
            .transpose()?,
        ClosestHitShader::new(device, CLOSEST_HIT)?,
        callable_shader,
        None,
    )
    .map(Some)
}

struct CapturingMemoryManager {
    inner: DefaultMemoryManager,
    buffers: Vec<Arc<AllocatedBuffer>>,
}
impl DeviceOwned for CapturingMemoryManager {
    fn get_parent_device(&self) -> Arc<Device> {
        self.inner.get_parent_device()
    }
}
impl MemoryManagerTrait for CapturingMemoryManager {
    fn allocate_resources(
        &mut self,
        memory_type: &MemoryType,
        features: &MemoryPoolFeatures,
        resources: Vec<UnallocatedResource>,
        tags: MemoryManagementTags,
    ) -> VulkanResult<Vec<AllocatedResource>> {
        let allocations = self
            .inner
            .allocate_resources(memory_type, features, resources, tags)?;
        self.buffers
            .extend(allocations.iter().map(|allocation| allocation.buffer()));
        Ok(allocations)
    }
}

#[test]
fn gpu_sbt_regions_match_packed_driver_handles_and_absent_callable() -> VulkanResult<()> {
    let Some(pipeline) = setup_raytracing_pipeline()? else {
        return Ok(());
    };
    let device = pipeline.get_parent_device();
    let info = device.ray_tracing_info().as_ref().unwrap();
    let handle_size = info.shader_group_handle_size() as usize;
    let packed = unsafe {
        device
            .ash_ext_raytracing_pipeline_khr()
            .as_ref()
            .unwrap()
            .get_ray_tracing_shader_group_handles(
                pipeline.ash_handle(),
                0,
                pipeline.shader_group_size(),
                pipeline.shader_group_size() as usize * handle_size,
            )
    }?;
    let mut memory = CapturingMemoryManager {
        inner: DefaultMemoryManager::new(device.clone()),
        buffers: Vec::new(),
    };
    let sbt = RaytracingBindingTables::new(pipeline, &mut memory, MemoryManagementTags::default())?;
    assert_eq!(
        memory.buffers.len(),
        3,
        "an absent callable must not allocate a buffer"
    );
    let callable = sbt.ash_callable_strided();
    assert_eq!(
        (callable.device_address, callable.stride, callable.size),
        (0, 0, 0)
    );
    let regions = [
        sbt.ash_raygen_strided(),
        sbt.ash_miss_strided(),
        sbt.ash_closesthit_strided(),
    ];
    for (index, (buffer, region)) in memory.buffers.iter().zip(regions).enumerate() {
        let base = unsafe {
            device.ash_handle().get_buffer_device_address(
                &ash::vk::BufferDeviceAddressInfo::default().buffer(buffer.ash_handle()),
            )
        };
        let offset = (region.device_address - base) as usize;
        assert_eq!(
            region.device_address % u64::from(info.shader_group_base_alignment()),
            0
        );
        assert_eq!(
            region.stride % u64::from(info.shader_group_handle_alignment()),
            0
        );
        assert!(region.stride >= handle_size as u64);
        assert_eq!(region.size, region.stride);
        assert!(offset as u64 + region.size <= buffer.size());
        let mapping = MemoryMap::new(buffer.get_backing_memory_pool())?;
        let range = mapping.range::<u8>(buffer.clone() as Arc<dyn MemoryPoolBacked>)?;
        let record = &range.as_slice()[offset..offset + region.size as usize];
        assert_eq!(
            &record[..handle_size],
            &packed[index * handle_size..(index + 1) * handle_size]
        );
        assert!(record[handle_size..].iter().all(|byte| *byte == 0));
    }
    Ok(())
}

#[test]
fn gpu_optional_hit_and_callable_groups_match_all_packed_handles() -> VulkanResult<()> {
    for intersection in [false, true] {
        for any_hit in [false, true] {
            for callable in [false, true] {
                let Some(pipeline) =
                    setup_raytracing_pipeline_with_options(intersection, any_hit, callable)?
                else {
                    return Ok(());
                };
                let device = pipeline.get_parent_device();
                let info = device.ray_tracing_info().as_ref().unwrap();
                let handle_size = info.shader_group_handle_size() as usize;
                let hit_count = 1 + u32::from(intersection) + u32::from(any_hit);
                assert_eq!(
                    pipeline.shader_group_size(),
                    2 + hit_count + u32::from(callable)
                );
                assert_eq!(pipeline.callable_shader_present(), callable);
                let packed = unsafe {
                    device
                        .ash_ext_raytracing_pipeline_khr()
                        .as_ref()
                        .unwrap()
                        .get_ray_tracing_shader_group_handles(
                            pipeline.ash_handle(),
                            0,
                            pipeline.shader_group_size(),
                            pipeline.shader_group_size() as usize * handle_size,
                        )
                }?;
                let mut memory = CapturingMemoryManager {
                    inner: DefaultMemoryManager::new(device.clone()),
                    buffers: Vec::new(),
                };
                let sbt = RaytracingBindingTables::new(
                    pipeline,
                    &mut memory,
                    MemoryManagementTags::default(),
                )?;
                assert_eq!(memory.buffers.len(), 3 + usize::from(callable));
                let mut regions = vec![
                    (sbt.ash_raygen_strided(), 0, 1),
                    (sbt.ash_miss_strided(), 1, 1),
                    (sbt.ash_closesthit_strided(), 2, hit_count),
                ];
                if callable {
                    regions.push((sbt.ash_callable_strided(), 2 + hit_count, 1));
                } else {
                    let empty = sbt.ash_callable_strided();
                    assert_eq!((empty.device_address, empty.stride, empty.size), (0, 0, 0));
                }
                for (buffer, (region, first_group, record_count)) in
                    memory.buffers.iter().zip(regions)
                {
                    let base = unsafe {
                        device.ash_handle().get_buffer_device_address(
                            &ash::vk::BufferDeviceAddressInfo::default()
                                .buffer(buffer.ash_handle()),
                        )
                    };
                    let offset = (region.device_address - base) as usize;
                    assert_eq!(
                        region.device_address % u64::from(info.shader_group_base_alignment()),
                        0
                    );
                    assert_eq!(
                        region.stride % u64::from(info.shader_group_handle_alignment()),
                        0
                    );
                    assert_eq!(region.size, region.stride * u64::from(record_count));
                    assert!(offset as u64 + region.size <= buffer.size());
                    let mapping = MemoryMap::new(buffer.get_backing_memory_pool())?;
                    let range = mapping.range::<u8>(buffer.clone() as Arc<dyn MemoryPoolBacked>)?;
                    for record in 0..record_count as usize {
                        let destination = offset + record * region.stride as usize;
                        let source = (first_group as usize + record) * handle_size;
                        assert_eq!(
                            &range.as_slice()[destination..destination + handle_size],
                            &packed[source..source + handle_size]
                        );
                        assert!(range.as_slice()
                            [destination + handle_size..destination + region.stride as usize]
                            .iter()
                            .all(|byte| *byte == 0));
                    }
                }
            }
        }
    }
    Ok(())
}

const RAYGEN_CALLABLE_RESULT: &[u32] = inline_spirv!(
    r#"
#version 460
#extension GL_EXT_ray_tracing : require
layout(set = 0, binding = 0, std430) buffer Result { uint value; } result;
layout(location = 0) callableDataEXT uint payload;
void main() {
    payload = 10u;
    executeCallableEXT(0, 0);
    result.value = payload;
}
"#,
    glsl,
    rgen,
    vulkan1_2
);
const CALLABLE_RESULT: &[u32] = inline_spirv!(
    r#"
#version 460
#extension GL_EXT_ray_tracing : require
layout(location = 0) callableDataInEXT uint payload;
void main() { payload += 32u; }
"#,
    glsl,
    rcall,
    vulkan1_2
);

#[test]
fn gpu_callable_dispatch_uses_the_handle_after_all_hit_groups() -> VulkanResult<()> {
    let Some(existing_pipeline) = setup_raytracing_pipeline()? else {
        return Ok(());
    };
    let device = existing_pipeline.get_parent_device();
    let mut memory = DefaultMemoryManager::new(device.clone());
    let output = crate::buffer::Buffer::new(
        device.clone(),
        crate::buffer::ConcreteBufferDescriptor::new(
            ash::vk::BufferUsageFlags::STORAGE_BUFFER.into(),
            4,
        ),
        None,
        None,
    )?;
    let allocations = memory.allocate_resources(
        &MemoryType::host_visible_and_coherent(),
        &MemoryPoolFeatures::new(false),
        vec![output.into()],
        MemoryManagementTags::default(),
    )?;
    let output = allocations[0].buffer();
    {
        let mapping = MemoryMap::new(output.get_backing_memory_pool())?;
        let mut range = mapping.range::<u32>(output.clone() as Arc<dyn MemoryPoolBacked>)?;
        range.as_mut_slice()[0] = 0;
    }
    let binding = crate::shader_layout_binding::BindingDescriptor::new(
        ash::vk::ShaderStageFlags::RAYGEN_KHR.into(),
        crate::shader_layout_binding::BindingType::Native(
            crate::shader_layout_binding::NativeBindingType::StorageBuffer,
        ),
        0,
        1,
    );
    let set_layout =
        crate::descriptor_set_layout::DescriptorSetLayout::new(device.clone(), &[binding])?;
    let sizes = crate::descriptor_pool::DescriptorPoolSizesConcreteDescriptor::new(
        0, 0, 0, 0, 0, 0, 1, 0, 0, None,
    );
    let pool = crate::descriptor_pool::DescriptorPool::new(
        device.clone(),
        crate::descriptor_pool::DescriptorPoolConcreteDescriptor::new(sizes, 1),
        None,
    )?;
    let set = crate::descriptor_set::DescriptorSet::new(pool, set_layout.clone())?;
    set.bind_resources(|binder| {
        let buffer: Arc<dyn BufferTrait> = output.clone();
        binder
            .bind_storage_buffers(0, &[(buffer, None, None)])
            .unwrap();
    })?;
    let layout = PipelineLayout::new(device.clone(), &[set_layout], &[], None)?;
    let pipeline = RaytracingPipeline::new(
        layout.clone(),
        1,
        RaygenShader::new(device.clone(), RAYGEN_CALLABLE_RESULT)?,
        Some(IntersectionShader::new(device.clone(), INTERSECTION)?),
        MissShader::new(device.clone(), MISS)?,
        Some(AnyHitShader::new(device.clone(), ANY_HIT)?),
        ClosestHitShader::new(device.clone(), CLOSEST_HIT)?,
        Some(CallableShader::new(device.clone(), CALLABLE_RESULT)?),
        None,
    )?;
    assert_eq!(pipeline.shader_group_size(), 6);
    let sbt = RaytracingBindingTables::new(
        pipeline.clone(),
        &mut memory,
        MemoryManagementTags::default(),
    )?;
    assert_eq!(
        sbt.ash_closesthit_strided().size,
        3 * sbt.ash_closesthit_strided().stride
    );
    assert_ne!(sbt.ash_callable_strided().device_address, 0);
    let family = QueueFamily::new(device.clone(), 0)?;
    let pool = CommandPool::new(family.clone(), None)?;
    let commands = PrimaryCommandBuffer::new(pool, None)?;
    commands.record_one_time_submit(|recorder| {
        recorder.bind_ray_tracing_pipeline(pipeline.clone());
        recorder.bind_descriptor_sets_for_ray_tracing_pipeline(layout.clone(), 0, &[set.clone()]);
        recorder.trace_rays(sbt.clone(), Image3DDimensions::new(1, 1, 1));
    })?;
    let queue = Queue::new(family, None)?;
    drop(queue.submit(&[commands], &[], &[], Fence::new(device, false, None)?)?);
    let mapping = MemoryMap::new(output.get_backing_memory_pool())?;
    let range = mapping.range::<u32>(output as Arc<dyn MemoryPoolBacked>)?;
    assert_eq!(
        range.as_slice()[0],
        42,
        "the callable must update the raygen payload"
    );
    Ok(())
}

#[test]
fn recorded_trace_retains_sbt_through_cancellation_and_execution() -> VulkanResult<()> {
    let Some(pipeline) = setup_raytracing_pipeline_with_options(true, true, true)? else {
        return Ok(());
    };
    let device = pipeline.get_parent_device();
    let mut memory = DefaultMemoryManager::new(device.clone());
    let sbt = RaytracingBindingTables::new(
        pipeline.clone(),
        &mut memory,
        MemoryManagementTags::default(),
    )?;
    let weak = Arc::downgrade(&sbt);
    let family = QueueFamily::new(device.clone(), 0)?;
    let pool = CommandPool::new(family.clone(), None)?;
    let command_buffer = PrimaryCommandBuffer::new(pool, None)?;
    command_buffer.record_one_time_submit(|recorder| {
        recorder.bind_ray_tracing_pipeline(pipeline.clone());
        recorder.trace_rays(sbt.clone(), Image3DDimensions::new(1, 1, 1));
        recorder.trace_rays(sbt.clone(), Image3DDimensions::new(1, 1, 1));
    })?;
    assert_eq!(
        Arc::strong_count(&sbt),
        2,
        "the resource set must deduplicate references to the same SBT"
    );
    drop(sbt);
    assert!(weak.upgrade().is_some());
    command_buffer.mark_execution_begin()?;
    command_buffer.mark_execution_cancel()?;
    assert!(
        weak.upgrade().is_some(),
        "cancel must retain one-time resources"
    );
    let queue = Queue::new(family, None)?;
    let waiter = queue.submit(
        &[command_buffer.clone()],
        &[],
        &[],
        Fence::new(device, false, None)?,
    )?;
    drop(waiter);
    assert!(
        weak.upgrade().is_none(),
        "one-time completion must release the SBT"
    );
    Ok(())
}

#[test]
fn pipeline_rejects_foreign_optional_shader_and_excessive_recursion_depth() -> VulkanResult<()> {
    let Some(pipeline) = setup_raytracing_pipeline()? else {
        return Ok(());
    };
    let Some(foreign_pipeline) = setup_raytracing_pipeline()? else {
        return Ok(());
    };
    let device = pipeline.get_parent_device();
    let layout = PipelineLayout::new(device.clone(), &[], &[], None)?;
    let raygen = RaygenShader::new(device.clone(), RAYGEN)?;
    let miss = MissShader::new(device.clone(), MISS)?;
    let closest_hit = ClosestHitShader::new(device.clone(), CLOSEST_HIT)?;
    let foreign_any_hit = AnyHitShader::new(foreign_pipeline.get_parent_device(), ANY_HIT)?;
    assert!(matches!(
        RaytracingPipeline::new(
            layout.clone(),
            1,
            raygen.clone(),
            None,
            miss.clone(),
            Some(foreign_any_hit),
            closest_hit.clone(),
            None,
            None
        ),
        Err(VulkanError::Framework(
            FrameworkError::ResourceFromIncompatibleDevice
        ))
    ));
    if let Some(invalid_depth) = device
        .ray_tracing_info()
        .as_ref()
        .unwrap()
        .max_ray_recursion_depth()
        .checked_add(1)
    {
        assert!(matches!(
            RaytracingPipeline::new(
                layout,
                invalid_depth,
                raygen,
                None,
                miss,
                None,
                closest_hit,
                None,
                None
            ),
            Err(VulkanError::Vulkan(
                ash::vk::Result::ERROR_INITIALIZATION_FAILED
            ))
        ));
    }
    Ok(())
}

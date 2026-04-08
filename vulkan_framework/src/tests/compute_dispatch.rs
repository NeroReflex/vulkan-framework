#[cfg(test)]
mod compute_dispatch_tests {
    use std::{mem::size_of, sync::Arc};

    use crate::memory_management::MemoryManagerTrait;
    use crate::memory_pool::MemoryPoolBacked;
    use crate::prelude::*;
    use inline_spirv::inline_spirv;

    const COMPUTE_SPV: &[u32] = inline_spirv!(
        r#"
#version 460
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
layout(std430, binding = 0) buffer Buf { uint data[]; } buf;
void main() {
    uint idx = gl_GlobalInvocationID.x;
    buf.data[idx] = idx + 1u;
}
"#,
        glsl,
        comp,
        vulkan1_2,
        entry = "main"
    );

    #[test]
    fn test_compute_dispatch() -> VulkanResult<()> {
        match crate::tests::common::setup_test_device() {
            Ok((_instance, device)) => {
                let queue_family = crate::queue_family::QueueFamily::new(device.clone(), 0)?;
                let command_pool =
                    crate::command_pool::CommandPool::new(queue_family.clone(), Some("test_pool"))?;
                let cmd_buffer = crate::command_buffer::PrimaryCommandBuffer::new(
                    command_pool.clone(),
                    Some("test_cb"),
                )?;
                let queue = crate::queue::Queue::new(queue_family.clone(), Some("test_queue"))?;

                let element_count = 64u32;
                let element_size = size_of::<u32>() as u64;
                let buffer_size = (element_count as u64) * element_size;

                // create unallocated storage buffer
                let buffer_unalloc = crate::buffer::Buffer::new(
                    device.clone(),
                    crate::buffer::ConcreteBufferDescriptor::new(
                        crate::buffer::BufferUsage::from(
                            (ash::vk::BufferUsageFlags::STORAGE_BUFFER
                                | ash::vk::BufferUsageFlags::TRANSFER_SRC)
                                .as_raw(),
                        ),
                        buffer_size,
                    ),
                    None,
                    Some("compute_buf"),
                )?;

                let mut mem_mgr =
                    crate::memory_management::DefaultMemoryManager::new(device.clone());

                let allocations = mem_mgr.allocate_resources(
                    &crate::memory_heap::MemoryType::host_visible_and_coherent(),
                    &crate::memory_pool::MemoryPoolFeatures::new(false),
                    vec![buffer_unalloc.into()],
                    crate::memory_management::MemoryManagementTags::default(),
                )?;

                let buffer = allocations[0].buffer();

                // Descriptor layout/pool/set
                let binding = crate::shader_layout_binding::BindingDescriptor::new(
                    crate::shader_stage_access::ShaderStagesAccess::compute(),
                    crate::shader_layout_binding::BindingType::Native(
                        crate::shader_layout_binding::NativeBindingType::StorageBuffer,
                    ),
                    0,
                    1,
                );

                let ds_layout = crate::descriptor_set_layout::DescriptorSetLayout::new(
                    device.clone(),
                    &[binding],
                )?;

                let pool_sizes = crate::descriptor_pool::DescriptorPoolSizesConcreteDescriptor::new(
                    0, 0, 0, 0, 0, 0, 1, 0, 0, None,
                );
                let pool_desc =
                    crate::descriptor_pool::DescriptorPoolConcreteDescriptor::new(pool_sizes, 1);
                let desc_pool = crate::descriptor_pool::DescriptorPool::new(
                    device.clone(),
                    pool_desc,
                    Some("test_desc_pool"),
                )?;

                let desc_set = crate::descriptor_set::DescriptorSet::new(
                    desc_pool.clone(),
                    ds_layout.clone(),
                )?;

                desc_set.bind_resources(|binder| {
                    let buffer_trait: Arc<dyn crate::buffer::BufferTrait> = buffer.clone();
                    binder
                        .bind_storage_buffers(0, [(buffer_trait.clone(), None, None)].as_slice())
                        .unwrap();
                })?;

                let pipeline_layout = crate::pipeline_layout::PipelineLayout::new(
                    device.clone(),
                    &[ds_layout.clone()],
                    &[],
                    Some("compute_pl_layout"),
                )?;

                let shader = crate::shaders::compute_shader::ComputeShader::new(
                    device.clone(),
                    COMPUTE_SPV,
                )?;
                let pipeline = crate::compute_pipeline::ComputePipeline::new(
                    None,
                    pipeline_layout.clone(),
                    (shader.clone(), None),
                    Some("test_compute"),
                )?;

                // Record compute dispatch
                cmd_buffer.record_one_time_submit(|rec| {
                    rec.bind_compute_pipeline(pipeline.clone());
                    rec.bind_descriptor_sets_for_compute_pipeline(
                        pipeline_layout.clone(),
                        0,
                        &[desc_set.clone()],
                    );
                    rec.dispatch(element_count, 1, 1);
                })?;

                let fence = crate::fence::Fence::new(device.clone(), false, Some("test_fence"))?;
                let cbs: Vec<Arc<dyn crate::command_buffer::CommandBufferTrait>> =
                    vec![cmd_buffer.clone()];
                let waiter = queue.submit(cbs.as_slice(), &[], &[], fence.clone())?;
                drop(waiter);

                // verify results
                let mem_map = crate::memory_pool::MemoryMap::new(buffer.get_backing_memory_pool())?;
                let range = mem_map
                    .range::<u32>(buffer.clone() as Arc<dyn crate::memory_pool::MemoryPoolBacked>)?;
                let slice = range.as_slice();
                for i in 0..element_count as usize {
                    assert_eq!(slice[i], (i as u32) + 1u32);
                }

                Ok(())
            }
            Err(err) => {
                eprintln!("Skipping test_compute_dispatch: {}", err);
                Ok(())
            }
        }
    }
}

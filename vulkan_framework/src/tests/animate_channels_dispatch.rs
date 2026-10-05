#[cfg(test)]
mod animate_channels_dispatch_tests {
    use std::mem::size_of;
    use std::sync::Arc;

    use inline_spirv::inline_spirv;

    use crate::memory_management::MemoryManagerTrait;
    use crate::memory_pool::MemoryPoolBacked;
    use crate::prelude::*;

    const CHANNELS_SPV: &[u32] = inline_spirv!(
        r#"
#version 460
#include "engine/shaders/skin/animate_channels.comp"
"#,
        glsl,
        comp,
        vulkan1_2,
        entry = "main"
    );

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct SkeletonGpuElement {
        offset_matrix: [f32; 16],
        armature_node_index: u32,
        _pad: [u32; 3],
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct ArmatureGpuElement {
        transform: [f32; 16],
        parent_index: u32,
        _pad: [u32; 3],
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct AnimationGpuChannel {
        armature_element_index: u32,
        position_key_count: u32,
        position_key_times: [f32; 64],
        position_key_value_x: [f32; 64],
        position_key_value_y: [f32; 64],
        position_key_value_z: [f32; 64],
        rotation_key_count: u32,
        rotation_key_times: [f32; 64],
        rotation_key_value_x: [f32; 64],
        rotation_key_value_y: [f32; 64],
        rotation_key_value_z: [f32; 64],
        rotation_key_value_w: [f32; 64],
        scaling_key_count: u32,
        scaling_key_times: [f32; 64],
        scaling_key_value_x: [f32; 64],
        scaling_key_value_y: [f32; 64],
        scaling_key_value_z: [f32; 64],
    }

    fn identity_col_major() -> [f32; 16] {
        [
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ]
    }

    fn cpu_bind_pose(
        original: &SkeletonGpuElement,
        armature: &[ArmatureGpuElement],
    ) -> [f32; 16] {
        let mut current = original.armature_node_index as usize;
        let mut global = identity_col_major();
        loop {
            let node = &armature[current];
            global = mul_col_major(&node.transform, &global);
            if node.parent_index as usize == current {
                break;
            }
            current = node.parent_index as usize;
        }
        mul_col_major(&global, &original.offset_matrix)
    }

    fn mul_col_major(a: &[f32; 16], b: &[f32; 16]) -> [f32; 16] {
        let mut out = [0.0f32; 16];
        for col in 0..4 {
            for row in 0..4 {
                let mut sum = 0.0;
                for k in 0..4 {
                    sum += a[k * 4 + row] * b[col * 4 + k];
                }
                out[col * 4 + row] = sum;
            }
        }
        out
    }

    #[test]
    fn animate_channels_dispatch() -> VulkanResult<()> {
        let (_instance, device) = match crate::tests::common::setup_test_device_validated() {
            Ok(pair) => pair,
            Err(err) => {
                eprintln!("Skipping animate_channels_dispatch: {err}");
                return Ok(());
            }
        };

        let queue_family = crate::queue_family::QueueFamily::new(device.clone(), 0)?;
        let command_pool =
            crate::command_pool::CommandPool::new(queue_family.clone(), Some("skin_test_pool"))?;
        let cmd_buffer = crate::command_buffer::PrimaryCommandBuffer::new(
            command_pool.clone(),
            Some("skin_test_cb"),
        )?;
        let queue = crate::queue::Queue::new(queue_family.clone(), Some("skin_test_queue"))?;

        let original = SkeletonGpuElement {
            offset_matrix: identity_col_major(),
            armature_node_index: 0,
            _pad: [0; 3],
        };
        let armature = [ArmatureGpuElement {
            transform: identity_col_major(),
            parent_index: 0,
            _pad: [0; 3],
        }];
        let channel = AnimationGpuChannel {
            armature_element_index: 0,
            position_key_count: 0,
            position_key_times: [0.0; 64],
            position_key_value_x: [0.0; 64],
            position_key_value_y: [0.0; 64],
            position_key_value_z: [0.0; 64],
            rotation_key_count: 0,
            rotation_key_times: [0.0; 64],
            rotation_key_value_x: [0.0; 64],
            rotation_key_value_y: [0.0; 64],
            rotation_key_value_z: [0.0; 64],
            rotation_key_value_w: [0.0; 64],
            scaling_key_count: 0,
            scaling_key_times: [0.0; 64],
            scaling_key_value_x: [0.0; 64],
            scaling_key_value_y: [0.0; 64],
            scaling_key_value_z: [0.0; 64],
        };

        let per_frame = [0.0f32; 16];
        let buffers = [
            (size_of::<SkeletonGpuElement>(), &original as *const _ as *const u8),
            (size_of::<[f32; 16]>(), per_frame.as_ptr() as *const u8),
            (size_of::<ArmatureGpuElement>(), armature.as_ptr() as *const u8),
            (size_of::<AnimationGpuChannel>(), &channel as *const _ as *const u8),
        ];

        let mut allocated = Vec::new();
        let mut mem_mgr = crate::memory_management::DefaultMemoryManager::new(device.clone());
        for (index, (bytes, _)) in buffers.iter().enumerate() {
            let buffer = crate::buffer::Buffer::new(
                device.clone(),
                crate::buffer::ConcreteBufferDescriptor::new(
                    crate::buffer::BufferUsage::from(
                        crate::ash::vk::BufferUsageFlags::STORAGE_BUFFER.as_raw(),
                    ),
                    *bytes as u64,
                ),
                None,
                Some(&format!("skin_buf_{index}")),
            )?;
            let allocs = mem_mgr.allocate_resources(
                &crate::memory_heap::MemoryType::host_visible_and_coherent(),
                &crate::memory_pool::MemoryPoolFeatures::new(false),
                vec![buffer.into()],
                crate::memory_management::MemoryManagementTags::default(),
            )?;
            allocated.push(allocs[0].buffer());
        }

        {
            let map = crate::memory_pool::MemoryMap::new(
                allocated[0].get_backing_memory_pool(),
            )?;
            let mut range = map.range::<u8>(allocated[0].clone() as Arc<dyn MemoryPoolBacked>)?;
            unsafe {
                std::ptr::copy_nonoverlapping(
                    &original as *const SkeletonGpuElement as *const u8,
                    range.as_mut_slice().as_mut_ptr(),
                    size_of::<SkeletonGpuElement>(),
                );
            }
        }
        {
            let map = crate::memory_pool::MemoryMap::new(
                allocated[2].get_backing_memory_pool(),
            )?;
            let mut range = map.range::<u8>(allocated[2].clone() as Arc<dyn MemoryPoolBacked>)?;
            unsafe {
                std::ptr::copy_nonoverlapping(
                    armature.as_ptr() as *const u8,
                    range.as_mut_slice().as_mut_ptr(),
                    size_of::<ArmatureGpuElement>(),
                );
            }
        }
        {
            let map = crate::memory_pool::MemoryMap::new(
                allocated[3].get_backing_memory_pool(),
            )?;
            let mut range = map.range::<u8>(allocated[3].clone() as Arc<dyn MemoryPoolBacked>)?;
            unsafe {
                std::ptr::copy_nonoverlapping(
                    &channel as *const AnimationGpuChannel as *const u8,
                    range.as_mut_slice().as_mut_ptr(),
                    size_of::<AnimationGpuChannel>(),
                );
            }
        }

        let binding = |binding: u32| {
            crate::shader_layout_binding::BindingDescriptor::new(
                crate::shader_stage_access::ShaderStagesAccess::compute(),
                crate::shader_layout_binding::BindingType::Native(
                    crate::shader_layout_binding::NativeBindingType::StorageBuffer,
                ),
                binding,
                1,
            )
        };
        let ds_layout = crate::descriptor_set_layout::DescriptorSetLayout::new(
            device.clone(),
            &[binding(0), binding(1), binding(2), binding(3)],
        )?;
        let pool = crate::descriptor_pool::DescriptorPool::new(
            device.clone(),
            crate::descriptor_pool::DescriptorPoolConcreteDescriptor::new(
                crate::descriptor_pool::DescriptorPoolSizesConcreteDescriptor::new(
                    0, 0, 0, 0, 0, 0, 4, 0, 0, None,
                ),
                1,
            ),
            Some("skin_desc_pool"),
        )?;
        let desc_set = crate::descriptor_set::DescriptorSet::new(pool, ds_layout.clone())?;
        desc_set.bind_resources(|binder| {
            for (binding, buffer) in allocated.iter().enumerate() {
                let b: Arc<dyn crate::buffer::BufferTrait> = buffer.clone();
                binder
                    .bind_storage_buffers(binding as u32, [(b, None, None)].as_slice())
                    .unwrap();
            }
            Ok::<(), VulkanError>(())
        })?;

        let push_range = crate::push_constant_range::PushConstanRange::new(
            0,
            8,
            crate::shader_stage_access::ShaderStagesAccess::compute(),
        );
        let pipeline_layout = crate::pipeline_layout::PipelineLayout::new(
            device.clone(),
            &[ds_layout],
            &[push_range],
            Some("skin_channels_layout"),
        )?;
        let shader = crate::shaders::compute_shader::ComputeShader::new(device.clone(), CHANNELS_SPV)?;
        let pipeline = crate::compute_pipeline::ComputePipeline::new(
            None,
            pipeline_layout.clone(),
            (shader, None),
            Some("skin_channels_pipeline"),
        )?;

        cmd_buffer.record_one_time_submit(|rec| {
            rec.bind_compute_pipeline(pipeline.clone());
            rec.bind_descriptor_sets_for_compute_pipeline(
                pipeline_layout.clone(),
                0,
                &[desc_set.clone()],
            );
            let mut push = [0u8; 8];
            push[4..8].copy_from_slice(&1u32.to_le_bytes());
            rec.push_constant(
                pipeline_layout.clone(),
                crate::shader_stage_access::ShaderStagesAccess::compute(),
                0,
                &push,
            );
            rec.dispatch(1, 1, 1);
        })?;

        let fence = crate::fence::Fence::new(device.clone(), false, Some("skin_fence"))?;
        let cbs: Vec<Arc<dyn crate::command_buffer::CommandBufferTrait>> = vec![cmd_buffer.clone()];
        let waiter = queue.submit(cbs.as_slice(), &[], &[], fence.clone())?;
        drop(waiter);

        let map = crate::memory_pool::MemoryMap::new(allocated[1].get_backing_memory_pool())?;
        let range = map.range::<f32>(allocated[1].clone() as Arc<dyn MemoryPoolBacked>)?;
        let gpu = &range.as_slice()[..16];
        let cpu = cpu_bind_pose(&original, &armature);
        for i in 0..16 {
            assert!(
                (gpu[i] - cpu[i]).abs() < 1e-4,
                "matrix mismatch at {i}: gpu={} cpu={}",
                gpu[i],
                cpu[i]
            );
        }

        Ok(())
    }
}

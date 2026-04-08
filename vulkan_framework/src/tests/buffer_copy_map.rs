use std::{mem::size_of, sync::Arc};

use crate::memory_management::MemoryManagerTrait;
use crate::memory_pool::MemoryPoolBacked;
use crate::prelude::*;

#[test]
fn test_buffer_copy_and_map() -> VulkanResult<()> {
    match crate::tests::common::setup_test_device() {
        Ok((_instance, device)) => {
            // Prepare queue/family and command pool + buffer allocation helper
            let queue_family = crate::queue_family::QueueFamily::new(device.clone(), 0)?;
            let command_pool =
                crate::command_pool::CommandPool::new(queue_family.clone(), Some("test_pool"))?;
            let cmd_buffer = crate::command_buffer::PrimaryCommandBuffer::new(
                command_pool.clone(),
                Some("test_cb"),
            )?;
            let queue = crate::queue::Queue::new(queue_family.clone(), Some("test_queue"))?;

            let element_count = 16u32;
            let element_size = size_of::<u32>() as u64;
            let buffer_size = (element_count as u64) * element_size;

            // Create unallocated buffers
            let src_unalloc = crate::buffer::Buffer::new(
                device.clone(),
                crate::buffer::ConcreteBufferDescriptor::new(
                    crate::buffer::BufferUsage::from(
                        (ash::vk::BufferUsageFlags::TRANSFER_SRC
                            | ash::vk::BufferUsageFlags::TRANSFER_DST)
                            .as_raw(),
                    ),
                    buffer_size,
                ),
                None,
                Some("src"),
            )?;

            let dst_unalloc = crate::buffer::Buffer::new(
                device.clone(),
                crate::buffer::ConcreteBufferDescriptor::new(
                    crate::buffer::BufferUsage::from(
                        (ash::vk::BufferUsageFlags::TRANSFER_SRC
                            | ash::vk::BufferUsageFlags::TRANSFER_DST)
                            .as_raw(),
                    ),
                    buffer_size,
                ),
                None,
                Some("dst"),
            )?;

            // Allocate them on host visible coherent memory for simplicity
            let mut mem_mgr = crate::memory_management::DefaultMemoryManager::new(device.clone());

            let allocations = mem_mgr.allocate_resources(
                &crate::memory_heap::MemoryType::host_visible_and_coherent(),
                &crate::memory_pool::MemoryPoolFeatures::new(false),
                vec![src_unalloc.into(), dst_unalloc.into()],
                crate::memory_management::MemoryManagementTags::default(),
            )?;

            let src = allocations[0].buffer();
            let dst = allocations[1].buffer();

            // Fill src via mapped memory
            {
                let mem_map = crate::memory_pool::MemoryMap::new(src.get_backing_memory_pool())?;
                let mut range = mem_map
                    .range::<u32>(src.clone() as Arc<dyn crate::memory_pool::MemoryPoolBacked>)?;
                let slice = range.as_mut_slice();
                for i in 0..element_count as usize {
                    slice[i] = i as u32;
                }
            }

            // Record copy
            cmd_buffer.record_one_time_submit(|rec| {
                rec.copy_buffer(
                    src.clone() as Arc<dyn crate::buffer::BufferTrait>,
                    dst.clone() as Arc<dyn crate::buffer::BufferTrait>,
                    &[(0, 0, buffer_size)],
                );
            })?;

            // Submit and wait
            let fence = crate::fence::Fence::new(device.clone(), false, Some("test_fence"))?;
            let cbs: Vec<Arc<dyn crate::command_buffer::CommandBufferTrait>> =
                vec![cmd_buffer.clone()];
            let waiter = queue.submit(cbs.as_slice(), &[], &[], fence.clone())?;
            drop(waiter);

            // Verify dst contents
            {
                let mem_map = crate::memory_pool::MemoryMap::new(dst.get_backing_memory_pool())?;
                let range = mem_map
                    .range::<u32>(dst.clone() as Arc<dyn crate::memory_pool::MemoryPoolBacked>)?;
                let slice = range.as_slice();
                for i in 0..element_count as usize {
                    assert_eq!(slice[i], i as u32);
                }
            }

            Ok(())
        }
        Err(err) => {
            eprintln!("Skipping test_buffer_copy_and_map: {}", err);
            Ok(())
        }
    }
}

use crate::memory_management::MemoryManagerTrait;
use crate::memory_pool::MemoryPoolBacked;
use std::sync::Arc;

use crate::pipeline_stage::{PipelineStage, PipelineStages};
use crate::prelude::*;
use crate::queue::{SemaphoreSignalOp, SemaphoreWaitOp};

/// Submits multiple chained batches using a timeline semaphore: batch K waits
/// on the payload value K and signals the payload value K+1 in the same
/// submission, possibly on two different queues.
///
/// This is the pattern used by the engine to reuse the global illumination
/// data between consecutive frames. Unlike a binary semaphore, a timeline
/// semaphore can bootstrap the chain with a wait on its initial payload.
#[test]
fn test_timeline_semaphore_chains_submissions() -> VulkanResult<()> {
    match crate::tests::common::setup_test_device_with_queue_count(2) {
        Ok((_instance, device)) => {
            let queue_family = crate::queue_family::QueueFamily::new(device.clone(), 0)?;

            // use two distinct queues when the family exposes more than one,
            // falling back to a shared queue otherwise
            let first_queue =
                crate::queue::Queue::new(queue_family.clone(), Some("timeline_queue_0"))?;
            let second_queue =
                match crate::queue::Queue::new(queue_family.clone(), Some("timeline_queue_1")) {
                    Ok(queue) => queue,
                    Err(VulkanError::Framework(FrameworkError::TooManyQueues(_, _))) => {
                        first_queue.clone()
                    }
                    Err(err) => return Err(err),
                };

            let command_pool = crate::command_pool::CommandPool::new(
                queue_family.clone(),
                Some("timeline_test_pool"),
            )?;

            let timeline = crate::semaphore::Semaphore::new_timeline(
                device.clone(),
                0,
                Some("test_timeline"),
            )?;

            let unallocated = crate::buffer::Buffer::new(
                device.clone(),
                crate::buffer::ConcreteBufferDescriptor::new(
                    crate::buffer::BufferUsage::from(
                        ash::vk::BufferUsageFlags::TRANSFER_DST.as_raw(),
                    ),
                    64,
                ),
                None,
                Some("timeline_chain_data"),
            )?;
            let mut memory = crate::memory_management::DefaultMemoryManager::new(device.clone());
            let allocations = memory.allocate_resources(
                &crate::memory_heap::MemoryType::host_visible_and_coherent(),
                &crate::memory_pool::MemoryPoolFeatures::new(false),
                vec![unallocated.into()],
                crate::memory_management::MemoryManagementTags::default(),
            )?;
            let buffer = allocations[0].buffer();

            let total_submissions = 8u64;
            let mut waiters = Vec::new();

            for frame in 0..total_submissions {
                let cmd_buffer = crate::command_buffer::PrimaryCommandBuffer::new(
                    command_pool.clone(),
                    Some(format!("timeline_chain_cb[{frame}]").as_str()),
                )?;

                cmd_buffer.record_one_time_submit(|rec| {
                    rec.fill_buffer(buffer.clone(), 0, 64, (frame + 1) as u32);
                })?;

                let fence = crate::fence::Fence::new(
                    device.clone(),
                    false,
                    Some(format!("timeline_chain_fence[{frame}]").as_str()),
                )?;

                // alternate between the queues when more than one is available
                let submit_queue = match frame % 2 {
                    0 => first_queue.clone(),
                    _ => second_queue.clone(),
                };

                let cbs: Vec<Arc<dyn crate::command_buffer::CommandBufferTrait>> =
                    vec![cmd_buffer.clone()];

                let wait_stages: PipelineStages = [PipelineStage::AllCommands].as_slice().into();
                let waits = [SemaphoreWaitOp::Timeline(
                    wait_stages,
                    timeline.clone(),
                    frame,
                )];
                let signals = [SemaphoreSignalOp::Timeline(timeline.clone(), frame + 1)];

                waiters.push(submit_queue.submit_mixed(
                    cbs.as_slice(),
                    waits.as_slice(),
                    signals.as_slice(),
                    fence,
                )?);
            }

            let semaphores = [timeline.ash_handle()];
            let values = [total_submissions];
            let info = ash::vk::SemaphoreWaitInfo::default()
                .semaphores(&semaphores)
                .values(&values);
            unsafe { device.ash_handle().wait_semaphores(&info, 5_000_000_000) }?;
            drop(waiters);
            let mapping = crate::memory_pool::MemoryMap::new(buffer.get_backing_memory_pool())?;
            let range = mapping.range::<u32>(buffer.clone() as Arc<dyn MemoryPoolBacked>)?;
            assert!(range
                .as_slice()
                .iter()
                .all(|value| *value == total_submissions as u32));
            assert_eq!(
                unsafe {
                    device
                        .ash_handle()
                        .get_semaphore_counter_value(timeline.ash_handle())
                }?,
                total_submissions
            );

            println!("Timeline semaphore chained submissions completed");
            Ok(())
        }
        Err(err) => {
            eprintln!(
                "Skipping test_timeline_semaphore_chains_submissions: {}",
                err
            );
            Ok(())
        }
    }
}

#[cfg(test)]
mod submit_fence_tests {
    use std::{sync::Arc, time::Duration};

    use crate::prelude::*;

    #[test]
    fn test_empty_submit_and_fence() -> VulkanResult<()> {
        match crate::tests::common::setup_test_device() {
            Ok((_instance, device)) => {
                // Build a queue family/command pool/command buffer and submit a no-op
                let queue_family = crate::queue_family::QueueFamily::new(device.clone(), 0)?;

                let command_pool = crate::command_pool::CommandPool::new(queue_family.clone(), Some("test_pool"))?;

                let cmd_buffer = crate::command_buffer::PrimaryCommandBuffer::new(command_pool.clone(), Some("test_cb"))?;

                // Record a one-time submit with no commands
                cmd_buffer.record_one_time_submit(|_rec| {
                    // intentionally empty
                })?;

                let queue = crate::queue::Queue::new(queue_family.clone(), Some("test_queue"))?;

                let fence = crate::fence::Fence::new(device.clone(), false, Some("test_fence"))?;

                // Prepare command buffer trait-object slice
                let cbs: Vec<Arc<dyn crate::command_buffer::CommandBufferTrait>> = vec![cmd_buffer.clone()];

                // Capture the returned FenceWaiter so its Drop waits for completion.
                let fence_waiter = queue.submit(cbs.as_slice(), &[], &[], fence.clone())?;
                drop(fence_waiter);

                println!("Submit and fence wait completed");
                Ok(())
            }
            Err(err) => {
                eprintln!("Skipping test_empty_submit_and_fence: {}", err);
                Ok(())
            }
        }
    }
}

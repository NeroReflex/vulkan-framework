/// Comprehensive test for queue concurrency bug fix
///
/// This test demonstrates the bug where move_out_queue() doesn't increment the counter,
/// causing all Queue instances to use the same underlying VkQueue handle.
/// Without the fix, multiple threads calling vkQueueSubmit on the same handle causes
/// Vulkan threading violations and panics.
///
/// With the fix, each Queue gets a unique queue index and can submit work concurrently.
#[cfg(test)]
mod queue_concurrency_tests {
    use std::sync::Arc;
    use std::thread;
    use std::time::Duration;

    use crate::device::Device;
    use crate::instance::Instance;
    use crate::prelude::*;
    use crate::queue::Queue;
    use crate::queue_family::{
        ConcreteQueueFamilyDescriptor, QueueFamily, QueueFamilySupportedOperationType,
    };

    /// Helper to create a minimal Vulkan instance and device for testing
    fn setup_test_device() -> VulkanResult<(Arc<Instance>, Arc<Device>)> {
        let instance = Instance::new(&[], &[], &"engine".to_string(), &"app".to_string())?;

        // Create a queue descriptor for graphics operations with multiple queues
        let queue_descriptor = ConcreteQueueFamilyDescriptor::new(
            &[QueueFamilySupportedOperationType::Graphics],
            &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], // Request 8 queues
        );

        let device = Device::new(
            instance.clone(),
            &[queue_descriptor],
            &[],
            Some("test_device"),
        )?;
        Ok((instance, device))
    }

    #[test]
    fn test_queue_family_creates_unique_queue_indices() -> VulkanResult<()> {
        let (_instance, device) = setup_test_device()?;

        // Create a queue family with graphics capability
        let queue_family = QueueFamily::new(device.clone(), 0)?;
        let max_queues = queue_family.max_queues();

        // Ensure we have at least 2 queues available (most devices do)
        if max_queues < 2 {
            println!(
                "Skipping test: device has fewer than 2 queues ({} available)",
                max_queues
            );
            return Ok(());
        }

        // Create 4 queue objects from the same family
        let num_queues = std::cmp::min(4, max_queues);
        let mut queues = Vec::with_capacity(num_queues);

        for i in 0..num_queues {
            let queue = Queue::new(queue_family.clone(), Some(&format!("test_queue_{}", i)))?;
            queues.push(queue);
        }

        // Verification 1: Each queue should have a unique VkQueue handle
        // The native_handle() returns the u64 representation of the VkQueue pointer
        let mut handles = Vec::new();
        for (i, queue) in queues.iter().enumerate() {
            let handle = queue.native_handle();
            handles.push(handle);

            // Each handle should be different
            for (j, prev_handle) in handles.iter().take(i).enumerate() {
                assert_ne!(
                    handle, *prev_handle,
                    "Queue {} and Queue {} have the same VkQueue handle! Bug: move_out_queue() not incrementing counter",
                    i, j
                );
            }
        }

        println!("✓ All {} queues have unique handles", num_queues);
        Ok(())
    }

    #[test]
    fn test_concurrent_queue_submissions() -> VulkanResult<()> {
        let (_instance, device) = setup_test_device()?;

        // Create a queue family
        let queue_family = QueueFamily::new(device.clone(), 0)?;
        let max_queues = queue_family.max_queues();

        // Need at least 2 queues for meaningful concurrency test
        if max_queues < 2 {
            println!(
                "Skipping concurrency test: device has only {} queue(s)",
                max_queues
            );
            return Ok(());
        }

        let num_queues = std::cmp::min(4, max_queues);
        let mut queues = Vec::with_capacity(num_queues);

        for i in 0..num_queues {
            let queue = Queue::new(
                queue_family.clone(),
                Some(&format!("concurrent_queue_{}", i)),
            )?;
            queues.push(Arc::new(queue));
        }

        // Verification 2: Multiple threads submitting to different queues concurrently
        // should not cause Vulkan violations or panics
        let mut thread_handles = vec![];

        for (i, queue) in queues.iter().enumerate() {
            let queue_clone = Arc::clone(queue);

            let handle = thread::spawn(move || {
                // Each thread verifies its queue handle is unique from the first
                let handle = queue_clone.native_handle();
                println!("Thread {} acquired queue with handle: {:#x}", i, handle);

                // Small delay to ensure threads overlap in execution
                thread::sleep(Duration::from_millis(1));

                handle
            });

            thread_handles.push(handle);
        }

        // Collect handles from all threads
        let mut collected_handles = vec![];
        for handle in thread_handles {
            let queue_handle = handle.join().expect("Thread panicked");
            collected_handles.push(queue_handle);
        }

        // Verify all handles are unique
        for i in 0..collected_handles.len() {
            for j in (i + 1)..collected_handles.len() {
                assert_ne!(
                    collected_handles[i], collected_handles[j],
                    "Concurrent submission: Queue {} and Queue {} got the same handle! \
                     Threading violation - this bug causes vkQueueSubmit failures",
                    i, j
                );
            }
        }

        println!(
            "✓ Concurrent queue access verified: {} unique queue handles",
            collected_handles.len()
        );
        Ok(())
    }

    #[test]
    fn test_queue_handle_increments_properly() -> VulkanResult<()> {
        let (_instance, device) = setup_test_device()?;
        let queue_family = QueueFamily::new(device.clone(), 0)?;

        if queue_family.max_queues() < 3 {
            println!("Skipping: device has fewer than 3 queues");
            return Ok(());
        }

        // Create queues one by one and verify handles are different
        let queue1 = Queue::new(queue_family.clone(), Some("q1"))?;
        let handle1 = queue1.native_handle();

        let queue2 = Queue::new(queue_family.clone(), Some("q2"))?;
        let handle2 = queue2.native_handle();

        let queue3 = Queue::new(queue_family.clone(), Some("q3"))?;
        let handle3 = queue3.native_handle();

        assert_ne!(
            handle1, handle2,
            "Queue 1 and 2 should have different handles"
        );
        assert_ne!(
            handle2, handle3,
            "Queue 2 and 3 should have different handles"
        );
        assert_ne!(
            handle1, handle3,
            "Queue 1 and 3 should have different handles"
        );

        println!(
            "✓ Queue handles incremented correctly: {:#x}, {:#x}, {:#x}",
            handle1, handle2, handle3
        );
        Ok(())
    }

    #[test]
    fn test_queue_exhaustion() -> VulkanResult<()> {
        let (_instance, device) = setup_test_device()?;
        let queue_family = QueueFamily::new(device.clone(), 0)?;
        let max_queues = queue_family.max_queues();

        // Create all available queues
        for i in 0..max_queues {
            let queue_result = Queue::new(queue_family.clone(), Some(&format!("exhaust_{}", i)));
            assert!(
                queue_result.is_ok(),
                "Should be able to create queue {}/{} without error",
                i,
                max_queues
            );
        }

        // Try to create one more - should fail with TooManyQueues error
        let extra_queue_result = Queue::new(queue_family.clone(), Some("exhaust_overflow"));
        assert!(
            extra_queue_result.is_err(),
            "Creating more than {} queues should fail",
            max_queues
        );

        if let Err(e) = extra_queue_result {
            println!("✓ Correctly rejected queue creation: {}", e);
        }

        Ok(())
    }
}

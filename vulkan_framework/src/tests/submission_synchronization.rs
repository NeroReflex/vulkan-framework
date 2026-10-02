use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Barrier,
};
use std::time::Duration;

use crate::{
    command_buffer::{CommandBufferTrait, PrimaryCommandBuffer, SubmittableCommandBufferTrait},
    command_pool::{CommandPool, CommandPoolOwned},
    device::Device,
    fence::{Fence, FenceWaitFor},
    pipeline_stage::{PipelineStage, PipelineStages},
    prelude::{FrameworkError, VulkanError, VulkanResult},
    queue::{Queue, SemaphoreSignalOp, SemaphoreWaitOp},
    queue_family::QueueFamily,
    semaphore::Semaphore,
};
use ash::vk::Handle;

#[derive(Default)]
struct MockCommandBuffer {
    running: AtomicBool,
    completed: AtomicBool,
    reject: bool,
}

impl SubmittableCommandBufferTrait for MockCommandBuffer {
    fn mark_execution_begin(&self) -> VulkanResult<()> {
        if self.reject || self.running.swap(true, Ordering::SeqCst) {
            return Err(FrameworkError::CommandBufferSubmitAlreadyRunning.into());
        }
        Ok(())
    }
    fn mark_execution_complete(&self) -> VulkanResult<()> {
        self.completed.store(true, Ordering::SeqCst);
        self.running.store(false, Ordering::SeqCst);
        Ok(())
    }
    fn mark_execution_cancel(&self) -> VulkanResult<()> {
        self.running.store(false, Ordering::SeqCst);
        Ok(())
    }
}
impl CommandPoolOwned for MockCommandBuffer {
    fn get_parent_command_pool(&self) -> Arc<CommandPool> {
        unreachable!("reservation tests do not access Vulkan command pools")
    }
}
impl CommandBufferTrait for MockCommandBuffer {
    fn native_handle(&self) -> u64 {
        0
    }
}

#[test]
fn submit2_semaphore_stages_and_values() {
    let semaphore = ash::vk::Semaphore::from_raw(1);
    let stages: PipelineStages = [PipelineStage::Transfer, PipelineStage::ComputeShader]
        .as_slice()
        .into();
    let wait = Queue::semaphore_submit_info(semaphore, stages.into(), 17);
    assert_eq!(
        wait.stage_mask,
        ash::vk::PipelineStageFlags2::TRANSFER | ash::vk::PipelineStageFlags2::COMPUTE_SHADER
    );
    assert_eq!(wait.value, 17);
    for value in [0, 18] {
        let signal = Queue::signal_submit_info(semaphore, value);
        assert_eq!(
            signal.stage_mask,
            ash::vk::PipelineStageFlags2::ALL_COMMANDS
        );
        assert_eq!(signal.value, value);
        assert_eq!(signal.semaphore, semaphore);
    }
}

#[test]
fn marking_failure_cancels_only_the_reserved_prefix() {
    let first = Arc::new(MockCommandBuffer::default());
    let failing = Arc::new(MockCommandBuffer {
        reject: true,
        ..Default::default()
    });
    let last = Arc::new(MockCommandBuffer::default());
    let command_buffers: Vec<Arc<dyn CommandBufferTrait>> =
        vec![first.clone(), failing, last.clone()];
    assert!(Queue::submit_marked(&command_buffers, || panic!(
        "must not submit after a marking error"
    ))
    .is_err());
    assert!(!first.running.load(Ordering::SeqCst));
    assert!(!first.completed.load(Ordering::SeqCst));
    assert!(!last.running.load(Ordering::SeqCst));
}

#[test]
fn driver_failure_cancels_all_reserved_buffers_without_completion() {
    let first = Arc::new(MockCommandBuffer::default());
    let second = Arc::new(MockCommandBuffer::default());
    let command_buffers: Vec<Arc<dyn CommandBufferTrait>> = vec![first.clone(), second.clone()];
    let result = Queue::submit_marked(&command_buffers, || {
        Err(ash::vk::Result::ERROR_OUT_OF_HOST_MEMORY.into())
    });
    assert!(matches!(
        result,
        Err(VulkanError::Vulkan(
            ash::vk::Result::ERROR_OUT_OF_HOST_MEMORY
        ))
    ));
    for cb in [&first, &second] {
        assert!(!cb.running.load(Ordering::SeqCst));
        assert!(!cb.completed.load(Ordering::SeqCst));
    }
}

#[test]
fn duplicate_command_buffer_reservation_rolls_back() {
    let cb = Arc::new(MockCommandBuffer::default());
    let command_buffers: Vec<Arc<dyn CommandBufferTrait>> = vec![cb.clone(), cb.clone()];
    assert!(Queue::mark_command_buffers_as_running(&command_buffers).is_err());
    assert!(!cb.running.load(Ordering::SeqCst));
    assert!(!cb.completed.load(Ordering::SeqCst));
}

#[test]
fn queue_priorities_are_clamped_and_owned() -> VulkanResult<()> {
    let priorities = vec![1.0, 0.75, 0.5, 0.25];
    let selected = Device::select_queue_priorities(&priorities, 2)?;
    drop(priorities);
    assert_eq!(selected, [1.0, 0.75]);
    assert_eq!(Device::select_queue_priorities(&[0.5], 8)?, [0.5]);
    for priorities in [&[][..], &[-0.1][..], &[1.1][..], &[f32::NAN][..]] {
        assert!(Device::select_queue_priorities(priorities, 1).is_err());
    }
    assert!(Device::select_queue_priorities(&[1.0], 0).is_err());
    Ok(())
}

#[test]
fn requested_queue_count_is_clamped_to_the_physical_family() -> VulkanResult<()> {
    use crate::instance::InstanceOwned;
    let (_instance, device) = super::common::setup_test_device_with_queue_count(1024)?;
    let family = QueueFamily::new(device.clone(), 0)?;
    let properties = unsafe {
        device
            .get_parent_instance()
            .ash_handle()
            .get_physical_device_queue_family_properties(device.physical_device)
    };
    let available = properties[family.get_family_index() as usize].queue_count as usize;
    assert_eq!(family.max_queues(), available.min(1024));
    assert!(family.max_queues() > 0);
    for _ in 0..family.max_queues() {
        Queue::new(family.clone(), None)?;
    }
    assert!(matches!(
        Queue::new(family, None),
        Err(VulkanError::Framework(FrameworkError::TooManyQueues(_, _)))
    ));
    Ok(())
}

fn setup_queue() -> VulkanResult<(Arc<Device>, Arc<QueueFamily>, Arc<Queue>)> {
    let (_instance, device) = super::common::setup_test_device()?;
    let family = QueueFamily::new(device.clone(), 0)?;
    let queue = Queue::new(family.clone(), None)?;
    Ok((device, family, queue))
}

#[test]
fn one_time_buffers_survive_marking_and_driver_failures() -> VulkanResult<()> {
    let (device, family, queue) = setup_queue()?;
    let pool = CommandPool::new(family, None)?;
    let ready = PrimaryCommandBuffer::new(pool.clone(), None)?;
    ready.record_one_time_submit(|_| {})?;
    let unrecorded = PrimaryCommandBuffer::new(pool, None)?;
    let fence = Fence::new(device, false, None)?;
    let bad_batch: Vec<Arc<dyn CommandBufferTrait>> = vec![ready.clone(), unrecorded];
    assert!(matches!(
        queue.submit(&bad_batch, &[], &[], fence.clone()),
        Err(VulkanError::Framework(
            FrameworkError::CommandBufferSubmitNoCommands
        ))
    ));
    fence.reset()?;
    let good_batch: Vec<Arc<dyn CommandBufferTrait>> = vec![ready.clone()];
    assert!(Queue::submit_marked(&good_batch, || Err(
        ash::vk::Result::ERROR_OUT_OF_HOST_MEMORY.into()
    ))
    .is_err());
    let waiter = queue.submit(&good_batch, &[], &[], fence)?;
    drop(waiter);
    assert!(matches!(
        ready.mark_execution_begin(),
        Err(VulkanError::Framework(
            FrameworkError::CommandBufferSubmitNoCommands
        ))
    ));
    Ok(())
}

#[test]
fn fence_is_owned_until_waiter_drop_and_then_reusable() -> VulkanResult<()> {
    let (device, _, queue) = setup_queue()?;
    let fence = Fence::new(device, true, None)?;
    assert!(queue.submit(&[], &[], &[], fence.clone()).is_err());
    fence.reset()?;
    let waiter = queue.submit(&[], &[], &[], fence.clone())?;
    Fence::wait_for_fences(&[fence.clone()], FenceWaitFor::All, Duration::from_secs(5))?;
    assert!(waiter.complete()?);
    assert!(fence.reset().is_err());
    assert!(Fence::reset_fences(&[fence.clone()]).is_err());
    assert!(queue.submit(&[], &[], &[], fence.clone()).is_err());
    drop(waiter);
    assert!(!fence.is_signaled()?);
    Fence::reset_fences(&[fence.clone(), fence.clone()])?;
    drop(queue.submit(&[], &[], &[], fence)?);
    Ok(())
}

#[test]
fn semaphore_types_and_timeline_values_are_checked_before_submit() -> VulkanResult<()> {
    let (device, _, queue) = setup_queue()?;
    let binary = Semaphore::new(device.clone(), None)?;
    let timeline = Semaphore::new_timeline(device.clone(), 7, None)?;
    assert!(!binary.is_timeline());
    assert!(timeline.is_timeline());
    let fence = Fence::new(device, false, None)?;
    let stages: PipelineStages = [PipelineStage::AllCommands].as_slice().into();
    assert!(queue
        .submit_mixed(
            &[],
            &[SemaphoreWaitOp::Timeline(stages, binary.clone(), 0)],
            &[],
            fence.clone()
        )
        .is_err());
    assert!(queue
        .submit_mixed(
            &[],
            &[],
            &[SemaphoreSignalOp::Timeline(binary, 1)],
            fence.clone()
        )
        .is_err());
    assert!(queue
        .submit(&[], &[(stages, timeline.clone())], &[], fence.clone())
        .is_err());
    assert!(queue
        .submit(&[], &[], &[timeline.clone()], fence.clone())
        .is_err());
    for value in [6, 7] {
        assert!(queue
            .submit_mixed(
                &[],
                &[],
                &[SemaphoreSignalOp::Timeline(timeline.clone(), value)],
                fence.clone()
            )
            .is_err());
    }
    assert!(queue
        .submit_mixed(
            &[],
            &[SemaphoreWaitOp::Timeline(stages, timeline.clone(), 8)],
            &[SemaphoreSignalOp::Timeline(timeline.clone(), 8)],
            fence.clone()
        )
        .is_err());
    drop(queue.submit_mixed(
        &[],
        &[SemaphoreWaitOp::Timeline(stages, timeline.clone(), 7)],
        &[SemaphoreSignalOp::Timeline(timeline, 8)],
        fence,
    )?);
    Ok(())
}

#[test]
fn binary_wait_signal_chain_uses_a_previously_submitted_signal() -> VulkanResult<()> {
    let (device, _, queue) = setup_queue()?;
    let semaphore = Semaphore::new(device.clone(), None)?;
    let stages: PipelineStages = [PipelineStage::AllCommands].as_slice().into();
    let first = queue.submit(
        &[],
        &[],
        &[semaphore.clone()],
        Fence::new(device.clone(), false, None)?,
    )?;
    let second = queue.submit(
        &[],
        &[(stages, semaphore.clone())],
        &[semaphore.clone()],
        Fence::new(device.clone(), false, None)?,
    )?;
    let third = queue.submit(
        &[],
        &[(stages, semaphore)],
        &[],
        Fence::new(device, false, None)?,
    )?;
    drop(third);
    drop(second);
    drop(first);
    Ok(())
}

#[test]
fn aliased_queue_submits_are_externally_synchronized() -> VulkanResult<()> {
    let (device, _, queue) = setup_queue()?;
    let barrier = Arc::new(Barrier::new(8));
    let mut threads = Vec::new();
    for _ in 0..8 {
        let queue = queue.clone();
        let device = device.clone();
        let barrier = barrier.clone();
        threads.push(std::thread::spawn(move || -> VulkanResult<()> {
            let fence = Fence::new(device, false, None)?;
            barrier.wait();
            for _ in 0..16 {
                drop(queue.submit(&[], &[], &[], fence.clone())?);
            }
            Ok(())
        }));
    }
    for _ in 0..16 {
        device.wait_idle()?;
    }
    for thread in threads {
        thread.join().expect("queue submission thread panicked")?;
    }
    device.wait_idle()?;
    Ok(())
}

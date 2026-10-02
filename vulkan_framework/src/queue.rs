use ash::vk::Handle;
use smallvec::SmallVec;

use crate::{
    command_buffer::CommandBufferTrait,
    device::DeviceOwned,
    fence::{Fence, FenceWaiter},
    pipeline_stage::PipelineStages,
    prelude::{VulkanError, VulkanResult},
    queue_family::*,
    semaphore::Semaphore,
};

use std::sync::{Arc, Mutex, MutexGuard};

/// A semaphore wait operation to be performed as part of a submission.
///
/// Binary semaphores wait until they reach the signaled state, while timeline
/// semaphores wait until their payload reaches the given value.
pub enum SemaphoreWaitOp {
    Binary(PipelineStages, Arc<Semaphore>),
    Timeline(PipelineStages, Arc<Semaphore>, u64),
}

/// A semaphore signal operation to be performed as part of a submission.
///
/// Binary semaphores are put in the signaled state, while the payload of
/// timeline semaphores is set to the given value.
pub enum SemaphoreSignalOp {
    Binary(Arc<Semaphore>),
    Timeline(Arc<Semaphore>, u64),
}

pub struct Queue {
    _name_bytes: Vec<u8>,
    queue_family: Arc<QueueFamily>,
    priority: f32,
    queue: ash::vk::Queue,
    host_access: Arc<Mutex<()>>,
}

impl QueueFamilyOwned for Queue {
    fn get_parent_queue_family(&self) -> Arc<QueueFamily> {
        self.queue_family.clone()
    }
}

impl Drop for Queue {
    fn drop(&mut self) {
        // Nothing to be done here, seems like queues are not to be deleted... A real shame!
    }
}

impl Queue {
    pub(crate) fn mark_command_buffers_as_running(
        command_buffers: &[Arc<dyn CommandBufferTrait>],
    ) -> VulkanResult<()> {
        // Mark every command buffer as "execution started": on the first failure
        // rollback the ones that have already been marked so that the caller
        // does not end up with a partially marked submission.
        for (index, command_buffer) in command_buffers.iter().enumerate() {
            if let Err(err) = command_buffer.mark_execution_begin() {
                for rollback_index in 0..index {
                    command_buffers[rollback_index].mark_execution_cancel()?;
                }

                return Err(err);
            }
        }

        Ok(())
    }

    /// Externally synchronize queue operations, including presentation, with submissions
    /// through aliases and device-wide idle waits. Distinct queues share this host lock;
    /// their GPU execution can still overlap.
    pub(crate) fn lock(&self) -> VulkanResult<MutexGuard<'_, ()>> {
        self.host_access
            .lock()
            .map_err(|err| crate::prelude::FrameworkError::MutexError(format!("{err}")).into())
    }

    pub(crate) fn submit_marked<F>(
        command_buffers: &[Arc<dyn CommandBufferTrait>],
        submit: F,
    ) -> VulkanResult<()>
    where
        F: FnOnce() -> VulkanResult<()>,
    {
        Self::mark_command_buffers_as_running(command_buffers)?;
        if let Err(err) = submit() {
            for command_buffer in command_buffers {
                command_buffer.mark_execution_cancel()?;
            }
            return Err(err);
        }
        Ok(())
    }

    pub(crate) fn semaphore_submit_info(
        semaphore: ash::vk::Semaphore,
        stage_mask: ash::vk::PipelineStageFlags2,
        value: u64,
    ) -> ash::vk::SemaphoreSubmitInfo<'static> {
        ash::vk::SemaphoreSubmitInfo::default()
            .semaphore(semaphore)
            .stage_mask(stage_mask)
            .value(value)
    }

    pub(crate) fn signal_submit_info(
        semaphore: ash::vk::Semaphore,
        value: u64,
    ) -> ash::vk::SemaphoreSubmitInfo<'static> {
        // The public signal API has no stage argument: cover all submitted work,
        // not NONE, which provides no execution or memory dependency for that work.
        Self::semaphore_submit_info(semaphore, ash::vk::PipelineStageFlags2::ALL_COMMANDS, value)
    }

    pub fn submit(
        &self,
        command_buffers: &[Arc<dyn CommandBufferTrait>],
        wait_semaphores: &[(PipelineStages, Arc<Semaphore>)],
        signal_semaphores: &[Arc<Semaphore>],
        fence: Arc<Fence>,
    ) -> VulkanResult<FenceWaiter> {
        let wait_ops: SmallVec<[SemaphoreWaitOp; 8]> = wait_semaphores
            .iter()
            .map(|(wait_cond, wait_sem)| {
                SemaphoreWaitOp::Binary(wait_cond.to_owned(), wait_sem.clone())
            })
            .collect();

        let signal_ops: SmallVec<[SemaphoreSignalOp; 8]> = signal_semaphores
            .iter()
            .map(|sem| SemaphoreSignalOp::Binary(sem.clone()))
            .collect();

        self.submit_mixed(
            command_buffers,
            wait_ops.as_slice(),
            signal_ops.as_slice(),
            fence,
        )
    }

    /**
     * Submits the given command buffers to this queue, waiting and signaling
     * both binary and timeline semaphores.
     *
     * Binary waits require a previously submitted signal operation. Timeline waits
     * can instead wait on the initial payload (e.g. K=0), then signal K+1 in the
     * same submission. Signals must increase in execution order across queues.
     *
     * Wait stages are preserved: callers must include the earliest stages accessing
     * the shared resources. Signals cover ALL_COMMANDS. Invalid semaphore kinds,
     * duplicate operations, timeline signal <= wait, and owned/signaled fences are
     * rejected with ERROR_UNKNOWN before submission.
     */
    pub fn submit_mixed(
        &self,
        command_buffers: &[Arc<dyn CommandBufferTrait>],
        wait_semaphores: &[SemaphoreWaitOp],
        signal_semaphores: &[SemaphoreSignalOp],
        fence: Arc<Fence>,
    ) -> VulkanResult<FenceWaiter> {
        let device = self.queue_family.get_parent_device();
        if device != fence.get_parent_device() {
            return Err(VulkanError::Framework(
                crate::prelude::FrameworkError::ResourceFromIncompatibleDevice,
            ));
        }

        for command_buffer in command_buffers {
            let family = command_buffer
                .get_parent_command_pool()
                .get_parent_queue_family();
            if family.get_parent_device() != device {
                return Err(crate::prelude::FrameworkError::ResourceFromIncompatibleDevice.into());
            }
            if family.get_family_index() != self.queue_family.get_family_index() {
                return Err(ash::vk::Result::ERROR_UNKNOWN.into());
            }
        }

        let mut wait_semaphore_infos = SmallVec::<[ash::vk::SemaphoreSubmitInfo; 8]>::new();
        for wait_op in wait_semaphores {
            let (stages, semaphore, value, timeline) = match wait_op {
                SemaphoreWaitOp::Binary(stages, sem) => (*stages, sem, 0, false),
                SemaphoreWaitOp::Timeline(stages, sem, value) => (*stages, sem, *value, true),
            };
            if semaphore.get_parent_device() != device {
                return Err(crate::prelude::FrameworkError::ResourceFromIncompatibleDevice.into());
            }
            let stages: ash::vk::PipelineStageFlags2 = stages.into();
            if semaphore.is_timeline() != timeline
                || stages.contains(ash::vk::PipelineStageFlags2::HOST)
                || wait_semaphore_infos
                    .iter()
                    .any(|info| info.semaphore == semaphore.ash_handle())
            {
                return Err(ash::vk::Result::ERROR_UNKNOWN.into());
            }
            wait_semaphore_infos.push(Self::semaphore_submit_info(
                semaphore.ash_handle(),
                stages,
                value,
            ));
        }

        let mut signal_semaphore_infos = SmallVec::<[ash::vk::SemaphoreSubmitInfo; 8]>::new();
        for signal_op in signal_semaphores {
            let (semaphore, value, timeline) = match signal_op {
                SemaphoreSignalOp::Binary(sem) => (sem, 0, false),
                SemaphoreSignalOp::Timeline(sem, value) => (sem, *value, true),
            };
            if semaphore.get_parent_device() != device {
                return Err(crate::prelude::FrameworkError::ResourceFromIncompatibleDevice.into());
            }
            if semaphore.is_timeline() != timeline
                || signal_semaphore_infos
                    .iter()
                    .any(|info| info.semaphore == semaphore.ash_handle())
                || (timeline
                    && wait_semaphore_infos.iter().any(|info| {
                        info.semaphore == semaphore.ash_handle() && value <= info.value
                    }))
            {
                return Err(ash::vk::Result::ERROR_UNKNOWN.into());
            }
            if timeline {
                let current_value = unsafe {
                    device
                        .ash_handle()
                        .get_semaphore_counter_value(semaphore.ash_handle())
                }?;
                if value <= current_value {
                    return Err(ash::vk::Result::ERROR_UNKNOWN.into());
                }
            }
            signal_semaphore_infos.push(Self::signal_submit_info(semaphore.ash_handle(), value));
        }

        let cmd_buffer_infos = command_buffers
            .iter()
            .map(|f| {
                ash::vk::CommandBufferSubmitInfo::default()
                    .command_buffer(ash::vk::CommandBuffer::from_raw(f.native_handle()))
            })
            .collect::<smallvec::SmallVec<[ash::vk::CommandBufferSubmitInfo; 4]>>();

        let submit_info = ash::vk::SubmitInfo2::default()
            .command_buffer_infos(cmd_buffer_infos.as_slice())
            .signal_semaphore_infos(signal_semaphore_infos.as_slice())
            .wait_semaphore_infos(wait_semaphore_infos.as_slice());

        let submits = [submit_info];

        let _guard = self.lock()?;
        fence.reserve_submission()?;
        if let Err(err) = Self::submit_marked(command_buffers, || {
            Ok(unsafe {
                device
                    .ash_handle()
                    .queue_submit2(self.ash_handle(), &submits, fence.ash_handle())
            }?)
        }) {
            fence.cancel_submission()?;
            return Err(err);
        }

        let mut used_semaphores: crate::fence::FenceWaiterSemaphoresType =
            smallvec::SmallVec::new();
        for wait_op in wait_semaphores.iter() {
            match wait_op {
                SemaphoreWaitOp::Binary(_, sem) => used_semaphores.push(sem.clone()),
                SemaphoreWaitOp::Timeline(_, sem, _) => used_semaphores.push(sem.clone()),
            }
        }
        for signal_op in signal_semaphores.iter() {
            match signal_op {
                SemaphoreSignalOp::Binary(sem) => used_semaphores.push(sem.clone()),
                SemaphoreSignalOp::Timeline(sem, _) => used_semaphores.push(sem.clone()),
            }
        }

        Ok(FenceWaiter::new(fence, command_buffers, used_semaphores))
    }

    #[inline]
    pub fn native_handle(&self) -> u64 {
        ash::vk::Handle::as_raw(self.queue)
    }

    #[inline]
    pub(crate) fn ash_handle(&self) -> ash::vk::Queue {
        self.queue
    }

    #[inline]
    pub fn get_priority(&self) -> f32 {
        self.priority
    }

    pub fn new(
        queue_family: Arc<QueueFamily>,
        debug_name: Option<&str>,
    ) -> VulkanResult<Arc<Self>> {
        match queue_family.move_out_queue() {
            Ok((queue_index, priority)) => {
                let queue = unsafe {
                    queue_family
                        .get_parent_device()
                        .ash_handle()
                        .get_device_queue(queue_family.get_family_index(), queue_index)
                };

                let mut obj_name_bytes = vec![];

                let device = queue_family.get_parent_device();
                let host_access = device.queue_host_access.clone();
                let _guard = device
                    .queue_host_access
                    .lock()
                    .map_err(|err| crate::prelude::FrameworkError::MutexError(format!("{err}")))?;

                if let Some(ext) = device.ash_ext_debug_utils_ext() {
                    if let Some(name) = debug_name {
                        for name_ch in name.as_bytes().iter() {
                            obj_name_bytes.push(*name_ch);
                        }
                        obj_name_bytes.push(0x00);

                        unsafe {
                            let object_name = std::ffi::CStr::from_bytes_with_nul_unchecked(
                                obj_name_bytes.as_slice(),
                            );
                            // set device name for debugging
                            let dbg_info = ash::vk::DebugUtilsObjectNameInfoEXT::default()
                                .object_handle(queue)
                                .object_name(object_name);

                            if let Err(err) = ext.set_debug_utils_object_name(&dbg_info) {
                                #[cfg(debug_assertions)]
                                {
                                    println!("Error setting the Debug name for the newly created Queue, will use handle. Error: {}", err);
                                }
                            }
                        }
                    }
                }

                Ok(Arc::new(Self {
                    _name_bytes: obj_name_bytes,
                    queue_family,
                    priority,
                    queue,
                    host_access,
                }))
            }
            Err(err) => Err(err),
        }
    }
}

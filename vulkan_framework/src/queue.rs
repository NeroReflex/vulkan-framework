use ash::vk::Handle;
use smallvec::{smallvec, SmallVec};

use crate::{
    command_buffer::CommandBufferTrait,
    device::DeviceOwned,
    fence::{Fence, FenceWaiter},
    pipeline_stage::PipelineStages,
    prelude::{VulkanError, VulkanResult},
    queue_family::*,
    semaphore::Semaphore,
};

use std::sync::Arc;

pub struct Queue {
    _name_bytes: Vec<u8>,
    queue_family: Arc<QueueFamily>,
    priority: f32,
    queue: ash::vk::Queue,
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
    fn mark_command_buffers_as_running(
        command_buffers: &[Arc<dyn CommandBufferTrait>],
    ) -> VulkanResult<()> {
        let mut marked = 1;
        let mut maybe_error = Option::None;
        while marked <= command_buffers.len() {
            match command_buffers[marked - 1].mark_execution_begin() {
                Ok(_) => marked += 1,
                Err(err) => maybe_error = Some(err),
            };
        }

        match maybe_error {
            None => Ok(()),
            Some(err) => {
                while marked > 0 {
                    command_buffers[marked].mark_execution_complete().unwrap();
                }

                Err(err)
            }
        }
    }

    #[deprecated]
    pub fn submit_no_features2(
        &self,
        command_buffers: &[Arc<dyn CommandBufferTrait>],
        wait_semaphores: &[(PipelineStages, Arc<Semaphore>)],
        signal_semaphores: &[Arc<Semaphore>],
        fence: Arc<Fence>,
    ) -> VulkanResult<FenceWaiter> {
        if self.get_parent_queue_family().get_parent_device() != fence.get_parent_device() {
            return Err(VulkanError::Framework(
                crate::prelude::FrameworkError::ResourceFromIncompatibleDevice,
            ));
        }

        // TODO: assert queue.device == command_buffers.device

        let mut wait_sems: SmallVec<[ash::vk::Semaphore; 8]> = smallvec![];
        let mut wait_stages: SmallVec<[ash::vk::PipelineStageFlags; 8]> = smallvec![];

        for pipeline_bubble in wait_semaphores.iter() {
            // TODO: assert f.device == self.device

            let (wait_cond, wait_sem) = pipeline_bubble;

            wait_sems.push(wait_sem.ash_handle());
            wait_stages.push(wait_cond.to_owned().into());
        }

        let native_signal_semaphores = signal_semaphores
            .iter()
            .map(|sem| {
                // TODO: check self.device == sem.device

                sem.ash_handle()
            })
            .collect::<smallvec::SmallVec<[ash::vk::Semaphore; 8]>>();

        let cmd_buffers = command_buffers
            .iter()
            .map(|f| {
                // TODO: assert f.device == self.device

                ash::vk::CommandBuffer::from_raw(f.native_handle())
            })
            .collect::<Vec<ash::vk::CommandBuffer>>();

        let submit_info = ash::vk::SubmitInfo::default()
            .command_buffers(cmd_buffers.as_slice())
            .signal_semaphores(native_signal_semaphores.as_slice())
            .wait_dst_stage_mask(wait_stages.as_slice())
            .wait_semaphores(wait_sems.as_slice());

        let submits = [submit_info];

        Self::mark_command_buffers_as_running(command_buffers)?;

        unsafe {
            self.get_parent_queue_family()
                .get_parent_device()
                .ash_handle()
                .queue_submit(self.ash_handle(), submits.as_slice(), fence.ash_handle())
        }?;

        let used_semaphores: crate::fence::FenceWaiterSemaphoresType = wait_semaphores
            .iter()
            .map(|(_, sem)| sem.clone())
            .chain(signal_semaphores.iter().cloned())
            .collect();

        Ok(FenceWaiter::new(fence, command_buffers, used_semaphores))
    }

    pub fn submit(
        &self,
        command_buffers: &[Arc<dyn CommandBufferTrait>],
        wait_semaphores: &[(PipelineStages, Arc<Semaphore>)],
        signal_semaphores: &[Arc<Semaphore>],
        fence: Arc<Fence>,
    ) -> VulkanResult<FenceWaiter> {
        if self.get_parent_queue_family().get_parent_device() != fence.get_parent_device() {
            return Err(VulkanError::Framework(
                crate::prelude::FrameworkError::ResourceFromIncompatibleDevice,
            ));
        }

        // TODO: assert queue.device == command_buffers.device

        let wait_semaphore_infos: SmallVec<[ash::vk::SemaphoreSubmitInfo; 8]> = wait_semaphores
            .iter()
            .map(|(wait_cond, wait_sem)| {
                ash::vk::SemaphoreSubmitInfo::default()
                    .semaphore(wait_sem.ash_handle())
                    .stage_mask(wait_cond.to_owned().into())
            })
            .collect();

        let signal_semaphore_infos = signal_semaphores
            .iter()
            .map(|sem| {
                // TODO: check self.device == sem.device

                ash::vk::SemaphoreSubmitInfo::default().semaphore(sem.ash_handle())
                //.stage_mask(todo!())
            })
            .collect::<smallvec::SmallVec<[ash::vk::SemaphoreSubmitInfo; 8]>>();

        let cmd_buffer_infos = command_buffers
            .iter()
            .map(|f| {
                // TODO: assert f.device == self.device

                ash::vk::CommandBufferSubmitInfo::default()
                    .command_buffer(ash::vk::CommandBuffer::from_raw(f.native_handle()))
            })
            .collect::<smallvec::SmallVec<[ash::vk::CommandBufferSubmitInfo; 4]>>();

        let submit_info = ash::vk::SubmitInfo2::default()
            .command_buffer_infos(cmd_buffer_infos.as_slice())
            .signal_semaphore_infos(signal_semaphore_infos.as_slice())
            .wait_semaphore_infos(wait_semaphore_infos.as_slice());

        let submits = [submit_info];

        Self::mark_command_buffers_as_running(command_buffers)?;

        unsafe {
            self.get_parent_queue_family()
                .get_parent_device()
                .ash_handle()
                .queue_submit2(self.ash_handle(), submits.as_slice(), fence.ash_handle())
        }?;

        let used_semaphores: crate::fence::FenceWaiterSemaphoresType = wait_semaphores
            .iter()
            .map(|(_, sem)| sem.clone())
            .chain(signal_semaphores.iter().cloned())
            .collect();

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
                }))
            }
            Err(err) => Err(err),
        }
    }
}

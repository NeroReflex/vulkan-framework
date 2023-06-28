use ash::vk::Handle;
use smallvec::{smallvec, SmallVec};

use crate::{
    command_buffer::CommandBufferTrait,
    device::DeviceOwned,
    fence::Fence,
    instance::InstanceOwned,
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
    pub fn submit_unchecked(
        &self,
        _command_buffers: &[Arc<dyn CommandBufferTrait>],
    ) -> VulkanResult<()> {
        // TODO: assert queue.device == self.device
        todo!()
    }

    pub fn submit(
        &self,
        command_buffers: &[Arc<dyn CommandBufferTrait>],
        wait_semaphores: &[(PipelineStages, Arc<Semaphore>)],
        signal_semaphores: &[Arc<Semaphore>],
        fence: Arc<Fence>,
    ) -> VulkanResult<()> {
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
            wait_stages.push(wait_cond.ash_flags());
        }

        let signal_semaphores = signal_semaphores
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

        let submit_info = ash::vk::SubmitInfo::builder()
            .command_buffers(cmd_buffers.as_slice())
            .signal_semaphores(signal_semaphores.as_slice())
            .wait_dst_stage_mask(wait_stages.as_slice())
            .wait_semaphores(wait_sems.as_slice())
            .build();

        let submits = [submit_info];

        match unsafe {
            self.get_parent_queue_family()
                .get_parent_device()
                .ash_handle()
                .queue_submit(self.ash_handle(), submits.as_slice(), fence.ash_handle())
        } {
            Ok(_) => Ok(()),
            Err(err) => Err(VulkanError::Vulkan(
                err.as_raw(),
                Some(format!(
                    "Error submitting command buffers to the current queue: {}",
                    err
                )),
            )),
        }
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
                let instance = device.get_parent_instance();

                if let Some(ext) = instance.get_debug_ext_extension() {
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
                            let dbg_info = ash::vk::DebugUtilsObjectNameInfoEXT::builder()
                                .object_type(ash::vk::ObjectType::QUEUE)
                                .object_handle(ash::vk::Handle::as_raw(queue))
                                .object_name(object_name)
                                .build();

                            if let Err(err) = ext.set_debug_utils_object_name(
                                device.ash_handle().handle(),
                                &dbg_info,
                            ) {
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

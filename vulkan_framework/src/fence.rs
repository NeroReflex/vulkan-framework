use std::{sync::Arc, time::Duration};

use crate::{
    command_buffer::CommandBufferTrait,
    device::{Device, DeviceOwned},
    instance::InstanceOwned,
    pipeline_stage::PipelineStages,
    prelude::{FrameworkError, VulkanError, VulkanResult},
    semaphore::Semaphore,
};

pub struct Fence {
    device: Arc<Device>,
    fence: ash::vk::Fence,
}

impl Drop for Fence {
    fn drop(&mut self) {
        unsafe {
            self.device.ash_handle().destroy_fence(
                self.fence,
                self.device.get_parent_instance().get_alloc_callbacks(),
            )
        }
    }
}

impl DeviceOwned for Fence {
    fn get_parent_device(&self) -> Arc<Device> {
        self.device.clone()
    }
}

pub enum FenceWaitFor {
    All,
    One,
}

impl Fence {
    #[inline]
    pub(crate) fn ash_handle(&self) -> ash::vk::Fence {
        self.fence
    }

    pub fn is_signaled(&self) -> VulkanResult<bool> {
        match unsafe {
            self.get_parent_device()
                .ash_handle()
                .get_fence_status(self.fence)
        } {
            Ok(status) => Ok(status),
            Err(err) => Err(VulkanError::Vulkan(
                err.as_raw(),
                Some(format!("Error reading fence status: {}", err)),
            )),
        }
    }

    pub fn reset(&self) -> VulkanResult<()> {
        match unsafe {
            self.get_parent_device()
                .ash_handle()
                .reset_fences(&[self.fence])
        } {
            Ok(()) => Ok(()),
            Err(err) => Err(VulkanError::Vulkan(err.as_raw(), None)),
        }
    }

    /**
     * This function accepts fences that have been created from the same device.
     */
    pub fn reset_fences(fences: &[Arc<Self>]) -> VulkanResult<()> {
        let mut device: Option<Arc<Device>> = None;
        let mut native_fences = Vec::<ash::vk::Fence>::new();
        for fence in fences {
            match &device {
                Some(dev) => {
                    if fence.get_parent_device().native_handle() != dev.native_handle() {
                        return Err(VulkanError::Framework(
                            FrameworkError::ResourceFromIncompatibleDevice,
                        ));
                    }
                }
                None => device = Some(fence.device.clone()),
            }

            native_fences.push(fence.fence)
        }

        match &device {
            Some(dev) => {
                let reset_result = unsafe { dev.ash_handle().reset_fences(native_fences.as_ref()) };

                match reset_result {
                    Ok(_) => Ok(()),
                    Err(err) => Err(VulkanError::Vulkan(err.as_raw(), None)),
                }
            }
            // list of fences are simply empty
            None => Ok(()),
        }
    }

    #[inline]
    pub fn native_handle(&self) -> u64 {
        ash::vk::Handle::as_raw(self.fence)
    }

    /**
     * This function accepts fences that have been created from the same device.
     */
    pub fn wait_for_fences(
        fences: &[Arc<Self>],
        wait_target: FenceWaitFor,
        device_timeout: Duration,
    ) -> VulkanResult<()> {
        let mut device: Option<Arc<Device>> = None;
        let mut native_fences = smallvec::SmallVec::<[ash::vk::Fence; 4]>::new();
        for fence in fences {
            match &device {
                Some(dev) => {
                    if fence.get_parent_device().native_handle() != dev.native_handle() {
                        return Err(VulkanError::Framework(
                            FrameworkError::ResourceFromIncompatibleDevice,
                        ));
                    }
                }
                None => device = Some(fence.get_parent_device()),
            }

            native_fences.push(fence.fence)
        }

        let timeout_ns = device_timeout.as_nanos();

        match &device {
            Some(dev) => {
                let wait_result = unsafe {
                    dev.ash_handle().wait_for_fences(
                        native_fences.as_ref(),
                        match wait_target {
                            FenceWaitFor::All => true,
                            FenceWaitFor::One => false,
                        },
                        if timeout_ns >= (u64::MAX as u128) {
                            u64::MAX
                        } else {
                            timeout_ns as u64
                        },
                    )
                };

                match wait_result {
                    Ok(_) => Ok(()),
                    Err(err) => Err(VulkanError::Vulkan(err.as_raw(), None)),
                }
            }
            // No fences to wait for
            None => Ok(()),
        }
    }

    pub fn new(
        device: Arc<Device>,
        starts_in_signaled_state: bool,
        debug_name: Option<&str>,
    ) -> VulkanResult<Arc<Self>> {
        let create_info =
            ash::vk::FenceCreateInfo::default().flags(match starts_in_signaled_state {
                true => ash::vk::FenceCreateFlags::SIGNALED,
                false => ash::vk::FenceCreateFlags::from_raw(0x00u32),
            });

        match unsafe {
            device.ash_handle().create_fence(
                &create_info,
                device.get_parent_instance().get_alloc_callbacks(),
            )
        } {
            Ok(fence) => {
                let mut obj_name_bytes = vec![];
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
                                .object_handle(fence)
                                .object_name(object_name);

                            if let Err(err) = ext.set_debug_utils_object_name(&dbg_info) {
                                #[cfg(debug_assertions)]
                                {
                                    println!("Error setting the Debug name for the newly created Queue, will use handle. Error: {}", err)
                                }
                            }
                        }
                    }
                }

                Ok(Arc::new(Self { device, fence }))
            }
            Err(err) => Err(VulkanError::Vulkan(
                err.as_raw(),
                Some(format!("Error creating the fence: {}", err)),
            )),
        }
    }
}

type FenceWaiterCommandBuffersType = smallvec::SmallVec<[Arc<dyn CommandBufferTrait>; 8]>;
pub(crate) type FenceWaiterSemaphoresType = smallvec::SmallVec<[Arc<Semaphore>; 32]>;

pub struct FenceWaiter {
    fence: Arc<Fence>,
    _command_buffers: FenceWaiterCommandBuffersType,
    _semaphores: FenceWaiterSemaphoresType,
}

impl Drop for FenceWaiter {
    fn drop(&mut self) {
        loop {
            match Fence::wait_for_fences(
                &[self.fence.clone()],
                FenceWaitFor::All,
                Duration::from_millis(u64::MAX),
            ) {
                Ok(_) => return self.fence.reset().unwrap(),
                Err(err) => match err.is_timeout() {
                    true => continue,
                    false => panic!("Error while waiting for fence"),
                },
            }
        }
    }
}

impl FenceWaiter {
    pub(crate) fn new(
        fence: Arc<Fence>,
        command_buffers: &[Arc<dyn CommandBufferTrait>],
        semaphores: FenceWaiterSemaphoresType,
    ) -> Self {
        let command_buffers = command_buffers.iter().cloned().collect();
        let semaphores = semaphores.iter().cloned().collect();

        Self {
            fence,
            _command_buffers: command_buffers,
            _semaphores: semaphores,
        }
    }
}

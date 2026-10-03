use std::{
    sync::{Arc, Mutex, MutexGuard},
    time::Duration,
};

use crate::{
    command_buffer::CommandBufferTrait,
    device::{Device, DeviceOwned},
    instance::InstanceOwned,
    prelude::{FrameworkError, VulkanError, VulkanResult},
    semaphore::Semaphore,
};

pub struct Fence {
    device: Arc<Device>,
    fence: ash::vk::Fence,
    submission_owned: Mutex<bool>,
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
        let status = unsafe {
            self.get_parent_device()
                .ash_handle()
                .get_fence_status(self.fence)
        }?;

        Ok(status)
    }

    fn lock_submission(&self) -> VulkanResult<MutexGuard<'_, bool>> {
        self.submission_owned
            .lock()
            .map_err(|err| FrameworkError::MutexError(format!("{err}")).into())
    }

    pub(crate) fn reserve_submission(&self) -> VulkanResult<()> {
        let mut owned = self.lock_submission()?;
        if *owned || self.is_signaled()? {
            return Err(ash::vk::Result::ERROR_UNKNOWN.into());
        }
        *owned = true;
        Ok(())
    }

    pub(crate) fn cancel_submission(&self) -> VulkanResult<()> {
        *self.lock_submission()? = false;
        Ok(())
    }

    fn finish_submission(&self) -> VulkanResult<()> {
        let mut owned = self.lock_submission()?;
        unsafe { self.device.ash_handle().reset_fences(&[self.fence]) }?;
        *owned = false;
        Ok(())
    }

    /// Reset an unowned fence. A submitted fence remains owned until its waiter is dropped.
    pub fn reset(&self) -> VulkanResult<()> {
        let owned = self.lock_submission()?;
        if *owned {
            return Err(ash::vk::Result::ERROR_UNKNOWN.into());
        }
        unsafe {
            self.get_parent_device()
                .ash_handle()
                .reset_fences(&[self.fence])
        }?;

        Ok(())
    }

    /**
     * This function accepts fences that have been created from the same device.
     */
    pub fn reset_fences(fences: &[Arc<Self>]) -> VulkanResult<()> {
        let mut device: Option<Arc<Device>> = None;
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
        }

        // Stable lock order avoids deadlocking concurrent resets with reversed input order.
        let mut ordered_fences: Vec<_> = fences.iter().collect();
        ordered_fences.sort_unstable_by_key(|fence| fence.native_handle());
        ordered_fences.dedup_by_key(|fence| fence.native_handle());
        let guards: Vec<_> = ordered_fences
            .iter()
            .map(|fence| fence.lock_submission())
            .collect::<VulkanResult<_>>()?;
        if guards.iter().any(|owned| **owned) {
            return Err(ash::vk::Result::ERROR_UNKNOWN.into());
        }
        let native_fences: Vec<_> = ordered_fences.iter().map(|fence| fence.fence).collect();
        match &device {
            Some(dev) => unsafe { dev.ash_handle().reset_fences(native_fences.as_ref()) }?,
            // list of fences are simply empty
            None => {}
        }

        Ok(())
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
                unsafe {
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
                }?;

                Ok(())
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

        let fence = unsafe {
            device.ash_handle().create_fence(
                &create_info,
                device.get_parent_instance().get_alloc_callbacks(),
            )
        }?;

        let mut obj_name_bytes = vec![];
        if let Some(ext) = device.ash_ext_debug_utils_ext() {
            if let Some(name) = debug_name {
                for name_ch in name.as_bytes().iter() {
                    obj_name_bytes.push(*name_ch);
                }
                obj_name_bytes.push(0x00);

                unsafe {
                    let object_name =
                        std::ffi::CStr::from_bytes_with_nul_unchecked(obj_name_bytes.as_slice());
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

        Ok(Arc::new(Self {
            device,
            fence,
            submission_owned: Mutex::new(false),
        }))
    }
}

type FenceWaiterCommandBuffersType = smallvec::SmallVec<[Arc<dyn CommandBufferTrait>; 8]>;
pub(crate) type FenceWaiterSemaphoresType = smallvec::SmallVec<[Arc<Semaphore>; 32]>;

pub struct FenceWaiter {
    fence: Arc<Fence>,
    command_buffers: FenceWaiterCommandBuffersType,
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
                Ok(_) => break,
                Err(err) => match err.is_timeout() {
                    true => continue,
                    false => panic!("Error while waiting for fence: {err:?}"),
                },
            }
        }

        for cb in self.command_buffers.iter() {
            cb.mark_execution_complete().unwrap();
        }

        self.fence.finish_submission().unwrap()
    }
}

impl FenceWaiter {
    pub(crate) fn new(
        fence: Arc<Fence>,
        command_buffers: &[Arc<dyn CommandBufferTrait>],
        semaphores: FenceWaiterSemaphoresType,
    ) -> Self {
        let command_buffers = command_buffers.iter().cloned().collect();
        Self {
            fence,
            command_buffers,
            _semaphores: semaphores,
        }
    }

    #[inline]
    pub fn complete(&self) -> VulkanResult<bool> {
        self.fence.is_signaled()
    }
}

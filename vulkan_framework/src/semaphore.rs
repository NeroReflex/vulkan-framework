use std::sync::Arc;

use crate::{
    device::{Device, DeviceOwned},
    instance::InstanceOwned,
    prelude::VulkanResult,
};

pub struct Semaphore {
    device: Arc<Device>,
    semaphore: ash::vk::Semaphore,
    semaphore_type: ash::vk::SemaphoreType,
}

impl Drop for Semaphore {
    fn drop(&mut self) {
        unsafe {
            self.device.ash_handle().destroy_semaphore(
                self.semaphore,
                self.device.get_parent_instance().get_alloc_callbacks(),
            )
        }
    }
}

impl DeviceOwned for Semaphore {
    fn get_parent_device(&self) -> Arc<Device> {
        self.device.clone()
    }
}

impl Semaphore {
    #[inline]
    pub(crate) fn ash_handle(&self) -> ash::vk::Semaphore {
        self.semaphore
    }

    #[inline]
    pub fn native_handle(&self) -> u64 {
        ash::vk::Handle::as_raw(self.semaphore)
    }

    /// Whether this semaphore carries a timeline payload rather than a binary signal.
    pub fn is_timeline(&self) -> bool {
        self.semaphore_type == ash::vk::SemaphoreType::TIMELINE
    }

    /**
     * This function accepts fences that have been created from the same device.
     */
    /*pub fn wait_for_fences(semaphores: &[Self], device_timeout_ns: u64) -> VulkanResult<()> {
        let mut device_native_handle: Option<Arc<Device>> = None;
        let mut native_fences = Vec::<ash::vk::Semaphore>::new();
        for semaphore in semaphores {
            match &device_native_handle {
                Some(old_device) => {
                    if semaphore.native_handle() != old_device.native_handle() {
                        return Err(...)
                    }
                },
                None => {
                    device_native_handle = Some(semaphore.device.clone())
                }
            }

            native_fences.push(semaphore.semaphore)
        }

        match device_native_handle {
            Some(device) => {
                let wait_result = unsafe {
                    device.ash_handle().wait_semaphores(native_fences.as_ref(), device_timeout_ns)
                };

                return match wait_result {
                    Ok(_) => { Ok(()) },
                    Err(err) => {
                        Err(...)
                    }
                }
            },
            None => {Err(...)}
        }
    }*/

    pub fn new(device: Arc<Device>, debug_name: Option<&str>) -> VulkanResult<Arc<Self>> {
        let create_info = ash::vk::SemaphoreCreateInfo::default();

        Self::create_semaphore(
            device,
            &create_info,
            ash::vk::SemaphoreType::BINARY,
            debug_name,
        )
    }

    /**
     * Creates a new timeline semaphore (core since Vulkan 1.2).
     *
     * Unlike binary semaphores a timeline semaphore carries a monotonically increasing u64 payload:
     * a wait operation blocks until the payload reaches the requested value, and a signal operation
     * sets the payload to the given value.
     *
     * A single submission is allowed to wait on the value K and to signal the value K+1 of the same
     * timeline semaphore: this is the correct way to express a dependency between two consecutive
     * submissions (eg. two frames reusing the same global illumination data). A binary semaphore
     * cannot bootstrap this chain: its wait requires a previously submitted signal operation.
     * Timeline signal values must increase in execution order, including across different queues;
     * callers must order those signals with appropriate waits.
     */
    pub fn new_timeline(
        device: Arc<Device>,
        initial_value: u64,
        debug_name: Option<&str>,
    ) -> VulkanResult<Arc<Self>> {
        let mut type_create_info = ash::vk::SemaphoreTypeCreateInfo::default()
            .semaphore_type(ash::vk::SemaphoreType::TIMELINE)
            .initial_value(initial_value);

        let create_info = ash::vk::SemaphoreCreateInfo::default().push_next(&mut type_create_info);

        Self::create_semaphore(
            device,
            &create_info,
            ash::vk::SemaphoreType::TIMELINE,
            debug_name,
        )
    }

    fn create_semaphore(
        device: Arc<Device>,
        create_info: &ash::vk::SemaphoreCreateInfo,
        semaphore_type: ash::vk::SemaphoreType,
        debug_name: Option<&str>,
    ) -> VulkanResult<Arc<Self>> {
        let semaphore = unsafe {
            device.ash_handle().create_semaphore(
                create_info,
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
                        .object_handle(semaphore)
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
            device,
            semaphore,
            semaphore_type,
        }))
    }
}

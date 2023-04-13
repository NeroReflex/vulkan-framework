use std::{ops::Deref, sync::Mutex};

use ash::vk::Queue;

use std::sync::Arc;

use crate::{
    device::{Device, DeviceOwned},
    prelude::{VulkanError, VulkanResult}, instance::InstanceOwned,
};

pub struct Fence {
    device: Arc<Device>,
    fence: ash::vk::Fence
}

impl Drop for Fence {
    fn drop(&mut self) {
        unsafe {
            self.device.ash_handle().destroy_fence(self.fence.clone(), self.device.get_parent_instance().get_alloc_callbacks())
        }
    }
}

impl DeviceOwned for Fence {
    fn get_parent_device(&self) -> Arc<Device> {
        self.device.clone()
    }
}

impl Fence {
    pub fn new(device: Arc<Device>, starts_in_signaled_state: bool, debug_name: Option<&str>,) -> VulkanResult<Arc<Self>> {
        let create_info = ash::vk::FenceCreateInfo::builder()
            .flags(match starts_in_signaled_state {
                true => { ash::vk::FenceCreateFlags::SIGNALED },
                false => { ash::vk::FenceCreateFlags::from_raw( 0x00u32 )}
            })
            .build();

        match unsafe { device.ash_handle().create_fence(&create_info, device.get_parent_instance().get_alloc_callbacks()) }{
            Ok(fence) => {
                let mut obj_name_bytes = vec![];
                match device.get_parent_instance().get_debug_ext_extension() {
                    Some(ext) => {
                        match debug_name {
                            Some(name) => {
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
                                        .object_type(ash::vk::ObjectType::FENCE)
                                        .object_handle(ash::vk::Handle::as_raw(fence.clone()))
                                        .object_name(object_name)
                                        .build();

                                    match ext.set_debug_utils_object_name(
                                        device.ash_handle().handle(),
                                        &dbg_info,
                                    ) {
                                        Ok(_) => {
                                            #[cfg(debug_assertions)]
                                            {
                                                println!("Queue Debug object name changed");
                                            }
                                        }
                                        Err(err) => {
                                            #[cfg(debug_assertions)]
                                            {
                                                println!("Error setting the Debug name for the newly created Queue, will use handle. Error: {}", err);
                                                assert_eq!(true, false);
                                            }
                                        }
                                    }
                                }
                            }
                            None => {}
                        };
                    }
                    None => {}
                }

                Ok(Arc::new(Self {
                    device,
                    fence
                }))
            },
            Err(err) => {
                #[cfg(debug_assertions)]
                {
                    println!("Error creating the fence: {}", err);
                }

                Err(VulkanError::Unspecified)
            }
        }

    }
}
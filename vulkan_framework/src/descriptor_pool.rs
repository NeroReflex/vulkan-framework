use std::sync::Arc;

use crate::{
    device::{Device, DeviceOwned},
    instance::InstanceOwned,
    prelude::{VulkanError, VulkanResult},
};

pub struct DescriptorPoolConcreteDescriptor {
    sampler: u32,
    combined_image_sampler: u32,
    sampled_image: u32,
    storage_image: u32,
    uniform_texel_buffer: u32,
    uniform_texel_storage: u32,
    storage_buffer: u32,
    uniform_buffer: u32,
    input_attachment: u32,
}

pub struct DescriptorPool {
    device: Arc<Device>,
    pool: ash::vk::DescriptorPool,
    descriptor: DescriptorPoolConcreteDescriptor,
}

pub(crate) trait DescriptorPoolOwned {
    fn get_parent_descriptor_pool(&self) -> Arc<DescriptorPool>;
}

impl DeviceOwned for DescriptorPool {
    fn get_parent_device(&self) -> Arc<Device> {
        self.device.clone()
    }
}

impl Drop for DescriptorPool {
    fn drop(&mut self) {
        unsafe {
            self.device.ash_handle().destroy_descriptor_pool(
                self.pool,
                self.device.get_parent_instance().get_alloc_callbacks(),
            )
        }
    }
}

impl DescriptorPool {
    pub fn new(
        device: Arc<Device>,
        descriptor: DescriptorPoolConcreteDescriptor,
        debug_name: Option<&str>,
    ) -> VulkanResult<Arc<Self>> {
        let create_info = ash::vk::DescriptorPoolCreateInfo::builder().build();

        match unsafe {
            device.ash_handle().create_descriptor_pool(
                &create_info,
                device.get_parent_instance().get_alloc_callbacks(),
            )
        } {
            Ok(pool) => {
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
                                        .object_type(ash::vk::ObjectType::DESCRIPTOR_POOL)
                                        .object_handle(ash::vk::Handle::as_raw(pool))
                                        .object_name(object_name)
                                        .build();

                                    match ext.set_debug_utils_object_name(
                                        device.ash_handle().handle(),
                                        &dbg_info,
                                    ) {
                                        Ok(_) => {
                                            #[cfg(debug_assertions)]
                                            {
                                                println!(
                                                    "Descriptor Pool Debug object name changed"
                                                );
                                            }
                                        }
                                        Err(err) => {
                                            #[cfg(debug_assertions)]
                                            {
                                                println!("Error setting the Debug name for the newly created DescriptorPool, will use handle. Error: {}", err);
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
                    pool,
                    descriptor,
                }))
            }
            Err(err) => {
                #[cfg(debug_assertions)]
                {
                    println!("Error creating the descriptor pool: {}", err);
                    assert_eq!(true, false)
                }

                return Err(VulkanError::Unspecified);
            }
        }
    }
}

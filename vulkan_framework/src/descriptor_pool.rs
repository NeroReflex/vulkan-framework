use std::sync::Arc;

use crate::{
    device::{Device, DeviceOwned},
    instance::InstanceOwned,
    prelude::{VulkanError, VulkanResult},
};

#[derive(Copy, Clone, PartialEq, Eq)]
pub struct DescriptorPoolSizesAcceletarionStructureKHR {
    acceleration_structure: u32,
}

impl DescriptorPoolSizesAcceletarionStructureKHR {
    pub(crate) fn ash_pool_sizes(&self) -> Vec<ash::vk::DescriptorPoolSize> {
        let mut pool_sizes = Vec::<ash::vk::DescriptorPoolSize>::new();

        if self.acceleration_structure() > 0 {
            pool_sizes.push(
                ash::vk::DescriptorPoolSize::builder()
                    .ty(ash::vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
                    .descriptor_count(self.acceleration_structure())
                    .build(),
            )
        }

        pool_sizes
    }

    pub fn acceleration_structure(&self) -> u32 {
        self.acceleration_structure
    }

    pub fn new(acceleration_structure: u32) -> Self {
        Self {
            acceleration_structure,
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq)]
pub struct DescriptorPoolSizesConcreteDescriptor {
    sampler: u32,
    combined_image_sampler: u32,
    sampled_image: u32,
    storage_image: u32,
    uniform_texel_buffer: u32,
    storage_texel_buffer: u32,
    storage_buffer: u32,
    uniform_buffer: u32,
    input_attachment: u32,
    acceleration_structure: DescriptorPoolSizesAcceletarionStructureKHR,
}

impl DescriptorPoolSizesConcreteDescriptor {
    pub(crate) fn ash_pool_sizes(&self) -> Vec<ash::vk::DescriptorPoolSize> {
        let mut pool_sizes = Vec::<ash::vk::DescriptorPoolSize>::new();

        if self.sampler() > 0 {
            pool_sizes.push(
                ash::vk::DescriptorPoolSize::builder()
                    .ty(ash::vk::DescriptorType::SAMPLER)
                    .descriptor_count(self.sampler())
                    .build(),
            )
        }

        if self.combined_image_sampler() > 0 {
            pool_sizes.push(
                ash::vk::DescriptorPoolSize::builder()
                    .ty(ash::vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .descriptor_count(self.combined_image_sampler())
                    .build(),
            )
        }

        if self.sampled_image() > 0 {
            pool_sizes.push(
                ash::vk::DescriptorPoolSize::builder()
                    .ty(ash::vk::DescriptorType::SAMPLED_IMAGE)
                    .descriptor_count(self.sampled_image())
                    .build(),
            )
        }

        if self.storage_image() > 0 {
            pool_sizes.push(
                ash::vk::DescriptorPoolSize::builder()
                    .ty(ash::vk::DescriptorType::STORAGE_IMAGE)
                    .descriptor_count(self.storage_image())
                    .build(),
            )
        }

        if self.uniform_texel_buffer() > 0 {
            pool_sizes.push(
                ash::vk::DescriptorPoolSize::builder()
                    .ty(ash::vk::DescriptorType::UNIFORM_TEXEL_BUFFER)
                    .descriptor_count(self.uniform_texel_buffer())
                    .build(),
            )
        }

        if self.storage_texel_buffer() > 0 {
            pool_sizes.push(
                ash::vk::DescriptorPoolSize::builder()
                    .ty(ash::vk::DescriptorType::STORAGE_TEXEL_BUFFER)
                    .descriptor_count(self.storage_texel_buffer())
                    .build(),
            )
        }

        if self.storage_buffer() > 0 {
            pool_sizes.push(
                ash::vk::DescriptorPoolSize::builder()
                    .ty(ash::vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(self.storage_buffer())
                    .build(),
            )
        }

        if self.uniform_buffer() > 0 {
            pool_sizes.push(
                ash::vk::DescriptorPoolSize::builder()
                    .ty(ash::vk::DescriptorType::UNIFORM_BUFFER)
                    .descriptor_count(self.uniform_buffer())
                    .build(),
            )
        }

        if self.input_attachment() > 0 {
            pool_sizes.push(
                ash::vk::DescriptorPoolSize::builder()
                    .ty(ash::vk::DescriptorType::INPUT_ATTACHMENT)
                    .descriptor_count(self.input_attachment())
                    .build(),
            )
        }

        pool_sizes.extend(self.acceleration_structure.ash_pool_sizes().iter().cloned());

        pool_sizes
    }

    pub fn new(
        sampler: u32,
        combined_image_sampler: u32,
        sampled_image: u32,
        storage_image: u32,
        uniform_texel_buffer: u32,
        storage_texel_buffer: u32,
        storage_buffer: u32,
        uniform_buffer: u32,
        input_attachment: u32,
        acceleration_structure: Option<DescriptorPoolSizesAcceletarionStructureKHR>,
    ) -> Self {
        Self {
            sampler,
            combined_image_sampler,
            sampled_image,
            storage_image,
            uniform_texel_buffer,
            storage_texel_buffer,
            storage_buffer,
            uniform_buffer,
            input_attachment,
            acceleration_structure: match acceleration_structure {
                Some(acc_s) => acc_s,
                None => DescriptorPoolSizesAcceletarionStructureKHR::new(0),
            },
        }
    }

    pub fn sampler(&self) -> u32 {
        self.sampler
    }

    pub fn combined_image_sampler(&self) -> u32 {
        self.combined_image_sampler
    }

    pub fn sampled_image(&self) -> u32 {
        self.sampled_image
    }

    pub fn storage_image(&self) -> u32 {
        self.storage_image
    }

    pub fn uniform_texel_buffer(&self) -> u32 {
        self.uniform_texel_buffer
    }

    pub fn storage_texel_buffer(&self) -> u32 {
        self.storage_texel_buffer
    }

    pub fn storage_buffer(&self) -> u32 {
        self.storage_buffer
    }

    pub fn uniform_buffer(&self) -> u32 {
        self.uniform_buffer
    }

    pub fn input_attachment(&self) -> u32 {
        self.input_attachment
    }
}

#[derive(Copy, Clone, PartialEq, Eq)]
pub struct DescriptorPoolConcreteDescriptor {
    pool_sizes: DescriptorPoolSizesConcreteDescriptor,
    max_sets: u32,
}

impl DescriptorPoolConcreteDescriptor {
    pub fn max_sets(&self) -> u32 {
        self.max_sets
    }

    pub(crate) fn ash_pool_sizes(&self) -> Vec<ash::vk::DescriptorPoolSize> {
        self.pool_sizes.ash_pool_sizes()
    }

    pub(crate) fn ash_flags(&self) -> ash::vk::DescriptorPoolCreateFlags {
        ash::vk::DescriptorPoolCreateFlags::from_raw(0)
    }

    pub fn new(pool_sizes: DescriptorPoolSizesConcreteDescriptor, max_sets: u32) -> Self {
        Self {
            pool_sizes,
            max_sets,
        }
    }
}

pub struct DescriptorPool {
    device: Arc<Device>,
    pool: ash::vk::DescriptorPool,
    descriptor: DescriptorPoolConcreteDescriptor,
    //allocated_sets: Arc<Mutex<u32>>,
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
    pub(crate) fn ash_handle(&self) -> ash::vk::DescriptorPool {
        self.pool
    }

    pub fn native_handle(&self) -> u64 {
        ash::vk::Handle::as_raw(self.pool)
    }

    pub fn descriptor(&self) -> DescriptorPoolConcreteDescriptor {
        self.descriptor
    }

    pub fn new(
        device: Arc<Device>,
        descriptor: DescriptorPoolConcreteDescriptor,
        debug_name: Option<&str>,
    ) -> VulkanResult<Arc<Self>> {
        let pool_sizes = descriptor.ash_pool_sizes();
        let create_info = ash::vk::DescriptorPoolCreateInfo::builder()
            .flags(ash::vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET | descriptor.ash_flags())
            .max_sets(descriptor.max_sets)
            .pool_sizes(pool_sizes.as_slice())
            .build();

        match unsafe {
            device.ash_handle().create_descriptor_pool(
                &create_info,
                device.get_parent_instance().get_alloc_callbacks(),
            )
        } {
            Ok(pool) => {
                let mut obj_name_bytes = vec![];
                if let Some(ext) = device.get_parent_instance().get_debug_ext_extension() {
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
                                .object_type(ash::vk::ObjectType::DESCRIPTOR_POOL)
                                .object_handle(ash::vk::Handle::as_raw(pool))
                                .object_name(object_name)
                                .build();

                            if let Err(err) = ext.set_debug_utils_object_name(
                                device.ash_handle().handle(),
                                &dbg_info,
                            ) {
                                #[cfg(debug_assertions)]
                                {
                                    println!("Error setting the Debug name for the newly created DescriptorPool, will use handle. Error: {}", err)
                                }
                            }
                        }
                    }
                }

                Ok(Arc::new(Self {
                    device,
                    pool,
                    descriptor,
                    //allocated_sets: Arc::new(Mutex::new(0)),
                    //lock_errored: AtomicI16::new(0i16),
                }))
            }
            Err(err) => Err(VulkanError::Vulkan(
                err.as_raw(),
                Some(format!("Error creating the descriptor pool: {}", err)),
            )),
        }
    }
}

use std::sync::Arc;

#[cfg(feature = "better_mutex")]
use parking_lot::{const_mutex, Mutex};

#[cfg(not(feature = "better_mutex"))]
use std::sync::Mutex;

use crate::{
    buffer::{Buffer, BufferTrait, BufferUsage, ConcreteBufferDescriptor},
    device::{Device, DeviceOwned},
    instance::InstanceOwned,
    memory_heap::MemoryType,
    memory_management::{MemoryManagementTags, MemoryManagerTrait},
    memory_pool::MemoryPoolFeatures,
    prelude::{VulkanError, VulkanResult},
};

fn scratch_allocation_size(size: u64, alignment: u64) -> Result<u64, ash::vk::Result> {
    if !alignment.is_power_of_two() {
        return Err(ash::vk::Result::ERROR_INITIALIZATION_FAILED);
    }
    // Vulkan buffers must be nonempty, even for an empty acceleration structure.
    size.max(1)
        .checked_add(alignment - 1)
        .ok_or(ash::vk::Result::ERROR_INITIALIZATION_FAILED)
}

fn aligned_scratch_address(
    base: u64,
    backing_size: u64,
    size: u64,
    alignment: u64,
) -> Result<u64, ash::vk::Result> {
    let error = ash::vk::Result::ERROR_INITIALIZATION_FAILED;
    if base == 0 || !alignment.is_power_of_two() {
        return Err(error);
    }
    let aligned = base.checked_add(alignment - 1).ok_or(error)? & !(alignment - 1);
    let required_size = size.max(1);
    let end_offset = (aligned - base).checked_add(required_size).ok_or(error)?;
    if end_offset > backing_size || aligned.checked_add(required_size).is_none() {
        return Err(error);
    }
    Ok(aligned)
}

pub struct DeviceScratchBuffer {
    buffer: Arc<dyn BufferTrait>,
    buffer_device_addr: u64,
}

impl DeviceOwned for DeviceScratchBuffer {
    fn get_parent_device(&self) -> Arc<Device> {
        self.buffer.get_parent_device()
    }
}

impl BufferTrait for DeviceScratchBuffer {
    #[inline]
    fn size(&self) -> u64 {
        self.buffer.size()
    }

    #[inline]
    fn native_handle(&self) -> u64 {
        self.buffer.native_handle()
    }
}

impl DeviceScratchBuffer {
    #[inline]
    pub(crate) fn addr(&self) -> ash::vk::DeviceOrHostAddressKHR {
        ash::vk::DeviceOrHostAddressKHR {
            device_address: self.buffer_device_addr,
        }
    }

    pub fn new(
        memory_manager: &mut dyn MemoryManagerTrait,
        size: u64,
        allocation_tags: MemoryManagementTags,
    ) -> VulkanResult<Arc<Self>> {
        let device = memory_manager.get_parent_device();
        if device.ash_ext_acceleration_structure_khr().is_none() {
            return Err(VulkanError::MissingExtension(String::from(
                "VK_KHR_acceleration_structure",
            )));
        }
        let alignment = match device.ray_tracing_info() {
            Some(info) => info.min_acceleration_structure_scratch_offset_alignment() as u64,
            None => {
                // AS builds can be enabled without the ray-tracing pipeline extension.
                let mut properties =
                    ash::vk::PhysicalDeviceAccelerationStructurePropertiesKHR::default();
                let mut properties2 =
                    ash::vk::PhysicalDeviceProperties2::default().push_next(&mut properties);
                unsafe {
                    device
                        .get_parent_instance()
                        .ash_handle()
                        .get_physical_device_properties2(
                            *device.ash_physical_device_handle(),
                            &mut properties2,
                        );
                }
                properties.min_acceleration_structure_scratch_offset_alignment as u64
            }
        };
        let allocation_size = scratch_allocation_size(size, alignment)?;
        let backing_buffer = Buffer::new(
            device.clone(),
            ConcreteBufferDescriptor::new(
                BufferUsage::from(
                    (ash::vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                        | ash::vk::BufferUsageFlags::STORAGE_BUFFER)
                        .as_raw(),
                ),
                allocation_size,
            ),
            None,
            None,
        )?;

        let buffer = memory_manager.allocate_resources(
            &MemoryType::device_local_and_host_visible(),
            &MemoryPoolFeatures::new(true),
            vec![backing_buffer.into()],
            allocation_tags,
        )?[0]
            .buffer();

        let info = ash::vk::BufferDeviceAddressInfo::default().buffer(buffer.ash_handle());

        let base = unsafe { device.ash_handle().get_buffer_device_address(&info) };
        // Align the actual device address, not merely the requested buffer size.
        let buffer_device_addr = aligned_scratch_address(base, buffer.size(), size, alignment)?;

        Ok(Arc::new(Self {
            buffer,
            buffer_device_addr,
        }))
    }
}

pub struct HostScratchBuffer {
    buffer: Mutex<Vec<u8>>,
}

impl HostScratchBuffer {
    pub(crate) fn address(&self) -> ash::vk::DeviceOrHostAddressKHR {
        #[cfg(feature = "better_mutex")]
        {
            let mut lck = self.buffer.lock();

            ash::vk::DeviceOrHostAddressKHR {
                host_address: lck.as_mut_slice().as_mut_ptr() as *mut std::ffi::c_void,
            }
        }

        #[cfg(not(feature = "better_mutex"))]
        {
            match self.buffer.lock() {
                Ok(mut lck) => ash::vk::DeviceOrHostAddressKHR {
                    host_address: lck.as_mut_slice().as_mut_ptr() as *mut std::ffi::c_void,
                },
                Err(_err) => {
                    todo!()
                }
            }
        }
    }

    pub fn new(size: u64) -> Arc<Self> {
        #[cfg(feature = "better_mutex")]
        {
            Arc::new(Self {
                buffer: const_mutex(Vec::<u8>::with_capacity(size as usize)),
            })
        }

        #[cfg(not(feature = "better_mutex"))]
        {
            Arc::new(Self {
                buffer: Mutex::new(Vec::<u8>::with_capacity(size as usize)),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{aligned_scratch_address, scratch_allocation_size};
    use ash::vk;

    #[test]
    fn scratch_address_aligns_every_base_offset_without_losing_capacity() {
        for alignment in [1, 16, 128, 256, 512] {
            for size in [0, 1, 255, 256, 1024] {
                let backing_size = scratch_allocation_size(size, alignment).unwrap();
                for offset in 0..alignment {
                    let base = 0x10000 + offset;
                    let address =
                        aligned_scratch_address(base, backing_size, size, alignment).unwrap();
                    assert_eq!(address % alignment, 0);
                    assert!(address >= base);
                    assert!(address - base < alignment);
                    assert!(address - base + size.max(1) <= backing_size);
                    if offset == 0 {
                        assert_eq!(address, base);
                    }
                }
            }
        }
    }

    #[test]
    fn scratch_address_rejects_insufficient_backing_storage() {
        let error = Err(vk::Result::ERROR_INITIALIZATION_FAILED);
        assert_eq!(aligned_scratch_address(0x1001, 256, 256, 256), error);
        assert_eq!(aligned_scratch_address(0x1000, 0, 0, 256), error);
        assert_eq!(aligned_scratch_address(0x1000, 255, 256, 256), error);
        assert_eq!(aligned_scratch_address(0x1001, 511, 256, 256), Ok(0x1100));
    }

    #[test]
    fn scratch_address_rejects_invalid_alignment_null_and_overflow() {
        let error = Err(vk::Result::ERROR_INITIALIZATION_FAILED);
        for alignment in [0, 3, 255] {
            assert_eq!(scratch_allocation_size(256, alignment), error);
            assert_eq!(aligned_scratch_address(0x1000, 4096, 256, alignment), error);
        }
        assert_eq!(scratch_allocation_size(u64::MAX, 256), error);
        assert_eq!(aligned_scratch_address(0, 4096, 256, 256), error);
        assert_eq!(aligned_scratch_address(u64::MAX, 4096, 256, 256), error);
        assert_eq!(
            aligned_scratch_address(u64::MAX - 255, 4096, 256, 256),
            error
        );
    }
}

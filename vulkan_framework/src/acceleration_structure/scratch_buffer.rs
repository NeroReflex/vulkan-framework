use std::sync::Arc;

#[cfg(feature = "better_mutex")]
use parking_lot::{const_mutex, Mutex};

#[cfg(not(feature = "better_mutex"))]
use std::sync::Mutex;

use crate::{
    buffer::{AllocatedBuffer, Buffer, BufferTrait, BufferUsage, ConcreteBufferDescriptor},
    device::{Device, DeviceOwned},
    graphics_pipeline::AttributeType,
    instance::InstanceOwned,
    memory_heap::MemoryHeapOwned,
    memory_pool::MemoryPool,
    prelude::{FrameworkError, VulkanError, VulkanResult},
};

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

    pub fn new(memory_pool: Arc<MemoryPool>, size: u64) -> VulkanResult<Arc<Self>> {
        let buffer = AllocatedBuffer::new(
            memory_pool.clone(),
            Buffer::new(
                memory_pool
                    .get_parent_memory_heap()
                    .get_parent_device()
                    .clone(),
                ConcreteBufferDescriptor::new(
                    BufferUsage::Unmanaged(
                        (ash::vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                            | ash::vk::BufferUsageFlags::STORAGE_BUFFER)
                            .as_raw(),
                    ),
                    size,
                ),
                None,
                None,
            )?,
        )?;
        let info = ash::vk::BufferDeviceAddressInfo::default().buffer(buffer.ash_handle());

        let buffer_device_addr = unsafe {
            memory_pool
                .get_parent_memory_heap()
                .get_parent_device()
                .ash_handle()
                .get_buffer_device_address(&info)
        };

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

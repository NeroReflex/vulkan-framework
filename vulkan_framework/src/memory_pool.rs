use ash::vk;

use crate::{
    device::DeviceOwned,
    instance::{InstanceOwned, InstanceAPIVersion},
    memory_allocator::*,
    memory_heap::{MemoryHeap, MemoryHeapOwned},
    prelude::{VulkanError, VulkanResult},
};

use std::{mem::size_of, sync::Arc};

pub struct MemoryPool
{
    memory_heap: Arc<MemoryHeap>,
    allocator: Arc<dyn MemoryAllocator>,
    memory: ash::vk::DeviceMemory,
}

impl MemoryHeapOwned for MemoryPool
{
    fn get_parent_memory_heap(&self) -> Arc<crate::memory_heap::MemoryHeap> {
        self.memory_heap.clone()
    }
}

pub trait MemoryPoolBacked
{
    fn get_backing_memory_pool(&self) -> Arc<MemoryPool>;

    fn allocation_offset(&self) -> u64;

    fn allocation_size(&self) -> u64;
}

impl Drop for MemoryPool
{
    fn drop(&mut self) {
        let memory_heap = self.get_parent_memory_heap();
        let device = memory_heap.get_parent_device();
        unsafe {
            device.ash_handle().free_memory(
                self.memory,
                device.get_parent_instance().get_alloc_callbacks(),
            );
        }
    }
}

impl MemoryPool
{
    pub fn native_handle(&self) -> u64 {
        ash::vk::Handle::as_raw(self.memory)
    }

    pub(crate) fn ash_handle(&self) -> ash::vk::DeviceMemory {
        self.memory
    }

    pub fn write_raw_data<T>(&self, offset: u64, src: &[T]) -> VulkanResult<()>
    where
        T: Copy + Sized,
    {
        let device = self.get_parent_memory_heap().get_parent_device();

        match unsafe {
            device
                .ash_handle()
                .map_memory(self.memory, offset, (src.len() + size_of::<T>()) as u64, vk::MemoryMapFlags::empty())
        } {
            Ok(ptr) => {
                // copy raw from data to ptr

                let mapped_typed_ptr = unsafe { std::slice::from_raw_parts_mut(ptr as *mut T, src.len())};
                mapped_typed_ptr.copy_from_slice(src);

                unsafe { device.ash_handle().unmap_memory(self.memory) }

                Ok(())
            }
            Err(err) => {
                #[cfg(debug_assertions)]
                {
                    panic!("Error mapping the memory: {}", err)
                }

                Err(VulkanError::Unspecified)
            }
        }
    }

    pub fn read_raw_data<T>(&self, offset: u64, size: u64) -> VulkanResult<Vec<T>>
    where
        T: Copy,
    {
        let device = self.get_parent_memory_heap().get_parent_device();

        match unsafe {
            device
                .ash_handle()
                .map_memory(self.memory, offset, size, vk::MemoryMapFlags::empty())
        } {
            Ok(ptr) => {
                let slice = std::ptr::slice_from_raw_parts(
                    ptr as *const T,
                    (size as usize) / size_of::<T>(),
                );

                let data = unsafe { (*slice).to_vec() };

                unsafe { device.ash_handle().unmap_memory(self.memory) }

                Ok(data)
            }
            Err(err) => {
                #[cfg(debug_assertions)]
                {
                    panic!("Error mapping the memory: {}", err)
                }

                Err(VulkanError::Unspecified)
            }
        }
    }

    pub fn new_with_device_address(memory_heap: Arc<MemoryHeap>, allocator: Arc<dyn MemoryAllocator>) -> VulkanResult<Arc<Self>> {
        let mut create_info = ash::vk::MemoryAllocateInfo::builder()
            .allocation_size(allocator.total_size())
            .memory_type_index(memory_heap.type_index())
            .build();

        let device = memory_heap.get_parent_device();

        // the flag I want is only available since vulkan version 1.2
        let mut memory_flags = ash::vk::MemoryAllocateFlagsInfo::builder()
            .flags(ash::vk::MemoryAllocateFlags::DEVICE_ADDRESS)
            .build();

        let instance_ver = device.get_parent_instance().instance_vulkan_version();
        if (instance_ver != InstanceAPIVersion::Version1_0) && (instance_ver != InstanceAPIVersion::Version1_1) {
            create_info.p_next = &mut memory_flags as *mut ash::vk::MemoryAllocateFlagsInfo as *mut std::ffi::c_void;
        }

        unsafe {
            match device.ash_handle().allocate_memory(
                &create_info,
                device.get_parent_instance().get_alloc_callbacks(),
            ) {
                Ok(memory) => Ok(Arc::new(Self {
                    memory_heap,
                    allocator,
                    memory,
                })),
                Err(err) => {
                    #[cfg(debug_assertions)]
                    {
                        panic!("Error creating the memory pool: {}", err)
                    }

                    Err(VulkanError::Unspecified)
                }
            }
        }
    }

    pub fn new(memory_heap: Arc<MemoryHeap>, allocator: Arc<dyn MemoryAllocator>) -> VulkanResult<Arc<Self>> {
        let create_info = ash::vk::MemoryAllocateInfo::builder()
            .allocation_size(allocator.total_size())
            .memory_type_index(memory_heap.type_index())
            .build();

        let device = memory_heap.get_parent_device();

        unsafe {
            match device.ash_handle().allocate_memory(
                &create_info,
                device.get_parent_instance().get_alloc_callbacks(),
            ) {
                Ok(memory) => Ok(Arc::new(Self {
                    memory_heap,
                    allocator,
                    memory,
                })),
                Err(err) => {
                    #[cfg(debug_assertions)]
                    {
                        panic!("Error creating the memory pool: {}", err)
                    }

                    Err(VulkanError::Unspecified)
                }
            }
        }
    }

    pub(crate) fn alloc(
        &self,
        memory_requirements: ash::vk::MemoryRequirements,
    ) -> Option<AllocationResult> {
        // check if this pool satisfy memory requirements...
        if
        /*(memory_requirements.memory_type_bits & todo!()) == memory_requirements.memory_type_bits*/
        true {
            // ...and if it does try to allocate the required memory
            self.allocator
                .alloc(memory_requirements.size, memory_requirements.alignment)
        } else {
            // ...if it does not inform the user that the allocation has simply failed
            None
        }
    }

    pub(crate) fn dealloc(&self, mem: &mut AllocationResult) {
        self.allocator.dealloc(mem)
    }
}

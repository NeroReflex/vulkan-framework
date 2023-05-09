use ash::vk;

use crate::{
    device::DeviceOwned,
    instance::InstanceOwned,
    memory_allocator::*,
    memory_heap::{MemoryHeap, MemoryHeapOwned},
    prelude::{VulkanError, VulkanResult},
};

use std::sync::Arc;

pub struct MemoryPool<Allocator>
where
    Allocator: MemoryAllocator + Send + Sync,
{
    memory_heap: Arc<MemoryHeap>,
    allocator: Allocator,
    memory: ash::vk::DeviceMemory,
}

impl<Allocator> MemoryHeapOwned for MemoryPool<Allocator>
where
    Allocator: MemoryAllocator + Send + Sync,
{
    fn get_parent_memory_heap(&self) -> Arc<crate::memory_heap::MemoryHeap> {
        self.memory_heap.clone()
    }
}

pub trait MemoryPoolBacked<Allocator>
where
    Allocator: MemoryAllocator + Send + Sync,
{
    fn get_backing_memory_pool(&self) -> Arc<MemoryPool<Allocator>>;
}

impl<Allocator> Drop for MemoryPool<Allocator>
where
    Allocator: MemoryAllocator + Send + Sync,
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

impl<Allocator> MemoryPool<Allocator>
where
    Allocator: MemoryAllocator + Send + Sync,
{
    pub fn native_handle(&self) -> u64 {
        ash::vk::Handle::as_raw(self.memory)
    }

    pub(crate) fn ash_handle(&self) -> ash::vk::DeviceMemory {
        self.memory
    }

    pub fn clone_raw_data(&self, offset: u64, size: u64) -> VulkanResult<Vec<u8>> {
        let device = self.get_parent_memory_heap().get_parent_device();
        
        let data: Vec<u8> = vec![];

        unsafe { device.ash_handle().map_memory(self.memory, offset, size, vk::MemoryMapFlags::empty()) };

        todo!();

        Ok(data)
    }

    pub fn new(memory_heap: Arc<MemoryHeap>, allocator: Allocator) -> VulkanResult<Arc<Self>> {
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

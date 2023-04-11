use crate::{
    device::{Device, DeviceOwned},
    instance::InstanceOwned,
    memory_allocator::*,
    memory_heap::{MemoryHeap, MemoryHeapOwned},
    prelude::{VulkanError, VulkanResult},
};

use std::sync::Arc;

pub struct MemoryPool<Allocator>
where
    Allocator: MemoryAllocator,
{
    memory_heap: Arc<MemoryHeap>,
    allocator: Allocator,
    memory: ash::vk::DeviceMemory,
}

impl<Allocator> MemoryHeapOwned for MemoryPool<Allocator>
where
    Allocator: MemoryAllocator,
{
    fn get_parent_memory_heap(&self) -> Arc<crate::memory_heap::MemoryHeap> {
        self.memory_heap.clone()
    }
}

trait MemoryPoolBacked<Allocator>
where
    Allocator: MemoryAllocator,
{
    fn get_backing_memory_pool(&self) -> Arc<MemoryPool<Allocator>>;
}

impl<Allocator> Drop for MemoryPool<Allocator>
where
    Allocator: MemoryAllocator,
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
    Allocator: MemoryAllocator,
{
    pub fn new(memory_heap: Arc<MemoryHeap>, allocator: Allocator) -> VulkanResult<Arc<Self>> {
        let create_info = ash::vk::MemoryAllocateInfo::builder()
            .allocation_size(allocator.total_size())
            .memory_type_index(memory_heap.heap_index())
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
                        println!("Error creating the memory pool: {}", err);
                        assert_eq!(true, false)
                    }

                    Err(VulkanError::Unspecified)
                }
            }
        }
    }
}

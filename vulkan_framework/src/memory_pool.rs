use crate::{
    memory_allocator::*,
    device::{Device, DeviceOwned},
    result::VkError,
    instance::InstanceOwned,
    memory_heap::{ MemoryHeapOwned, MemoryHeap }
};

pub struct MemoryPool<'ctx, 'instance, 'device, 'memory_heap, Allocator>
where
    Allocator: MemoryAllocator,
    'ctx: 'instance,
    'instance: 'device,
    'memory_heap: 'device
{
    memory_heap: &'memory_heap MemoryHeap<'ctx, 'instance, 'device>, 
    allocator: Allocator,
    memory: ash::vk::DeviceMemory,
}

impl<'ctx, 'instance, 'device, 'memory_heap, Allocator> MemoryHeapOwned<'ctx, 'instance, 'device> for MemoryPool<'ctx, 'instance, 'device, 'memory_heap, Allocator>
where
    Allocator: MemoryAllocator,
    'ctx: 'instance,
    'instance: 'device,
    'memory_heap: 'device
{
    fn get_parent_memory_heap(&self) -> &crate::memory_heap::MemoryHeap<'ctx, 'instance, 'device> {
        todo!()
    }
}

trait MemoryPoolBacked<'ctx, 'instance, 'device, 'memory_heap, Allocator>
where
    Allocator: MemoryAllocator,
    'ctx: 'instance,
    'instance: 'device,
    'memory_heap: 'device
{
    fn get_backing_memory_pool(&self) -> &MemoryPool<'ctx, 'instance, 'device, 'memory_heap, Allocator>;
}

impl<'ctx, 'instance, 'device, 'memory_heap, Allocator> Drop for MemoryPool<'ctx, 'instance, 'device, 'memory_heap, Allocator>
where
    Allocator: MemoryAllocator,
    'ctx: 'instance,
    'instance: 'device,
    'memory_heap: 'device
{
    fn drop(&mut self) {
        let memory_heap = self.get_parent_memory_heap();
        let device = memory_heap.get_parent_device();
        unsafe {
            device.ash_handle().free_memory(self.memory, device.get_parent_instance().get_alloc_callbacks());
        }
    }
}

impl<'ctx, 'instance, 'device, 'memory_heap, Allocator> MemoryPool<'ctx, 'instance, 'device, 'memory_heap, Allocator>
where
    Allocator: MemoryAllocator,
    'ctx: 'instance,
    'instance: 'device,
    'memory_heap: 'device
{
    pub fn new(memory_heap: &'memory_heap MemoryHeap<'ctx, 'instance, 'device>, allocator: Allocator) -> Result<Self, VkError> {
        let create_info = ash::vk::MemoryAllocateInfo::builder()
            .allocation_size(allocator.total_size())
            .memory_type_index(memory_heap.heap_index())
            .build();

        let device = memory_heap.get_parent_device();

        unsafe {
            match device.ash_handle().allocate_memory(&create_info, device.get_parent_instance().get_alloc_callbacks()) {
                Ok(memory) => {
                    Ok(
                        Self {
                            memory_heap,
                            allocator,
                            memory
                        }
                    )
                },
                Err(err) => {
                    #[cfg(debug_assertions)]
                    {
                        println!("Error creating the memory pool: {}", err);
                        assert_eq!(true, false)
                    }

                    Err(VkError {  })
                }
            }
        }
    }
}
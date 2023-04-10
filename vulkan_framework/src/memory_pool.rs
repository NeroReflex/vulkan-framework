use crate::{memory_allocator::*, device::Device, result::VkError, instance::InstanceOwned};

pub struct MemoryPool<'ctx, 'instance, 'device, Allocator>
where
    Allocator: MemoryAllocator
{
    device: &'device Device<'ctx, 'instance>, 
    allocator: Allocator,
    memory: ash::vk::DeviceMemory,
}

trait MemoryPoolBacked<'ctx, 'instance, 'device, Allocator>
where
    Allocator: MemoryAllocator
{
    fn get_backing_memory_pool(&self) -> &MemoryPool<'ctx, 'instance, 'device, Allocator>;
}

impl<'ctx, 'instance, 'device, Allocator> Drop for MemoryPool<'ctx, 'instance, 'device, Allocator>
where
    Allocator: MemoryAllocator
{
    fn drop(&mut self) {
        unsafe {
            self.device.ash_handle().free_memory(self.memory, self.device.get_parent_instance().get_alloc_callbacks());
        }
    }
}

impl<'ctx, 'instance, 'device, Allocator> MemoryPool<'ctx, 'instance, 'device, Allocator>
where
    Allocator: MemoryAllocator
{
    pub fn new(device: &'device Device<'ctx, 'instance>, allocator: Allocator) -> Result<Self, VkError> {
        let create_info = ash::vk::MemoryAllocateInfo::builder()
            .allocation_size(allocator.total_size())
            .memory_type_index(0)
            .build();

        unsafe {
            match device.ash_handle().allocate_memory(&create_info, device.get_parent_instance().get_alloc_callbacks()) {
                Ok(memory) => {
                    Ok(
                        Self {
                            device,
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
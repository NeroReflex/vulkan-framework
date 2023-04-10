use crate::{
    device::{Device, DeviceOwned},
    result::VkError,
};

pub struct MemoryHeap<'ctx, 'instance, 'device>
where
    'ctx: 'instance,
    'instance: 'device
{
    device: &'device Device<'ctx, 'instance>,
    //descriptor: Vec<f32>,
    //created_queues: Mutex<u64>,
    heap_index: u32,
}

pub(crate) trait MemoryHeapOwned<'ctx, 'instance, 'device>
where
    'ctx: 'instance,
    'instance: 'device
{
    fn get_parent_memory_heap(&self) -> &MemoryHeap<'ctx, 'instance, 'device>;
}

impl<'ctx, 'instance, 'device> DeviceOwned<'ctx, 'instance> for MemoryHeap<'ctx, 'instance, 'device>
where
    'ctx: 'instance,
    'instance: 'device
{
    fn get_parent_device(&self) -> &'device Device<'ctx, 'instance> {
        self.device.clone()
    }
}

impl<'ctx, 'instance, 'device> Drop for MemoryHeap<'ctx, 'instance, 'device>
where
    'ctx: 'instance,
    'instance: 'device
{
    fn drop(&mut self) {
        // Nothing to be done here
    }
}

impl<'ctx, 'instance, 'device> MemoryHeap<'ctx, 'instance, 'device>
where
    'ctx: 'instance,
    'instance: 'device
{
    pub(crate) fn heap_index(&self) -> u32 {
        self.heap_index
    } 

    pub fn new(device: &'device Device<'ctx, 'instance>) -> Result<Self, VkError> {

        todo!(); // change that zero below!!!

        Ok(
            Self {
                device: device,
                heap_index: 0
            }
        )
    }
}
use crate::{
    device::{Device, DeviceOwned},
    result::VkError,
};

#[derive(Clone)]
pub struct MemoryHostVisibility {
    host_coherence: bool, // false <= impossible if latter is false
    host_cached: bool, //false
}

#[derive(Clone)]
pub enum MemoryHostCoherence {
    Uncached, // host coherence is implemented via memory being uncached, as stated by vulkan specification: "uncached memory is always host coherent"

}

/**
 * If DeviceOnly(None) is specified a memory heap with both VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
 * and VK_MEMORY_PROPERTY_PROTECTED_BIT is selected.
 * 
 * If NotNecessarilyDeviceLocal is specified a memory heap with at least VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
 * is selected, if NotNecessarilyDeviceLocal(None) is selected a heap that is NOT host-coherent will be selected,
 * otherwise if Some(Uncached) is selected than a memory heap with VK_MEMORY_PROPERTY_HOST_CACHED_BIT unset.
 */
#[derive(Clone)]
pub enum MemoryType {
    //HostVisible({}),
    DeviceOnly(Option<MemoryHostVisibility>),
    NotNecessarilyDeviceLocal(Option<MemoryHostCoherence>)
}

#[derive(Clone)]
pub struct ConcreteMemoryHeapDescriptor {
    memory_type: MemoryType,
    memory_minimum_size: u64
}

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

    pub fn new(device: &'device Device<'ctx, 'instance>, index_of_required_heap: usize) -> Result<Self, VkError> {

        todo!(); // change that zero below!!!

        Ok(
            Self {
                device: device,
                heap_index: 0
            }
        )
    }
}
use crate::{
    device::{Device, DeviceOwned},
    result::VkError,
};

#[derive(Clone)]
pub enum MemoryHostVisibility {
    
}

#[derive(Clone)]
pub enum MemoryHostCoherence {
    Uncached, // host coherence is implemented via memory being uncached, as stated by vulkan specification: "uncached memory is always host coherent"
}

/**
 * If DeviceOnly(None) is specified a memory heap with both VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
 * and VK_MEMORY_PROPERTY_PROTECTED_BIT is selected.
 *
 * If HostLocal is specified a memory heap with at least VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
 * is selected, if HostLocal(None) is selected a heap that is NOT host-coherent will be selected,
 * otherwise if Some(Uncached) is selected than a memory heap with VK_MEMORY_PROPERTY_HOST_CACHED_BIT unset.
 */
#[derive(Clone)]
pub enum MemoryType {
    //HostVisible({}),
    DeviceLocal(Option<MemoryHostVisibility>),
    HostLocal(Option<MemoryHostCoherence>),
}

#[derive(Clone)]
pub struct ConcreteMemoryHeapDescriptor {
    memory_type: MemoryType,
    memory_minimum_size: u64,
}

impl ConcreteMemoryHeapDescriptor {
    pub fn memory_type(&self) -> MemoryType {
        self.memory_type.clone()
    }

    pub fn memory_minimum_size(&self) -> u64 {
        self.memory_minimum_size.clone()
    }

    pub fn new(memory_type: MemoryType, memory_minimum_size: u64) -> Self {
        Self {
            memory_type,
            memory_minimum_size
        }
    }
}

pub struct MemoryHeap<'ctx, 'instance, 'device>
where
    'ctx: 'instance,
    'instance: 'device,
{
    device: &'device Device<'ctx, 'instance>,
    descriptor: ConcreteMemoryHeapDescriptor,
    //created_queues: Mutex<u64>,
    heap_index: u32,
}

pub(crate) trait MemoryHeapOwned<'ctx, 'instance, 'device>
where
    'ctx: 'instance,
    'instance: 'device,
{
    fn get_parent_memory_heap(&self) -> &MemoryHeap<'ctx, 'instance, 'device>;
}

impl<'ctx, 'instance, 'device> DeviceOwned<'ctx, 'instance> for MemoryHeap<'ctx, 'instance, 'device>
where
    'ctx: 'instance,
    'instance: 'device,
{
    fn get_parent_device(&self) -> &'device Device<'ctx, 'instance> {
        self.device.clone()
    }
}

impl<'ctx, 'instance, 'device> Drop for MemoryHeap<'ctx, 'instance, 'device>
where
    'ctx: 'instance,
    'instance: 'device,
{
    fn drop(&mut self) {
        // Nothing to be done here
    }
}

impl<'ctx, 'instance, 'device> MemoryHeap<'ctx, 'instance, 'device>
where
    'ctx: 'instance,
    'instance: 'device,
{
    pub(crate) fn heap_index(&self) -> u32 {
        self.heap_index
    }

    pub fn new(
        device: &'device Device<'ctx, 'instance>,
        index_of_required_heap: usize,
    ) -> Result<Self, VkError> {
        match device.move_out_heap(index_of_required_heap) {
            Some((heap_index, descriptor)) => {
                Ok(Self {
                    device,
                    descriptor,
                    heap_index,
                })
            },
            None => {
                #[cfg(debug_assertions)]
                {
                    println!("Something bad happened while moving out the memory heap from the provided Device");
                    assert_eq!(true, false)
                }

                Err(VkError {})
            }
        }
    }
}

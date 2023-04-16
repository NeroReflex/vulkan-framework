use crate::{
    device::{Device, DeviceOwned},
    prelude::{VulkanError, VulkanResult},
};

use std::sync::Arc;

#[derive(Clone)]
pub enum MemoryHostVisibility {}

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
        self.memory_minimum_size
    }

    pub fn new(memory_type: MemoryType, memory_minimum_size: u64) -> Self {
        Self {
            memory_type,
            memory_minimum_size,
        }
    }
}

pub struct MemoryHeap {
    device: Arc<Device>,
    descriptor: ConcreteMemoryHeapDescriptor,
    heap_index: u32,
    heap_type_index: u32,
}

pub trait MemoryHeapOwned {
    fn get_parent_memory_heap(&self) -> Arc<MemoryHeap>;
}

impl DeviceOwned for MemoryHeap {
    fn get_parent_device(&self) -> Arc<Device> {
        self.device.clone()
    }
}

impl Drop for MemoryHeap {
    fn drop(&mut self) {
        // Nothing to be done here
    }
}

impl MemoryHeap {
    pub fn memory_type(&self) -> MemoryType {
        self.descriptor.memory_type.clone()
    }

    pub fn total_size(&self) -> u64 {
        self.descriptor.memory_minimum_size
    }

    pub(crate) fn type_index(&self) -> u32 {
        self.heap_type_index
    }

    pub(crate) fn heap_index(&self) -> u32 {
        self.heap_index
    }

    pub fn new(
        device: Arc<Device>,
        descriptor: ConcreteMemoryHeapDescriptor,
    ) -> VulkanResult<Arc<Self>> {
        match device.search_adequate_heap(&descriptor) {
            Some((heap_index, heap_type_index)) => Ok(Arc::new(Self {
                device,
                descriptor,
                heap_index,
                heap_type_index,
            })),
            None => {
                #[cfg(debug_assertions)]
                {
                    println!("A suitable memory heap was not found on the specified Device");
                }

                Err(VulkanError::Unspecified)
            }
        }
    }
}

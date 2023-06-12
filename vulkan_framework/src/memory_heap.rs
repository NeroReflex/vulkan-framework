use crate::{
    device::{Device, DeviceOwned},
    prelude::{VulkanError, VulkanResult},
};

use std::sync::Arc;

#[derive(PartialEq, Eq, Copy, Clone)]
pub struct MemoryHostVisibility {
    cached: bool,
}

impl MemoryHostVisibility {
    pub fn cached(&self) -> bool {
        self.cached
    }

    pub fn new(cached: bool) -> Self {
        Self { cached }
    }
}

#[derive(Copy, Clone, PartialEq, Eq)]
pub enum MemoryHostCoherence {
    Uncached, // host coherence is implemented via memory being uncached, as stated by vulkan specification: "uncached memory is always host coherent"
}

/**
 * If DeviceOnly(None) is specified the selected memory heap will have:
 *  - VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT bit set
 *  - VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT bit NOT set
 *
 * If HostLocal is specified a memory heap with at least VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
 * is selected, if HostLocal(None) is selected a heap that is NOT host-coherent will be selected,
 * otherwise if Some(Uncached) is selected than a memory heap with VK_MEMORY_PROPERTY_HOST_CACHED_BIT unset.
 */
#[derive(Copy, Clone, PartialEq, Eq)]
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
        self.memory_type
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
    heap_property_flags: u32,
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
    pub fn is_host_mappable(&self) -> bool {
        ash::vk::MemoryPropertyFlags::from_raw(self.heap_property_flags)
            .contains(ash::vk::MemoryPropertyFlags::HOST_VISIBLE)
    }

    pub fn heap_index(&self) -> u32 {
        self.heap_index
    }

    pub fn memory_type(&self) -> MemoryType {
        self.descriptor.memory_type
    }

    pub fn total_size(&self) -> u64 {
        self.descriptor.memory_minimum_size
    }

    pub(crate) fn type_index(&self) -> u32 {
        self.heap_type_index
    }

    pub fn check_memory_requirements_are_satified(&self, _memory_type_bits: u32) -> bool {
        // TODO: check if this heap satisfy memory requirements...
        // memoryTypeBits is a bitmask and contains one bit set for every supported memory type for the resource. Bit i is set if and only if the memory type i in the VkPhysicalDeviceMemoryProperties structure for the physical device is supported for the resource.

        true
    }

    pub fn new(
        device: Arc<Device>,
        descriptor: ConcreteMemoryHeapDescriptor,
    ) -> VulkanResult<Arc<Self>> {
        match device.search_adequate_heap(&descriptor) {
            Some((heap_index, heap_type_index, heap_property_flags)) => Ok(Arc::new(Self {
                device,
                descriptor,
                heap_index,
                heap_type_index,
                heap_property_flags,
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

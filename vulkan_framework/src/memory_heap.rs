use crate::{
    device::{Device, DeviceOwned},
    instance::InstanceOwned,
    memory_requiring::AllocationRequiring,
    prelude::{FrameworkError, VulkanError, VulkanResult},
};

use std::{sync::Arc, u32};

#[derive(Debug, PartialEq, Eq, Copy, Clone)]
pub enum MemoryHostVisibility {
    MemoryHostVisibile { cached: bool },
    MemoryHostHidden,
}

impl MemoryHostVisibility {
    pub fn visible(cached: bool) -> Self {
        Self::MemoryHostVisibile { cached }
    }

    pub fn hidden() -> Self {
        Self::MemoryHostHidden
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum MemoryHostCoherence {
    // host coherence is implemented via memory being uncached, as stated by vulkan specification:
    // "uncached memory is always host coherent"
    Uncached,
    Coherent,
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
/*
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum MemoryType {
    DeviceLocal(Option<MemoryHostVisibility>),
    HostLocal(Option<MemoryHostCoherence>),
}
*/

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[repr(transparent)]
pub struct MemoryType(crate::ash::vk::MemoryPropertyFlags);

impl MemoryType {
    pub fn device_local() -> Self {
        Self(crate::ash::vk::MemoryPropertyFlags::DEVICE_LOCAL)
    }

    pub fn host_visible() -> Self {
        Self(crate::ash::vk::MemoryPropertyFlags::HOST_VISIBLE)
    }

    pub fn device_local_and_host_visible() -> Self {
        Self(
            crate::ash::vk::MemoryPropertyFlags::DEVICE_LOCAL
                | crate::ash::vk::MemoryPropertyFlags::HOST_VISIBLE,
        )
    }

    pub fn host_visible_and_coherent() -> Self {
        Self(
            crate::ash::vk::MemoryPropertyFlags::HOST_VISIBLE
                | crate::ash::vk::MemoryPropertyFlags::HOST_COHERENT,
        )
    }

    pub fn host_cached() -> Self {
        Self(crate::ash::vk::MemoryPropertyFlags::HOST_CACHED)
    }
}

impl From<crate::ash::vk::MemoryPropertyFlags> for MemoryType {
    fn from(value: crate::ash::vk::MemoryPropertyFlags) -> Self {
        Self(value)
    }
}

impl MemoryType {
    fn satisfacted_by(&self, property_flags: crate::ash::vk::MemoryPropertyFlags) -> bool {
        (property_flags.as_raw() & self.0.as_raw()) == self.0.as_raw()
    }
}

#[derive(Debug, PartialEq, Eq, Clone)]
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

/// Represents memory requirements as reported by the driver
#[derive(Debug, Copy, Clone)]
pub struct MemoryRequirements {
    memory_type_bits_requirement: u32,
}

impl Default for MemoryRequirements {
    fn default() -> Self {
        Self {
            memory_type_bits_requirement: u32::MAX,
        }
    }
}

impl<T> From<&T> for MemoryRequirements
where
    T: AllocationRequiring,
{
    fn from(resource: &T) -> Self {
        let memory_type_bits_requirement = resource.allocation_requirements().memory_type_bits();
        Self {
            memory_type_bits_requirement,
        }
    }
}

impl TryFrom<&[&dyn AllocationRequiring]> for MemoryRequirements {
    type Error = VulkanError;

    fn try_from(value: &[&dyn AllocationRequiring]) -> Result<Self, Self::Error> {
        let mut memory_type_bits_requirement: u32 = u32::MAX;
        for resource in value.as_ref().iter() {
            memory_type_bits_requirement &= resource.allocation_requirements().memory_type_bits();
        }

        if memory_type_bits_requirement == 0u32 {
            return Err(VulkanError::Framework(
                FrameworkError::IncompatibleResources,
            ));
        }

        Ok(Self {
            memory_type_bits_requirement,
        })
    }
}

impl<T> TryFrom<&[T]> for MemoryRequirements
where
    T: AllocationRequiring,
{
    type Error = VulkanError;

    fn try_from(value: &[T]) -> Result<Self, Self::Error> {
        let mut memory_type_bits_requirement: u32 = u32::MAX;
        for resource in value.as_ref().iter() {
            memory_type_bits_requirement &= resource.allocation_requirements().memory_type_bits();
        }

        if memory_type_bits_requirement == 0u32 {
            return Err(VulkanError::Framework(
                FrameworkError::IncompatibleResources,
            ));
        }

        Ok(Self {
            memory_type_bits_requirement,
        })
    }
}

pub struct MemoryHeap {
    device: Arc<Device>,
    descriptor: ConcreteMemoryHeapDescriptor,
    heap_index: u32,
    heap_type_index: u32,
    heap_property_flags: ash::vk::MemoryPropertyFlags,
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
        self.heap_property_flags
            .contains(ash::vk::MemoryPropertyFlags::HOST_VISIBLE)
    }

    pub fn is_coherent(&self) -> bool {
        self.heap_property_flags
            .contains(ash::vk::MemoryPropertyFlags::HOST_COHERENT)
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

    pub fn suitable_memory_type(&self, requirements: &MemoryRequirements) -> bool {
        self.check_memory_type_bits_are_satified(requirements.memory_type_bits_requirement)
    }

    pub fn check_memory_type_bits_are_satified(&self, memory_type_bits_requirement: u32) -> bool {
        // Check if this heap satisfy memory requirements...
        // memoryTypeBits is a bitmask and contains one bit set for every supported memory type for the resource.
        // Bit i is set if and only if the memory type i in the VkPhysicalDeviceMemoryProperties structure
        // for the physical device is supported for the resource.
        Self::check_for_memory_type_bits(self.heap_type_index, memory_type_bits_requirement)
    }

    pub fn check_for_memory_type_bits(
        memory_index: u32,
        memory_type_bits_requirement: u32,
    ) -> bool {
        let memory_type_bits = 1u32 << memory_index;
        let memory_type_bits_satisfied = memory_type_bits_requirement & memory_type_bits;
        memory_type_bits_satisfied != 0u32
    }

    /// Derived from the official vulkan specification (findProperties method):
    /// https://registry.khronos.org/vulkan/specs/latest/html/vkspec.html#memory-device
    pub(crate) fn search_adequate_heap(
        device_memory_properties: &ash::vk::PhysicalDeviceMemoryProperties,
        current_requested_memory_heap_descriptor: &ConcreteMemoryHeapDescriptor,
        memory_type_bits_requirement: u32,
    ) -> Option<(u32, u32, ash::vk::MemoryPropertyFlags)> {
        let requested_size = current_requested_memory_heap_descriptor.memory_minimum_size();

        'suitable_heap_search: for memory_index in 0..device_memory_properties.memory_type_count {
            let heap_descriptor = &device_memory_properties.memory_types[memory_index as usize];

            // discard heaps that are too small
            if device_memory_properties.memory_heaps[heap_descriptor.heap_index as usize].size
                < requested_size
            {
                #[cfg(debug_assertions)]
                {
                    println!("Skipped memory heap due to being too small");
                }
                continue 'suitable_heap_search;
            }

            if !current_requested_memory_heap_descriptor
                .memory_type()
                .satisfacted_by(heap_descriptor.property_flags)
            {
                continue 'suitable_heap_search;
            }

            // If I'm here the previous search has find that the current heap is suitable...
            if Self::check_for_memory_type_bits(memory_index, memory_type_bits_requirement) {
                return Some((
                    heap_descriptor.heap_index,
                    memory_index,
                    heap_descriptor.property_flags,
                ));
            }
        }
        None
    }

    pub fn new(
        device: Arc<Device>,
        descriptor: ConcreteMemoryHeapDescriptor,
        requirements: MemoryRequirements,
    ) -> VulkanResult<Arc<Self>> {
        let device_memory_properties = unsafe {
            device
                .get_parent_instance()
                .ash_handle()
                .get_physical_device_memory_properties(device.physical_device.to_owned())
        };

        let adequate_heap = Self::search_adequate_heap(
            &device_memory_properties,
            &descriptor,
            requirements.memory_type_bits_requirement,
        );

        let (heap_index, heap_type_index, heap_property_flags) = match adequate_heap {
            Some(h) => h,
            None => {
                return Err(VulkanError::Framework(
                    FrameworkError::NoSuitableMemoryHeapFound,
                ))
            }
        };

        Ok(Arc::new(Self {
            device,
            descriptor,
            heap_index,
            heap_type_index,
            heap_property_flags,
        }))
    }
}

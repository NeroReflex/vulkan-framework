use ash::vk;

use crate::{
    device::DeviceOwned,
    instance::InstanceOwned,
    memory_allocator::*,
    memory_heap::{MemoryHeap, MemoryHeapOwned},
    prelude::{FrameworkError, VulkanError, VulkanResult},
};

use std::{
    mem::size_of,
    ops::{Deref, DerefMut},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum MemoryPoolFeature {
    DeviceAddressable,
}

#[derive(Debug, Default, Copy, Clone, PartialEq, Eq)]
pub struct MemoryPoolFeatures {
    device_addressable: bool,
}

impl MemoryPoolFeatures {
    pub fn device_addressable(&self) -> bool {
        self.device_addressable
    }

    pub fn new(device_addressable: bool) -> Self {
        Self { device_addressable }
    }
}

impl From<&[MemoryPoolFeature]> for MemoryPoolFeatures {
    fn from(features: &[MemoryPoolFeature]) -> Self {
        Self::new(features.contains(&MemoryPoolFeature::DeviceAddressable))
    }
}

pub struct MemoryPool {
    memory_heap: Arc<MemoryHeap>,
    allocator: Arc<dyn MemoryAllocator>,
    memory: ash::vk::DeviceMemory,
    features: MemoryPoolFeatures,
    mapped: AtomicBool,
}

impl AsRef<MemoryPool> for MemoryPool {
    fn as_ref(&self) -> &MemoryPool {
        self
    }
}

impl MemoryHeapOwned for MemoryPool {
    fn get_parent_memory_heap(&self) -> Arc<crate::memory_heap::MemoryHeap> {
        self.memory_heap.clone()
    }
}

/// Represents a resource that has been successfully allocated on a `MemoryPool`
pub trait MemoryPoolBacked {
    /// Get the memory pool that is backing the allocation
    fn get_backing_memory_pool(&self) -> Arc<MemoryPool>;

    /// Get the allocation offset
    fn allocation_offset(&self) -> u64;

    /// Get the allocation size
    fn allocation_size(&self) -> u64;
}

impl Drop for MemoryPool {
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

impl MemoryPool {
    pub fn native_handle(&self) -> u64 {
        ash::vk::Handle::as_raw(self.memory)
    }

    pub fn get_memory_allocator(&self) -> Arc<dyn MemoryAllocator> {
        self.allocator.clone()
    }

    pub fn features(&self) -> MemoryPoolFeatures {
        self.features
    }

    pub(crate) fn ash_handle(&self) -> ash::vk::DeviceMemory {
        self.memory
    }

    pub fn support_features(&self, features: &MemoryPoolFeatures) -> bool {
        self.features() == *features
    }

    pub fn new(
        memory_heap: Arc<MemoryHeap>,
        allocator: Arc<dyn MemoryAllocator>,
        features: MemoryPoolFeatures,
    ) -> VulkanResult<Arc<Self>> {
        let mut create_info = ash::vk::MemoryAllocateInfo::default()
            .allocation_size(allocator.total_size())
            .memory_type_index(memory_heap.type_index());

        if allocator.total_size() > memory_heap.total_size() {
            return Err(VulkanError::Framework(
                crate::prelude::FrameworkError::UnsuitableMemoryHeapForAllocator(
                    allocator.total_size(),
                    memory_heap.total_size(),
                ),
            ));
        }

        let device = memory_heap.get_parent_device();

        // the flag I want is only available since vulkan version 1.2
        let mut memory_flags = ash::vk::MemoryAllocateFlagsInfo::default()
            .flags(ash::vk::MemoryAllocateFlags::DEVICE_ADDRESS);

        if features.device_addressable() {
            create_info.p_next =
                &mut memory_flags as *mut ash::vk::MemoryAllocateFlagsInfo as *mut std::ffi::c_void;
        }

        let mapped = AtomicBool::new(false);

        let memory = unsafe {
            device.ash_handle().allocate_memory(
                &create_info,
                device.get_parent_instance().get_alloc_callbacks(),
            )?
        };

        Ok(Arc::new(Self {
            memory_heap,
            allocator,
            memory,
            features,
            mapped,
        }))
    }
}

pub struct MemoryMap {
    memory_pool: Arc<MemoryPool>,
    ptr: *mut std::ffi::c_void,
}

impl Drop for MemoryMap {
    fn drop(&mut self) {
        let memory_heap = self.memory_pool.get_parent_memory_heap();
        let device = memory_heap.get_parent_device();

        let old_val = self
            .memory_pool
            .mapped
            .compare_exchange(true, false, Ordering::SeqCst, Ordering::SeqCst)
            .unwrap();

        assert!(old_val);

        unsafe { device.ash_handle().unmap_memory(self.memory_pool.memory) }
    }
}

impl MemoryMap {
    pub fn new(memory_pool: Arc<MemoryPool>) -> VulkanResult<Self> {
        let memory_heap = memory_pool.get_parent_memory_heap();
        let device = memory_heap.get_parent_device();

        if !memory_heap.is_host_mappable() {
            return Err(VulkanError::Framework(
                crate::prelude::FrameworkError::MemoryNotMappableError,
            ));
        }

        let old_val = memory_pool
            .mapped
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .map_err(|_| VulkanError::Framework(FrameworkError::MemoryPoolAlreadyMapped))?;

        assert!(!old_val);

        let ptr = unsafe {
            device.ash_handle().map_memory(
                memory_pool.memory,
                0,
                memory_pool.allocator.total_size(),
                vk::MemoryMapFlags::empty(),
            )
        }?;

        Ok(Self { memory_pool, ptr })
    }

    pub fn range<'s, T>(
        &'s self,
        resource: impl AsRef<dyn MemoryPoolBacked>,
    ) -> VulkanResult<MemoryMappedRange<'s, T>>
    where
        T: Sized,
    {
        let res = resource.as_ref();

        // check the resource is from the very same memory pool
        if Arc::as_ptr(&res.get_backing_memory_pool()) != Arc::as_ptr(&self.memory_pool) {
            return Err(VulkanError::Framework(FrameworkError::WrongMemoryPool));
        }

        // check that at least one element exists, otherwise from_raw_parts will panic
        // and deref will access more data than what it is available
        if (res.allocation_size() as u128) < (std::mem::size_of::<T>() as u128) {
            return Err(VulkanError::Framework(FrameworkError::ResourceTooSmall));
        }

        Ok(MemoryMappedRange {
            content: unsafe { self.ptr.add(res.allocation_offset() as usize) } as *mut T,
            written: false,
            memory_map: self,
            offset: res.allocation_offset(),
            size: res.allocation_size(),
        })
    }
}

pub struct MemoryMappedRange<'map, T>
where
    T: Sized,
{
    content: *mut T,
    written: bool,
    memory_map: &'map MemoryMap,
    offset: ash::vk::DeviceSize,
    size: ash::vk::DeviceSize,
}

impl<'mem, T> Deref for MemoryMappedRange<'mem, T>
where
    T: Sized,
{
    type Target = T;

    fn deref(&self) -> &Self::Target {
        unsafe { &*self.content }
    }
}

impl<'mem, T> DerefMut for MemoryMappedRange<'mem, T>
where
    T: Sized,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.written = true;
        unsafe { &mut *self.content }
    }
}

impl<'mem, T> Drop for MemoryMappedRange<'mem, T>
where
    T: Sized,
{
    fn drop(&mut self) {
        if self.written
            && !self
                .memory_map
                .memory_pool
                .get_parent_memory_heap()
                .is_coherent()
        {
            let mapped_mem_range = ash::vk::MappedMemoryRange::default()
                .memory(self.memory_map.memory_pool.memory)
                .offset(self.offset)
                .size(self.size);

            unsafe {
                self.memory_map
                    .memory_pool
                    .get_parent_memory_heap()
                    .get_parent_device()
                    .ash_handle()
                    .flush_mapped_memory_ranges([mapped_mem_range].as_slice())
                    .unwrap()
            };
        }
    }
}

impl<'mem, T> MemoryMappedRange<'mem, T>
where
    T: Sized,
{
    pub fn as_slice(&self) -> &'mem [T] {
        unsafe { std::slice::from_raw_parts(self.content, (self.size as usize) / size_of::<T>()) }
    }

    pub fn as_mut_slice(&mut self) -> &'mem mut [T] {
        self.written = true;
        unsafe {
            std::slice::from_raw_parts_mut(self.content, (self.size as usize) / size_of::<T>())
        }
    }
}

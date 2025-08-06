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

pub trait MemoryPoolBacked {
    fn get_backing_memory_pool(&self) -> Arc<MemoryPool>;

    fn allocation_offset(&self) -> u64;

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

    pub fn write_raw_data<T>(&self, offset: u64, src: &[T]) -> VulkanResult<()>
    where
        T: Copy + Sized,
    {
        let device = self.get_parent_memory_heap().get_parent_device();

        if !self.get_parent_memory_heap().is_host_mappable() {
            return Err(VulkanError::Framework(
                crate::prelude::FrameworkError::MapMemoryError,
            ));
        }

        let ptr = unsafe {
            device.ash_handle().map_memory(
                self.memory,
                offset,
                (src.len() + size_of::<T>()) as u64,
                vk::MemoryMapFlags::empty(),
            )
        }?;

        // copy raw from data to ptr
        let mapped_typed_ptr = unsafe { std::slice::from_raw_parts_mut(ptr as *mut T, src.len()) };
        mapped_typed_ptr.copy_from_slice(src);

        unsafe { device.ash_handle().unmap_memory(self.memory) }

        Ok(())
    }

    pub fn read_raw_data<T>(&self, offset: u64, size: u64) -> VulkanResult<Vec<T>>
    where
        T: Copy,
    {
        let device = self.get_parent_memory_heap().get_parent_device();

        if !self.get_parent_memory_heap().is_host_mappable() {
            return Err(VulkanError::Framework(
                crate::prelude::FrameworkError::MapMemoryError,
            ));
        }

        let ptr = unsafe {
            device
                .ash_handle()
                .map_memory(self.memory, offset, size, vk::MemoryMapFlags::empty())
        }?;

        let slice =
            std::ptr::slice_from_raw_parts(ptr as *const T, (size as usize) / size_of::<T>());

        let data = unsafe { (*slice).to_vec() };

        unsafe { device.ash_handle().unmap_memory(self.memory) }

        Ok(data)
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

        //if self.memory_pool.get_parent_memory_heap().type_index()

        unsafe { device.ash_handle().unmap_memory(self.memory_pool.memory) }
    }
}

impl MemoryMap {
    pub fn new(memory_pool: Arc<MemoryPool>) -> VulkanResult<Self> {
        let memory_heap = memory_pool.get_parent_memory_heap();
        let device = memory_heap.get_parent_device();

        if !memory_heap.is_host_mappable() {
            return Err(VulkanError::Framework(
                crate::prelude::FrameworkError::MapMemoryError,
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

    pub fn as_slice<T>(&self, resource: impl AsRef<dyn MemoryPoolBacked>) -> VulkanResult<&[T]>
    where
        T: Sized,
    {
        let res = resource.as_ref();

        let slice = unsafe {
            std::slice::from_raw_parts(
                self.ptr.add(res.allocation_offset() as usize) as *const T,
                (res.allocation_size() as usize) / size_of::<T>(),
            )
        };

        Ok(slice)
    }

    pub fn as_mut_slice<T>(
        &mut self,
        resource: impl AsRef<dyn MemoryPoolBacked>,
    ) -> VulkanResult<&mut [T]>
    where
        T: Sized,
    {
        let res = resource.as_ref();

        let slice = unsafe {
            std::slice::from_raw_parts_mut(
                self.ptr.add(res.allocation_offset() as usize) as *mut T,
                (res.allocation_size() as usize) / size_of::<T>(),
            )
        };

        Ok(slice)
    }

    pub fn as_mut_slice_with_size<T>(
        &mut self,
        resource: impl AsRef<dyn MemoryPoolBacked>,
        size: u64,
    ) -> VulkanResult<&mut [T]>
    where
        T: Sized,
    {
        let res = resource.as_ref();

        if size > res.allocation_size() {
            todo!()
        }

        let slice = unsafe {
            std::slice::from_raw_parts_mut(
                self.ptr.add(res.allocation_offset() as usize) as *mut T,
                (size as usize) / size_of::<T>(),
            )
        };

        Ok(slice)
    }
}

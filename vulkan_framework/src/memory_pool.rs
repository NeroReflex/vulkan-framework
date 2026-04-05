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
    os::fd::OwnedFd,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum MemoryPoolFeature {
    DeviceAddressable,
    Exportable,
}

#[derive(Debug, Default, Copy, Clone, PartialEq, Eq)]
pub struct MemoryPoolFeatures {
    device_addressable: bool,
    exportable: bool,
}

impl MemoryPoolFeatures {
    pub fn device_addressable(&self) -> bool {
        self.device_addressable
    }

    pub fn exportable(&self) -> bool {
        self.exportable
    }

    pub fn new(device_addressable: bool) -> Self {
        Self {
            device_addressable,
            exportable: false,
        }
    }

    pub fn with_exportable(mut self) -> Self {
        self.exportable = true;
        self
    }
}

impl From<&[MemoryPoolFeature]> for MemoryPoolFeatures {
    fn from(features: &[MemoryPoolFeature]) -> Self {
        let mut f = Self::new(features.contains(&MemoryPoolFeature::DeviceAddressable));
        f.exportable = features.contains(&MemoryPoolFeature::Exportable);
        f
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

        let mut export_info = ash::vk::ExportMemoryAllocateInfo::default()
            .handle_types(ash::vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD);

        if features.device_addressable() {
            create_info.p_next =
                &mut memory_flags as *mut ash::vk::MemoryAllocateFlagsInfo as *mut std::ffi::c_void;
        }

        if features.exportable() {
            export_info.p_next = create_info.p_next;
            create_info.p_next =
                &mut export_info as *mut ash::vk::ExportMemoryAllocateInfo as *mut std::ffi::c_void;
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

    /// Export this memory pool's device memory as a POSIX file descriptor.
    ///
    /// Requires that the pool was created with `Exportable` feature and that the
    /// device has `VK_KHR_external_memory_fd` enabled.
    pub fn export_fd(&self) -> VulkanResult<OwnedFd> {
        let device = self.memory_heap.get_parent_device();
        let ext = device
            .ash_ext_external_memory_fd_khr()
            .as_ref()
            .ok_or(VulkanError::MissingExtension(
                "VK_KHR_external_memory_fd".into(),
            ))?;

        let get_fd_info = ash::vk::MemoryGetFdInfoKHR::default()
            .memory(self.memory)
            .handle_type(ash::vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD);

        let fd = unsafe { ext.get_memory_fd(&get_fd_info) }?;

        // SAFETY: vkGetMemoryFdKHR returns a new fd that we now own.
        Ok(unsafe { std::os::fd::FromRawFd::from_raw_fd(fd) })
    }

    /// Import device memory from a POSIX file descriptor.
    ///
    /// The fd is consumed (ownership transfers to the Vulkan driver).
    /// The caller must ensure `size` and `memory_type_index` match the
    /// exporter's allocation.
    pub fn import_from_fd(
        memory_heap: Arc<MemoryHeap>,
        allocator: Arc<dyn MemoryAllocator>,
        fd: OwnedFd,
    ) -> VulkanResult<Arc<Self>> {
        let device = memory_heap.get_parent_device();
        if device.ash_ext_external_memory_fd_khr().is_none() {
            return Err(VulkanError::MissingExtension(
                "VK_KHR_external_memory_fd".into(),
            ));
        }

        // Transfer ownership of the fd to Vulkan — use into_raw_fd so Drop doesn't close it.
        let raw_fd = std::os::fd::IntoRawFd::into_raw_fd(fd);

        let mut import_info = ash::vk::ImportMemoryFdInfoKHR::default()
            .handle_type(ash::vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD)
            .fd(raw_fd);

        let mut create_info = ash::vk::MemoryAllocateInfo::default()
            .allocation_size(allocator.total_size())
            .memory_type_index(memory_heap.type_index());

        create_info.p_next =
            &mut import_info as *mut ash::vk::ImportMemoryFdInfoKHR as *mut std::ffi::c_void;

        let memory = unsafe {
            device.ash_handle().allocate_memory(
                &create_info,
                device.get_parent_instance().get_alloc_callbacks(),
            )?
        };

        let features = MemoryPoolFeatures::default();
        let mapped = AtomicBool::new(false);

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

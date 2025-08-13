use ash::vk::Handle;

use crate::{
    device::{Device, DeviceOwned},
    instance::InstanceOwned,
    memory_allocator::SuccessfulAllocation,
    memory_heap::MemoryHeapOwned,
    memory_management::UnallocatedResource,
    memory_pool::{MemoryPool, MemoryPoolBacked},
    memory_requiring::{AllocationRequirements, AllocationRequiring},
    prelude::{FrameworkError, VulkanError, VulkanResult},
    queue_family::QueueFamily,
};

use std::{fmt::Debug, sync::Arc};

/**
 * Provided by VK_KHR_acceleration_structure
 */
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum BufferUseAsAccelerationStructureKHR {
    AccelerationStructureStorage,
    AccelerationStructureBuildInputReadOnly,
}

impl BufferUseAsAccelerationStructureKHR {
    pub(crate) fn ash_usage(&self) -> ash::vk::BufferUsageFlags {
        match self {
            Self::AccelerationStructureStorage => {
                ash::vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
            }
            Self::AccelerationStructureBuildInputReadOnly => {
                ash::vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
            }
        }
    }
}

/**
 * Provided by VK_KHR_acceleration_structure
 */
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct BufferUsageAccelerationStructureKHR {
    usage: ash::vk::BufferUsageFlags,
}

impl From<&[BufferUseAsAccelerationStructureKHR]> for BufferUsageAccelerationStructureKHR {
    fn from(value: &[BufferUseAsAccelerationStructureKHR]) -> Self {
        let mut usage = ash::vk::BufferUsageFlags::empty();
        for flag in value.iter() {
            usage |= flag.ash_usage()
        }

        Self { usage }
    }
}

impl BufferUsageAccelerationStructureKHR {
    pub(crate) fn ash_usage(&self) -> ash::vk::BufferUsageFlags {
        self.usage.to_owned()
    }
}

/*
 * Provided by VK_KHR_ray_tracing_pipeline
 */
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum BufferUseAsRayTracingPipelineKHR {
    ShaderBindingTable,
}

impl BufferUseAsRayTracingPipelineKHR {
    pub(crate) fn ash_usage(&self) -> ash::vk::BufferUsageFlags {
        match self {
            Self::ShaderBindingTable => ash::vk::BufferUsageFlags::SHADER_BINDING_TABLE_KHR,
        }
    }
}

/*
 * Provided by VK_KHR_ray_tracing_pipeline
 */
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct BufferUsageRayTracingPipelineKHR {
    usage: ash::vk::BufferUsageFlags,
}

impl From<&[BufferUseAsRayTracingPipelineKHR]> for BufferUsageRayTracingPipelineKHR {
    fn from(value: &[BufferUseAsRayTracingPipelineKHR]) -> Self {
        let mut usage = ash::vk::BufferUsageFlags::empty();
        for flag in value.iter() {
            usage |= flag.ash_usage()
        }

        Self { usage }
    }
}

impl BufferUsageRayTracingPipelineKHR {
    pub(crate) fn ash_usage(&self) -> ash::vk::BufferUsageFlags {
        self.usage.to_owned()
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum BufferUseAs {
    TransferSrc,
    TransferDst,
    UniformTexelBuffer,
    StorageTexelBuffer,
    UniformBuffer,
    StorageBuffer,
    IndexBuffer,
    VertexBuffer,
    IndirectBuffer,
    AccelerationStructureKHR(BufferUsageAccelerationStructureKHR),
    RayTracing(BufferUsageRayTracingPipelineKHR),
}

impl From<&BufferUseAs> for ash::vk::BufferUsageFlags {
    fn from(val: &BufferUseAs) -> Self {
        match val {
            BufferUseAs::TransferSrc => ash::vk::BufferUsageFlags::TRANSFER_SRC,
            BufferUseAs::TransferDst => ash::vk::BufferUsageFlags::TRANSFER_DST,
            BufferUseAs::UniformTexelBuffer => ash::vk::BufferUsageFlags::UNIFORM_TEXEL_BUFFER,
            BufferUseAs::StorageTexelBuffer => ash::vk::BufferUsageFlags::STORAGE_TEXEL_BUFFER,
            BufferUseAs::UniformBuffer => ash::vk::BufferUsageFlags::UNIFORM_BUFFER,
            BufferUseAs::StorageBuffer => ash::vk::BufferUsageFlags::STORAGE_BUFFER,
            BufferUseAs::IndexBuffer => ash::vk::BufferUsageFlags::INDEX_BUFFER,
            BufferUseAs::VertexBuffer => ash::vk::BufferUsageFlags::VERTEX_BUFFER,
            BufferUseAs::IndirectBuffer => ash::vk::BufferUsageFlags::INDIRECT_BUFFER,
            BufferUseAs::AccelerationStructureKHR(
                buffer_usage_flags_acceleration_structure_khr,
            ) => buffer_usage_flags_acceleration_structure_khr.ash_usage(),
            BufferUseAs::RayTracing(buffer_usage_flags_ray_tracing_pipeline_khr) => {
                buffer_usage_flags_ray_tracing_pipeline_khr.ash_usage()
            }
        }
    }
}

#[derive(Debug, Default, Copy, Clone, PartialEq, Eq)]
#[repr(transparent)]
pub struct BufferUsage(ash::vk::BufferUsageFlags);

impl From<u32> for BufferUsage {
    fn from(value: u32) -> Self {
        Self(ash::vk::BufferUsageFlags::from_raw(value))
    }
}

impl From<crate::ash::vk::BufferUsageFlags> for BufferUsage {
    fn from(usage: crate::ash::vk::BufferUsageFlags) -> Self {
        Self(usage)
    }
}

impl From<BufferUsage> for crate::ash::vk::BufferUsageFlags {
    fn from(val: BufferUsage) -> Self {
        val.0.to_owned()
    }
}

impl From<&[BufferUseAs]> for BufferUsage {
    fn from(value: &[BufferUseAs]) -> Self {
        let mut usage = ash::vk::BufferUsageFlags::empty();
        for flag in value.iter() {
            usage |= flag.into()
        }

        Self(usage)
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct ConcreteBufferDescriptor {
    usage: ash::vk::BufferUsageFlags,
    size: ash::vk::DeviceSize,
}

impl ConcreteBufferDescriptor {
    pub fn new(usage: BufferUsage, size: u64) -> Self {
        Self {
            size: size as ash::vk::DeviceSize,
            usage: usage.into(),
        }
    }

    pub(crate) fn ash_size(&self) -> ash::vk::DeviceSize {
        self.size as ash::vk::DeviceSize
    }

    pub(crate) fn ash_usage(&self) -> ash::vk::BufferUsageFlags {
        self.usage
    }
}

pub trait BufferTrait: Send + Sync + DeviceOwned {
    fn size(&self) -> u64;

    fn native_handle(&self) -> u64;
}

pub struct Buffer {
    device: Arc<Device>,
    descriptor: ConcreteBufferDescriptor,
    buffer: ash::vk::Buffer,
}

impl Debug for Buffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Buffer {}", ash::vk::Buffer::as_raw(self.buffer))
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        let device = self.get_parent_device();

        unsafe {
            device.ash_handle().destroy_buffer(
                self.buffer,
                device.get_parent_instance().get_alloc_callbacks(),
            )
        }
    }
}

impl DeviceOwned for Buffer {
    fn get_parent_device(&self) -> Arc<Device> {
        self.device.clone()
    }
}

impl Buffer {
    pub(crate) fn ash_handle(&self) -> ash::vk::Buffer {
        self.buffer
    }

    pub(crate) fn descriptor(&self) -> &ConcreteBufferDescriptor {
        &self.descriptor
    }

    pub fn new(
        device: Arc<Device>,
        descriptor: ConcreteBufferDescriptor,
        sharing: Option<&[std::sync::Weak<QueueFamily>]>,
        debug_name: Option<&str>,
    ) -> VulkanResult<Self> {
        let mut queue_family_indices = Vec::<u32>::new();
        if let Some(weak_queue_family_iter) = sharing {
            for allowed_queue_family in weak_queue_family_iter {
                if let Some(queue_family) = allowed_queue_family.upgrade() {
                    let family_index = queue_family.get_family_index();
                    if !queue_family_indices.contains(&family_index) {
                        queue_family_indices.push(family_index);
                    }
                }
            }
        }

        assert!(descriptor.ash_size() > 0);

        let create_info = ash::vk::BufferCreateInfo::default()
            .size(descriptor.ash_size())
            .usage(descriptor.ash_usage())
            .sharing_mode(match queue_family_indices.len() <= 1 {
                true => ash::vk::SharingMode::EXCLUSIVE,
                false => ash::vk::SharingMode::CONCURRENT,
            })
            .queue_family_indices(queue_family_indices.as_ref());

        let buffer = unsafe {
            device.ash_handle().create_buffer(
                &create_info,
                device.get_parent_instance().get_alloc_callbacks(),
            )
        }?;

        let mut obj_name_bytes = vec![];
        if let Some(ext) = device.ash_ext_debug_utils_ext() {
            if let Some(name) = debug_name {
                for name_ch in name.as_bytes().iter() {
                    obj_name_bytes.push(*name_ch);
                }
                obj_name_bytes.push(0x00);

                unsafe {
                    let object_name =
                        std::ffi::CStr::from_bytes_with_nul_unchecked(obj_name_bytes.as_slice());
                    // set device name for debugging
                    let dbg_info = ash::vk::DebugUtilsObjectNameInfoEXT::default()
                        .object_handle(buffer)
                        .object_name(object_name);

                    if let Err(err) = ext.set_debug_utils_object_name(&dbg_info) {
                        #[cfg(debug_assertions)]
                        {
                            println!("Error setting the Debug name for the newly created Buffer, will use handle. Error: {}", err);
                        }
                    }
                }
            }
        }

        Ok(Self {
            device,
            descriptor,
            buffer,
        })
    }
}

impl AllocationRequiring for Buffer {
    fn allocation_requirements(&self) -> AllocationRequirements {
        let requirements_info =
            ash::vk::BufferMemoryRequirementsInfo2::default().buffer(self.ash_handle());

        let mut requirements = ash::vk::MemoryRequirements2::default();

        unsafe {
            self.get_parent_device()
                .ash_handle()
                .get_buffer_memory_requirements2(&requirements_info, &mut requirements);
        };

        // Force alignment to 256 bytes for quick test (Vulkan minimum is 16, 64 recommended)
        let forced_alignment = 256u64;
        AllocationRequirements::new(
            requirements.memory_requirements.memory_type_bits,
            requirements.memory_requirements.size,
            std::cmp::max(requirements.memory_requirements.alignment, forced_alignment),
        )
    }
}

pub struct AllocatedBuffer {
    memory_pool: Arc<MemoryPool>,
    reserved_memory_from_pool: SuccessfulAllocation,
    buffer: Buffer,
}

impl AllocatedBuffer {
    pub(crate) fn ash_handle(&self) -> ash::vk::Buffer {
        self.buffer.ash_handle()
    }

    pub(crate) fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    pub fn new(memory_pool: Arc<MemoryPool>, buffer: Buffer) -> VulkanResult<Arc<Self>> {
        let device = memory_pool.get_parent_memory_heap().get_parent_device();

        if buffer.get_parent_device() != memory_pool.get_parent_memory_heap().get_parent_device() {
            return Err(VulkanError::Framework(
                FrameworkError::MemoryHeapAndResourceNotFromTheSameDevice,
            ));
        }

        let requirements = buffer.allocation_requirements();

        if !memory_pool
            .get_parent_memory_heap()
            .check_memory_type_bits_are_satified(requirements.memory_type_bits())
        {
            return Err(VulkanError::Framework(
                FrameworkError::IncompatibleMemoryHeapType,
            ));
        }

        match memory_pool
            .get_memory_allocator()
            .alloc(requirements.size(), requirements.alignment())
        {
            Some(reserved_memory_from_pool) => {
                unsafe {
                    device.ash_handle().bind_buffer_memory(
                        buffer.ash_handle(),
                        memory_pool.ash_handle(),
                        reserved_memory_from_pool.offset_in_pool(),
                    )
                }.inspect_err(|err| {
                    println!("ERROR: Error allocating memory: {err}, probably this is due to an incorrect implementation of the memory allocation algorithm");
                })?;

                Ok(Arc::new(Self {
                    memory_pool,
                    reserved_memory_from_pool,
                    buffer,
                }))
            }
            None => Err(VulkanError::Framework(FrameworkError::MallocFail(
                UnallocatedResource::Buffer(buffer),
            ))),
        }
    }
}

impl Drop for AllocatedBuffer {
    fn drop(&mut self) {
        self.memory_pool
            .get_memory_allocator()
            .dealloc(&mut self.reserved_memory_from_pool)
    }
}

impl MemoryPoolBacked for AllocatedBuffer {
    fn get_backing_memory_pool(&self) -> Arc<MemoryPool> {
        self.memory_pool.clone()
    }

    fn allocation_offset(&self) -> u64 {
        self.reserved_memory_from_pool.offset_in_pool()
    }

    fn allocation_size(&self) -> u64 {
        self.buffer.descriptor().size
    }
}

impl DeviceOwned for AllocatedBuffer {
    fn get_parent_device(&self) -> Arc<Device> {
        self.buffer.get_parent_device()
    }
}

impl BufferTrait for AllocatedBuffer {
    fn size(&self) -> u64 {
        self.buffer.descriptor().ash_size()
    }

    fn native_handle(&self) -> u64 {
        ash::vk::Handle::as_raw(self.buffer.ash_handle())
    }
}

pub struct BufferSubresourceRange {
    buffer: Arc<dyn BufferTrait>,
    offset: u64,
    size: u64,
}

impl BufferSubresourceRange {
    #[inline]
    pub fn buffer(&self) -> Arc<dyn BufferTrait> {
        self.buffer.clone()
    }

    #[inline]
    pub fn offset(&self) -> u64 {
        self.offset.to_owned()
    }

    #[inline]
    pub fn size(&self) -> u64 {
        self.size.to_owned()
    }

    #[inline]
    pub fn new(buffer: Arc<dyn BufferTrait>, offset: u64, size: u64) -> Self {
        assert!(offset + size <= buffer.size());

        Self {
            buffer,
            offset,
            size,
        }
    }
}

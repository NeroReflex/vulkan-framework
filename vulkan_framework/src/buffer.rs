use crate::{
    device::{Device, DeviceOwned},
    instance::{InstanceAPIVersion, InstanceOwned},
    memory_allocator::{AllocationResult, MemoryAllocator},
    memory_heap::MemoryHeapOwned,
    memory_pool::{MemoryPool, MemoryPoolBacked},
    prelude::{VulkanError, VulkanResult},
    queue_family::QueueFamily,
};

use std::sync::Arc;

/**
 * Provided by VK_KHR_acceleration_structure
 */
#[derive(Copy, Clone, PartialEq, Eq)]
pub enum BufferUsageFlagAccelerationStructureKHR {
    AccelerationStructureStorage,
    AccelerationStructureBuildInputReadOnly,
}

/**
 * Provided by VK_KHR_acceleration_structure
 */
#[derive(Copy, Clone, PartialEq, Eq)]
pub struct BufferUsageFlagsAccelerationStructureKHR {
    acceleration_structure_storage: bool,
    acceleration_structure_build_input_read_only: bool,
}

impl BufferUsageFlagsAccelerationStructureKHR {
    pub fn from(flags: &[BufferUsageFlagAccelerationStructureKHR]) -> Self {
        Self {
            acceleration_structure_storage: flags
                .contains(&BufferUsageFlagAccelerationStructureKHR::AccelerationStructureStorage),
            acceleration_structure_build_input_read_only: flags.contains(
                &BufferUsageFlagAccelerationStructureKHR::AccelerationStructureBuildInputReadOnly,
            ),
        }
    }

    pub fn empty() -> Self {
        Self {
            acceleration_structure_storage: false,
            acceleration_structure_build_input_read_only: false,
        }
    }

    pub(crate) fn ash_usage(&self) -> ash::vk::BufferUsageFlags {
        (match self.acceleration_structure_storage {
            true => ash::vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR,
            false => ash::vk::BufferUsageFlags::from_raw(0),
        }) | (match self.acceleration_structure_build_input_read_only {
            true => ash::vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
            false => ash::vk::BufferUsageFlags::from_raw(0),
        })
    }
}

/*
 * Provided by VK_KHR_ray_tracing_pipeline
 */
#[derive(Copy, Clone, PartialEq, Eq)]
pub enum BufferUsageFlagRayTracingPipelineKHR {
    ShaderBindingTable,
}

/*
 * Provided by VK_KHR_ray_tracing_pipeline
 */
#[derive(Copy, Clone, PartialEq, Eq)]
pub struct BufferUsageFlagsRayTracingPipelineKHR {
    shader_binding_table: bool,
}

impl BufferUsageFlagsRayTracingPipelineKHR {
    pub fn from(flags: &[BufferUsageFlagRayTracingPipelineKHR]) -> Self {
        Self {
            shader_binding_table: flags
                .contains(&BufferUsageFlagRayTracingPipelineKHR::ShaderBindingTable),
        }
    }

    pub fn empty() -> Self {
        Self {
            shader_binding_table: false,
        }
    }

    pub(crate) fn ash_usage(&self) -> ash::vk::BufferUsageFlags {
        match self.shader_binding_table {
            true => ash::vk::BufferUsageFlags::SHADER_BINDING_TABLE_KHR,
            false => ash::vk::BufferUsageFlags::from_raw(0),
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq)]
pub enum BufferUsageFlag {
    TransferSrc,
    TransferDst,
    UniformTexelBuffer,
    StorageTexelBuffer,
    UniformBuffer,
    StorageBuffer,
    IndexBuffer,
    VertexBuffer,
    IndirectBuffer,
}

#[derive(Copy, Clone, PartialEq, Eq)]
pub struct BufferUsageFlagsSpecifier {
    transfer_src: bool,
    transfer_dst: bool,
    uniform_texel_buffer: bool,
    storage_texel_buffer: bool,
    uniform_buffer: bool,
    storage_buffer: bool,
    index_buffer: bool,
    vertex_buffer: bool,
    indirect_buffer: bool,
    acceleration_structure: BufferUsageFlagsAccelerationStructureKHR,
    ray_tracing: BufferUsageFlagsRayTracingPipelineKHR,
}

impl BufferUsageFlagsSpecifier {
    pub fn empty() -> Self {
        Self::from(&[], None, None)
    }

    pub fn from(
        flags: &[BufferUsageFlag],
        acceleration_structure_flags: Option<&[BufferUsageFlagAccelerationStructureKHR]>,
        ray_tracing_flags: Option<&[BufferUsageFlagRayTracingPipelineKHR]>,
    ) -> Self {
        Self {
            transfer_src: flags.contains(&BufferUsageFlag::TransferSrc),
            transfer_dst: flags.contains(&BufferUsageFlag::TransferDst),
            uniform_texel_buffer: flags.contains(&BufferUsageFlag::UniformTexelBuffer),
            storage_texel_buffer: flags.contains(&BufferUsageFlag::StorageTexelBuffer),
            uniform_buffer: flags.contains(&BufferUsageFlag::UniformBuffer),
            storage_buffer: flags.contains(&BufferUsageFlag::StorageBuffer),
            index_buffer: flags.contains(&BufferUsageFlag::IndexBuffer),
            vertex_buffer: flags.contains(&BufferUsageFlag::VertexBuffer),
            indirect_buffer: flags.contains(&BufferUsageFlag::IndirectBuffer),
            acceleration_structure: match acceleration_structure_flags {
                Some(flags) => BufferUsageFlagsAccelerationStructureKHR::from(flags),
                None => BufferUsageFlagsAccelerationStructureKHR::empty(),
            },
            ray_tracing: match ray_tracing_flags {
                Some(flags) => BufferUsageFlagsRayTracingPipelineKHR::from(flags),
                None => BufferUsageFlagsRayTracingPipelineKHR::empty(),
            },
        }
    }

    pub(crate) fn ash_usage(&self) -> ash::vk::BufferUsageFlags {
        (ash::vk::BufferUsageFlags::empty())
            | (match self.transfer_src {
                true => ash::vk::BufferUsageFlags::TRANSFER_SRC,
                false => ash::vk::BufferUsageFlags::from_raw(0),
            })
            | (match self.transfer_dst {
                true => ash::vk::BufferUsageFlags::TRANSFER_DST,
                false => ash::vk::BufferUsageFlags::from_raw(0),
            })
            | (match self.uniform_texel_buffer {
                true => ash::vk::BufferUsageFlags::UNIFORM_TEXEL_BUFFER,
                false => ash::vk::BufferUsageFlags::from_raw(0),
            })
            | (match self.storage_texel_buffer {
                true => ash::vk::BufferUsageFlags::STORAGE_TEXEL_BUFFER,
                false => ash::vk::BufferUsageFlags::from_raw(0),
            })
            | (match self.uniform_buffer {
                true => ash::vk::BufferUsageFlags::UNIFORM_BUFFER,
                false => ash::vk::BufferUsageFlags::from_raw(0),
            })
            | (match self.storage_buffer {
                true => ash::vk::BufferUsageFlags::STORAGE_BUFFER,
                false => ash::vk::BufferUsageFlags::from_raw(0),
            })
            | (match self.index_buffer {
                true => ash::vk::BufferUsageFlags::INDEX_BUFFER,
                false => ash::vk::BufferUsageFlags::from_raw(0),
            })
            | (match self.vertex_buffer {
                true => ash::vk::BufferUsageFlags::VERTEX_BUFFER,
                false => ash::vk::BufferUsageFlags::from_raw(0),
            })
            | (match self.indirect_buffer {
                true => ash::vk::BufferUsageFlags::INDIRECT_BUFFER,
                false => ash::vk::BufferUsageFlags::from_raw(0),
            })
            | self.acceleration_structure.ash_usage()
            | self.ray_tracing.ash_usage()
    }

    pub fn new(
        transfer_src: bool,
        transfer_dst: bool,
        uniform_texel_buffer: bool,
        storage_texel_buffer: bool,
        uniform_buffer: bool,
        storage_buffer: bool,
        index_buffer: bool,
        vertex_buffer: bool,
        indirect_buffer: bool,
        maybe_acceleration_structure: Option<BufferUsageFlagsAccelerationStructureKHR>,
        maybe_ray_tracing: Option<BufferUsageFlagsRayTracingPipelineKHR>,
    ) -> Self {
        Self {
            transfer_src,
            transfer_dst,
            uniform_texel_buffer,
            storage_texel_buffer,
            uniform_buffer,
            storage_buffer,
            index_buffer,
            vertex_buffer,
            indirect_buffer,
            acceleration_structure: match maybe_acceleration_structure {
                Some(flags) => flags,
                None => BufferUsageFlagsAccelerationStructureKHR::empty(),
            },
            ray_tracing: match maybe_ray_tracing {
                Some(flags) => flags,
                None => BufferUsageFlagsRayTracingPipelineKHR::empty(),
            },
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq)]
pub enum BufferUsage {
    Managed(BufferUsageFlagsSpecifier),
    Unmanaged(u32),
}

impl BufferUsage {
    pub fn empty() -> Self {
        Self::Unmanaged(0)
    }

    pub fn from_raw(flags: u32) -> Self {
        Self::Unmanaged(flags)
    }

    pub fn from(flags: BufferUsageFlagsSpecifier) -> Self {
        Self::Managed(flags)
    }

    pub(crate) fn ash_usage(&self) -> ash::vk::BufferUsageFlags {
        match self {
            Self::Managed(spec) => spec.ash_usage(),
            Self::Unmanaged(raw) => ash::vk::BufferUsageFlags::from_raw(*raw),
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq)]
pub struct ConcreteBufferDescriptor {
    usage: BufferUsage,
    size: ash::vk::DeviceSize,
}

impl ConcreteBufferDescriptor {
    pub fn new(
        usage: BufferUsage,
        size: u64
    ) -> Self {
        Self {
            size: size as ash::vk::DeviceSize,
            usage: usage
        }
    }

    pub(crate) fn ash_size(&self) -> ash::vk::DeviceSize {
        self.size as ash::vk::DeviceSize
    }

    pub(crate) fn ash_usage(&self) -> ash::vk::BufferUsageFlags {
        self.usage.ash_usage()
    }
}

pub trait BufferTrait: Send + Sync + DeviceOwned {
    fn size(&self) -> u64;

    fn native_handle(&self) -> u64;
}

pub struct Buffer
{
    memory_pool: Arc<MemoryPool>,
    reserved_memory_from_pool: AllocationResult,
    descriptor: ConcreteBufferDescriptor,
    buffer: ash::vk::Buffer,
}

impl Drop for Buffer
{
    fn drop(&mut self) {
        let device = self
            .memory_pool
            .get_parent_memory_heap()
            .get_parent_device();
        unsafe {
            device.ash_handle().destroy_buffer(
                self.buffer,
                device.get_parent_instance().get_alloc_callbacks(),
            )
        }

        self.memory_pool.get_memory_allocator().dealloc(&mut self.reserved_memory_from_pool)
    }
}

impl MemoryPoolBacked for Buffer
{
    fn get_backing_memory_pool(&self) -> Arc<MemoryPool> {
        self.memory_pool.clone()
    }

    fn allocation_offset(&self) -> u64 {
        self.reserved_memory_from_pool.offset_in_pool()
    }

    fn allocation_size(&self) -> u64 {
        self.reserved_memory_from_pool.size()
    }
}

impl DeviceOwned for Buffer
{
    fn get_parent_device(&self) -> Arc<Device> {
        self.memory_pool
            .get_parent_memory_heap()
            .get_parent_device()
    }
}

impl BufferTrait for Buffer
{
    fn size(&self) -> u64 {
        self.descriptor.ash_size()
    }

    fn native_handle(&self) -> u64 {
        ash::vk::Handle::as_raw(self.buffer)
    }
}

impl Buffer
{
    pub(crate) fn ash_handle(&self) -> ash::vk::Buffer {
        self.buffer
    }

    pub fn new(
        memory_pool: Arc<MemoryPool>,
        descriptor: ConcreteBufferDescriptor,
        sharing: Option<&[std::sync::Weak<QueueFamily>]>,
        debug_name: Option<&str>,
    ) -> VulkanResult<Arc<Self>> {
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

        let create_info = ash::vk::BufferCreateInfo::builder()
            .size(descriptor.ash_size())
            .usage(descriptor.ash_usage())
            .sharing_mode(match queue_family_indices.len() <= 1 {
                true => ash::vk::SharingMode::EXCLUSIVE,
                false => ash::vk::SharingMode::CONCURRENT,
            })
            .queue_family_indices(queue_family_indices.as_ref())
            .build();

        let device = memory_pool.get_parent_memory_heap().get_parent_device();

        let buffer = match unsafe {
            device.ash_handle().create_buffer(
                &create_info,
                device.get_parent_instance().get_alloc_callbacks(),
            )
        } {
            Ok(buffer) => buffer,
            Err(err) => {
                #[cfg(debug_assertions)]
                {
                    panic!("Error creating the buffer: {}", err)
                }

                return Err(VulkanError::Unspecified);
            }
        };

        let mut obj_name_bytes = vec![];
        if let Some(ext) = device.get_parent_instance().get_debug_ext_extension() {
            if let Some(name) = debug_name {
                for name_ch in name.as_bytes().iter() {
                    obj_name_bytes.push(*name_ch);
                }
                obj_name_bytes.push(0x00);

                unsafe {
                    let object_name =
                        std::ffi::CStr::from_bytes_with_nul_unchecked(obj_name_bytes.as_slice());
                    // set device name for debugging
                    let dbg_info = ash::vk::DebugUtilsObjectNameInfoEXT::builder()
                        .object_type(ash::vk::ObjectType::BUFFER)
                        .object_handle(ash::vk::Handle::as_raw(buffer))
                        .object_name(object_name)
                        .build();

                    match ext.set_debug_utils_object_name(device.ash_handle().handle(), &dbg_info) {
                        Ok(_) => {
                            #[cfg(debug_assertions)]
                            {
                                println!("Queue Debug object name changed");
                            }
                        }
                        Err(err) => {
                            #[cfg(debug_assertions)]
                            {
                                panic!("Error setting the Debug name for the newly created Buffer, will use handle. Error: {}", err);
                            }
                        }
                    }
                }
            }
        }

        unsafe {
            let requirements = if device.get_parent_instance().instance_vulkan_version()
                == InstanceAPIVersion::Version1_0
            {
                device.ash_handle().get_buffer_memory_requirements(buffer)
            } else {
                let requirements_info = ash::vk::BufferMemoryRequirementsInfo2::builder()
                    .buffer(buffer)
                    .build();

                let mut requirements = ash::vk::MemoryRequirements2::default();

                device
                    .ash_handle()
                    .get_buffer_memory_requirements2(&requirements_info, &mut requirements);

                requirements.memory_requirements
            };

            if !memory_pool.get_parent_memory_heap().check_memory_requirements_are_satified(requirements.memory_type_bits) {
                return Err(VulkanError::Unspecified)
            }

            match memory_pool.get_memory_allocator().alloc(requirements.size, requirements.alignment) {
                Some(reserved_memory_from_pool) => {
                    match device.ash_handle().bind_buffer_memory(
                        buffer,
                        memory_pool.ash_handle(),
                        reserved_memory_from_pool.offset_in_pool(),
                    ) {
                        Ok(_) => Ok(Arc::new(Self {
                            memory_pool,
                            reserved_memory_from_pool,
                            buffer,
                            descriptor,
                        })),
                        Err(err) => {
                            // the buffer will not let this function, destroy it or it will leak
                            device.ash_handle().destroy_buffer(
                                buffer,
                                device.get_parent_instance().get_alloc_callbacks(),
                            );

                            #[cfg(debug_assertions)]
                            {
                                panic!("Error allocating memory on the device: {}, probably this is due to an incorrect implementation of the memory allocation algorithm", err)
                            }

                            Err(VulkanError::Unspecified)
                        }
                    }
                }
                None => {
                    // the buffer will not let this function, destroy it or it will leak
                    device
                        .ash_handle()
                        .destroy_buffer(buffer, device.get_parent_instance().get_alloc_callbacks());

                    Err(VulkanError::Unspecified)
                }
            }
        }
    }
}

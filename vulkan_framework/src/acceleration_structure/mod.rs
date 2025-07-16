use std::sync::Arc;

use ash::vk::BuildAccelerationStructureModeKHR;
#[cfg(feature = "better_mutex")]
use parking_lot::{const_mutex, Mutex};

#[cfg(not(feature = "better_mutex"))]
use std::sync::Mutex;

use crate::{
    buffer::{AllocatedBuffer, Buffer, BufferTrait, BufferUsage, ConcreteBufferDescriptor},
    device::{Device, DeviceOwned},
    graphics_pipeline::AttributeType,
    instance::InstanceOwned,
    memory_heap::MemoryHeapOwned,
    memory_pool::MemoryPool,
    prelude::{FrameworkError, VulkanError, VulkanResult},
};

pub mod bottom_level;
pub mod scratch_buffer;
pub mod top_level;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum AllowedBuildingDevice {
    HostOnly,
    DeviceOnly,
    HostAndDevice,
}

impl AllowedBuildingDevice {
    #[inline]
    pub(crate) fn ash_flags(&self) -> ash::vk::AccelerationStructureBuildTypeKHR {
        match self {
            AllowedBuildingDevice::HostOnly => ash::vk::AccelerationStructureBuildTypeKHR::HOST,
            AllowedBuildingDevice::DeviceOnly => ash::vk::AccelerationStructureBuildTypeKHR::DEVICE,
            AllowedBuildingDevice::HostAndDevice => {
                ash::vk::AccelerationStructureBuildTypeKHR::HOST_OR_DEVICE
            }
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum VertexIndexing {
    None,
    UInt16,
    UInt32,
}

impl VertexIndexing {
    pub fn size(&self) -> u64 {
        match self {
            VertexIndexing::None => 0u64,
            VertexIndexing::UInt16 => 2u64,
            VertexIndexing::UInt32 => 4u64,
        }
    }

    pub(crate) fn ash_index_type(&self) -> ash::vk::IndexType {
        match self {
            VertexIndexing::None => ash::vk::IndexType::NONE_KHR,
            VertexIndexing::UInt16 => ash::vk::IndexType::UINT16,
            VertexIndexing::UInt32 => ash::vk::IndexType::UINT32,
        }
    }
}

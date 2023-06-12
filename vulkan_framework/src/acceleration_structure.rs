use std::sync::Arc;

use crate::{
    buffer::Buffer,
    device::DeviceOwned,
    memory_heap::MemoryHeapOwned,
    memory_pool::MemoryPool,
    prelude::{FrameworkError, VulkanError, VulkanResult},
};

#[derive(Copy, Clone, PartialEq, Eq)]
pub enum AllowedBuildingDevice {
    HostOnly,
    DeviceOnly,
    HostAndDevice,
}

impl AllowedBuildingDevice {
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

pub struct BottomLevelAccelerationStructure {
    handle: ash::vk::AccelerationStructureKHR,
    buffer: Arc<Buffer>,
    device_memory: ash::vk::DeviceMemory,
    allowed_building_devices: AllowedBuildingDevice,
}

impl BottomLevelAccelerationStructure {
    pub fn allowed_building_devices(&self) -> AllowedBuildingDevice {
        self.allowed_building_devices
    }

    pub fn new(
        memory_pool: Arc<MemoryPool>,
        allowed_building_devices: AllowedBuildingDevice,
    ) -> VulkanResult<Arc<Self>> {
        let device = memory_pool.get_parent_memory_heap().get_parent_device();

        if !memory_pool.features().device_addressable() {
            return Err(VulkanError::Framework(FrameworkError::Unknown(Some(
                String::from("Missing feature on MemoryPool: device_addressable need to be set"),
            ))));
        }

        let geometry = ash::vk::AccelerationStructureGeometryDataKHR {
            triangles: ash::vk::AccelerationStructureGeometryTrianglesDataKHR::builder()
                .index_type(ash::vk::IndexType::UINT32)
                .build(),
        };

        let as_geometry = ash::vk::AccelerationStructureGeometryKHR::builder()
            .geometry_type(ash::vk::GeometryTypeKHR::TRIANGLES)
            .geometry(geometry)
            .build();

        // From vulkan specs: https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkGetAccelerationStructureBuildSizesKHR.html
        // The srcAccelerationStructure, dstAccelerationStructure, and mode members of pBuildInfo are ignored.
        // Any VkDeviceOrHostAddressKHR members of pBuildInfo are ignored by this command,
        // except that the hostAddress member of VkAccelerationStructureGeometryTrianglesDataKHR::transformData
        // will be examined to check if it is NULL.
        let geometry_info = ash::vk::AccelerationStructureBuildGeometryInfoKHR::builder()
            .geometries(&[as_geometry])
            .flags(ash::vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
            .ty(ash::vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
            .build();

        match device.ash_ext_acceleration_structure_khr() {
            Some(as_ext) => {
                let _build_sizes = unsafe {
                    as_ext.get_acceleration_structure_build_sizes(
                        allowed_building_devices.ash_flags(),
                        &geometry_info,
                        &[32],
                    )
                };

                let _create_info = ash::vk::AccelerationStructureCreateInfoKHR::builder().build();
                //TODO: as_ext.create_acceleration_structure(create_info, device.get_parent_instance().get_alloc_callbacks())

                todo!()
            }
            None => Err(VulkanError::MissingExtension(String::from(
                "VK_KHR_acceleration_structure",
            ))),
        }
    }
}

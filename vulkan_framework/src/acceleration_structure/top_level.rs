use std::sync::Arc;

use crate::{
    acceleration_structure::{scratch_buffer::DeviceScratchBuffer, AllowedBuildingDevice},
    buffer::{AllocatedBuffer, Buffer, BufferTrait, BufferUsage, ConcreteBufferDescriptor},
    device::{Device, DeviceOwned},
    instance::InstanceOwned,
    memory_heap::MemoryHeapOwned,
    memory_pool::MemoryPool,
    prelude::{FrameworkError, VulkanError, VulkanResult},
};

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct TopLevelBLASGroupDecl {
    array_of_pointers: bool,
}

impl Default for TopLevelBLASGroupDecl {
    fn default() -> Self {
        Self::new()
    }
}

impl TopLevelBLASGroupDecl {
    pub(crate) fn ash_geometry(&self) -> ash::vk::AccelerationStructureGeometryKHR {
        ash::vk::AccelerationStructureGeometryKHR::default()
            // TODO: .flags()
            .geometry_type(ash::vk::GeometryTypeKHR::INSTANCES)
            .geometry(ash::vk::AccelerationStructureGeometryDataKHR {
                instances: ash::vk::AccelerationStructureGeometryInstancesDataKHR::default()
                    .array_of_pointers(self.array_of_pointers()),
            })
    }

    pub fn array_of_pointers(&self) -> bool {
        self.array_of_pointers
    }

    pub fn new() -> Self {
        Self {
            array_of_pointers: false,
        }
    }
}

pub struct TopLevelAccelerationStructure {
    blas_decl: TopLevelBLASGroupDecl,

    handle: ash::vk::AccelerationStructureKHR,
    acceleration_structure_device_addr: u64,

    buffer: Arc<dyn BufferTrait>,
    buffer_device_addr: u64,

    allowed_building_devices: AllowedBuildingDevice,

    device_build_scratch_buffer: Arc<DeviceScratchBuffer>,
}

impl Drop for TopLevelAccelerationStructure {
    fn drop(&mut self) {
        let device = self.buffer.get_parent_device();
        match self
            .buffer
            .get_parent_device()
            .ash_ext_acceleration_structure_khr()
        {
            Some(as_ext) => unsafe {
                as_ext.destroy_acceleration_structure(
                    self.handle,
                    device.get_parent_instance().get_alloc_callbacks(),
                )
            },
            None => todo!(),
        }
    }
}

impl TopLevelAccelerationStructure {
    /*
     * This function allows the user to estimate a minimum size for provided geometry info.
     *
     * If the operation is successful the tuple returned will be the TLAS (minimum) size
     */
    pub fn query_minimum_sizes(
        device: Arc<Device>,
        allowed_building_devices: AllowedBuildingDevice,
        geometries_decl: &[TopLevelBLASGroupDecl],
    ) -> VulkanResult<u64> {
        let Some(as_ext) = device.ash_ext_acceleration_structure_khr() else {
            return Err(VulkanError::MissingExtension(String::from(
                "VK_KHR_acceleration_structure",
            )));
        };

        let geometries = geometries_decl
            .iter()
            .map(|g| g.ash_geometry())
            .collect::<Vec<ash::vk::AccelerationStructureGeometryKHR>>();
        let max_primitives_count = geometries_decl.iter().map(|_| 1u32).collect::<Vec<u32>>();

        // From vulkan specs: https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkGetAccelerationStructureBuildSizesKHR.html
        // The srcAccelerationStructure, dstAccelerationStructure, and mode members of pBuildInfo are ignored.
        // Any VkDeviceOrHostAddressKHR members of pBuildInfo are ignored by this command,
        // except that the hostAddress member of VkAccelerationStructureGeometryTrianglesDataKHR::transformData
        // will be examined to check if it is NULL.
        let geometry_info = ash::vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .geometries(geometries.as_slice())
            .flags(ash::vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
            .ty(ash::vk::AccelerationStructureTypeKHR::TOP_LEVEL);

        let mut build_sizes = ash::vk::AccelerationStructureBuildSizesInfoKHR::default();
        unsafe {
            as_ext.get_acceleration_structure_build_sizes(
                allowed_building_devices.ash_flags(),
                &geometry_info,
                max_primitives_count.as_slice(),
                &mut build_sizes,
            )
        };

        Ok(build_sizes.acceleration_structure_size)
    }

    /*
     * This function allows the user to estimate a minimum size for provided geometry info.
     *
     * If the operation is successful the tuple returned will be the TLAS (minimum) size
     */
    pub fn query_minimum_build_scratch_buffer_size(
        device: Arc<Device>,
        allowed_building_devices: AllowedBuildingDevice,
        geometries_decl: &[TopLevelBLASGroupDecl],
    ) -> VulkanResult<u64> {
        let Some(as_ext) = device.ash_ext_acceleration_structure_khr() else {
            return Err(VulkanError::MissingExtension(String::from(
                "VK_KHR_acceleration_structure",
            )));
        };

        let geometries = geometries_decl
            .iter()
            .map(|g| g.ash_geometry())
            .collect::<Vec<ash::vk::AccelerationStructureGeometryKHR>>();
        let max_primitives_count = geometries_decl.iter().map(|_| 1u32).collect::<Vec<u32>>();

        // See https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkGetAccelerationStructureBuildSizesKHR.html
        let geometry_info = ash::vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .geometries(geometries.as_slice())
            .flags(ash::vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
            .ty(ash::vk::AccelerationStructureTypeKHR::TOP_LEVEL);

        let mut build_sizes = ash::vk::AccelerationStructureBuildSizesInfoKHR::default();
        unsafe {
            as_ext.get_acceleration_structure_build_sizes(
                allowed_building_devices.ash_flags(),
                &geometry_info,
                max_primitives_count.as_slice(),
                &mut build_sizes,
            )
        };

        Ok(build_sizes.build_scratch_size)
    }

    pub(crate) fn ash_handle(&self) -> ash::vk::AccelerationStructureKHR {
        self.handle
    }

    pub(crate) fn device_build_scratch_buffer(&self) -> Arc<DeviceScratchBuffer> {
        self.device_build_scratch_buffer.clone()
    }

    pub fn blas_decl(&self) -> &TopLevelBLASGroupDecl {
        &self.blas_decl
    }

    pub fn buffer_device_addr(&self) -> u64 {
        self.buffer_device_addr
    }

    pub fn buffer_size(&self) -> u64 {
        self.buffer.size()
    }

    pub fn new(
        memory_pool: Arc<MemoryPool>,
        allowed_building_devices: AllowedBuildingDevice,
        blas_decl: TopLevelBLASGroupDecl,
        debug_name: Option<&str>,
    ) -> VulkanResult<Arc<Self>> {
        let device = memory_pool.get_parent_memory_heap().get_parent_device();

        let Some(as_ext) = device.ash_ext_acceleration_structure_khr() else {
            return Err(VulkanError::MissingExtension(String::from(
                "VK_KHR_acceleration_structure",
            )));
        };

        if !memory_pool.features().device_addressable() {
            return Err(VulkanError::Framework(FrameworkError::Unknown(Some(
                String::from("Missing feature on MemoryPool: device_addressable need to be set"),
            ))));
        }

        let tlas_mix_buffer_size = TopLevelAccelerationStructure::query_minimum_sizes(
            device.clone(),
            allowed_building_devices,
            &[blas_decl],
        )
        .unwrap();

        // TODO: review sharing mode and debug name
        let tlas_buffer_debug_name = debug_name.map(|name| format!("{name}_tlas_buffer"));
        let buffer = AllocatedBuffer::new(
            memory_pool.clone(),
            Buffer::new(
                device.clone(),
                ConcreteBufferDescriptor::new(
                    BufferUsage::Unmanaged(
                        (ash::vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
                            | ash::vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS)
                            .as_raw(),
                    ),
                    tlas_mix_buffer_size,
                ),
                None,
                match &tlas_buffer_debug_name {
                    Some(name) => Some(name.as_str()),
                    None => None,
                },
            )?,
        )?;

        let info = ash::vk::BufferDeviceAddressInfo::default().buffer(buffer.ash_handle());

        let buffer_device_addr = unsafe { device.ash_handle().get_buffer_device_address(&info) };

        // If deviceAddress is not zero, createFlags must include VK_ACCELERATION_STRUCTURE_CREATE_DEVICE_ADDRESS_CAPTURE_REPLAY_BIT_KHR
        let create_info = ash::vk::AccelerationStructureCreateInfoKHR::default()
            .buffer(buffer.ash_handle())
            .offset(0)
            .size(buffer.size())
            .ty(ash::vk::AccelerationStructureTypeKHR::TOP_LEVEL)
            .create_flags(ash::vk::AccelerationStructureCreateFlagsKHR::empty());
        //.device_address(buffer_device_addr)

        let handle = unsafe {
            as_ext.create_acceleration_structure(
                &create_info,
                device.get_parent_instance().get_alloc_callbacks(),
            )
        }
        .map_err(|err| {
            VulkanError::Vulkan(
                err.as_raw(),
                Some(format!("Error creating acceleration structure: {err}")),
            )
        })?;

        let info = ash::vk::AccelerationStructureDeviceAddressInfoKHR::default()
            .acceleration_structure(handle);
        let acceleration_structure_device_addr =
            unsafe { as_ext.get_acceleration_structure_device_address(&info) };

        let build_scratch_buffer_size = Self::query_minimum_build_scratch_buffer_size(
            device,
            allowed_building_devices,
            &[blas_decl],
        )?;
        let device_build_scratch_buffer =
            DeviceScratchBuffer::new(memory_pool, build_scratch_buffer_size)?;

        Ok(Arc::new(Self {
            blas_decl,
            handle,
            acceleration_structure_device_addr,
            buffer,
            buffer_device_addr,
            allowed_building_devices,
            device_build_scratch_buffer,
        }))
    }
}

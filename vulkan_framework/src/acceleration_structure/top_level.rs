use std::sync::Arc;

use ash::vk::{AccelerationStructureGeometryKHR, DeviceOrHostAddressConstKHR};

use crate::{
    acceleration_structure::{scratch_buffer::DeviceScratchBuffer, AllowedBuildingDevice},
    buffer::{AllocatedBuffer, Buffer, BufferTrait, BufferUsage, ConcreteBufferDescriptor},
    device::DeviceOwned,
    instance::InstanceOwned,
    memory_heap::MemoryHeapOwned,
    memory_pool::MemoryPool,
    prelude::{FrameworkError, VulkanError, VulkanResult},
    queue_family::QueueFamily,
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

pub struct TopLevelAccelerationStructureInstanceBuffer {
    buffer: Arc<AllocatedBuffer>,
    buffer_device_addr: u64,
}

impl TopLevelAccelerationStructureInstanceBuffer {
    pub fn new(
        memory_pool: Arc<MemoryPool>,
        usage: BufferUsage,
        max_instances: u32,
        sharing: Option<&[std::sync::Weak<QueueFamily>]>,
        debug_name: &Option<&str>,
    ) -> VulkanResult<Arc<Self>> {
        let device = memory_pool.get_parent_memory_heap().get_parent_device();

        // TODO: change index buffer sizes
        let instance_buffer_debug_name = debug_name.map(|name| format!("{name}_instance_buffer"));
        let instance_buffer = Buffer::new(
            device.clone(),
            ConcreteBufferDescriptor::new(
                BufferUsage::Unmanaged(
                    usage.ash_usage().as_raw() |
                    (ash::vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                        //| ash::vk::BufferUsageFlags::STORAGE_BUFFER
                        | ash::vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS)
                        .as_raw(),
                ),
                (core::mem::size_of::<ash::vk::AccelerationStructureInstanceKHR>() as u64)
                    * (max_instances as u64),
            ),
            sharing,
            match &instance_buffer_debug_name {
                Some(name) => Some(name.as_str()),
                None => None,
            },
        )?;

        let buffer = AllocatedBuffer::new(memory_pool.clone(), instance_buffer)?;

        let info = ash::vk::BufferDeviceAddressInfo::default().buffer(buffer.ash_handle());
        let buffer_device_addr = unsafe { device.ash_handle().get_buffer_device_address(&info) };

        Ok(Arc::new(Self {
            buffer,
            buffer_device_addr,
        }))
    }

    #[inline]
    pub fn buffer(&self) -> Arc<AllocatedBuffer> {
        self.buffer.clone()
    }

    #[inline]
    pub fn buffer_device_addr(&self) -> u64 {
        self.buffer_device_addr.to_owned()
    }
}

pub struct TopLevelAccelerationStructure {
    blas_decl: smallvec::SmallVec<[TopLevelBLASGroupDecl; 1]>,

    max_instances: u32,

    handle: ash::vk::AccelerationStructureKHR,

    buffer: Arc<dyn BufferTrait>,
    buffer_device_addr: u64,

    instance_buffer: Arc<TopLevelAccelerationStructureInstanceBuffer>,

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
    pub(crate) fn static_ash_geometry<'a>(
        blas_decl: &[&'a TopLevelBLASGroupDecl],
        instance_buffer: Arc<TopLevelAccelerationStructureInstanceBuffer>,
    ) -> smallvec::SmallVec<[ash::vk::AccelerationStructureGeometryKHR<'a>; 1]> {
        blas_decl
            .iter()
            .map(|g| {
                let mut data = g.ash_geometry();

                data.geometry.triangles.transform_data = DeviceOrHostAddressConstKHR {
                    device_address: instance_buffer.buffer_device_addr(),
                };

                data
            })
            .collect::<smallvec::SmallVec<_>>()
    }

    pub(crate) fn ash_geometry<'a>(
        &'a self,
    ) -> smallvec::SmallVec<[ash::vk::AccelerationStructureGeometryKHR<'a>; 1]> {
        self.blas_decl
            .iter()
            .map(|g| {
                let mut data = g.ash_geometry();

                data.geometry.instances.data = DeviceOrHostAddressConstKHR {
                    device_address: self.instance_buffer().buffer_device_addr(),
                };

                data
            })
            .collect::<smallvec::SmallVec<_>>()
    }

    /*
     * This function allows the user to estimate a minimum size for provided geometry info.
     *
     * If the operation is successful the tuple returned will be the TLAS (minimum) size
     */
    pub fn query_minimum_buffer_size(
        allowed_building_devices: AllowedBuildingDevice,
        blas_decl: &[&TopLevelBLASGroupDecl],
        max_primitives: u32,
        instance_buffer: Arc<TopLevelAccelerationStructureInstanceBuffer>,
    ) -> VulkanResult<u64> {
        let device = instance_buffer.buffer().get_parent_device();

        let Some(as_ext) = device.ash_ext_acceleration_structure_khr() else {
            return Err(VulkanError::MissingExtension(String::from(
                "VK_KHR_acceleration_structure",
            )));
        };

        let geometries = Self::static_ash_geometry(blas_decl, instance_buffer);
        let max_primitives_count: smallvec::SmallVec<[u32; 1]> =
            smallvec::smallvec![max_primitives];

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
        allowed_building_devices: AllowedBuildingDevice,
        blas_decl: &[&TopLevelBLASGroupDecl],
        max_primitives: u32,
        instance_buffer: Arc<TopLevelAccelerationStructureInstanceBuffer>,
    ) -> VulkanResult<u64> {
        let device = instance_buffer.buffer().get_parent_device();

        let Some(as_ext) = device.ash_ext_acceleration_structure_khr() else {
            return Err(VulkanError::MissingExtension(String::from(
                "VK_KHR_acceleration_structure",
            )));
        };

        let geometries = Self::static_ash_geometry(blas_decl, instance_buffer);
        let max_primitives_count: smallvec::SmallVec<[u32; 1]> =
            smallvec::smallvec![max_primitives];

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

    #[inline]
    pub(crate) fn ash_handle(&self) -> ash::vk::AccelerationStructureKHR {
        self.handle
    }

    #[inline]
    pub(crate) fn device_build_scratch_buffer(&self) -> Arc<DeviceScratchBuffer> {
        self.device_build_scratch_buffer.clone()
    }

    #[inline]
    pub fn max_instances(&self) -> u32 {
        self.max_instances.to_owned()
    }

    #[inline]
    pub fn blas_decl(&self) -> &[TopLevelBLASGroupDecl] {
        self.blas_decl.as_slice()
    }

    #[inline]
    pub fn allowed_building_devices(&self) -> AllowedBuildingDevice {
        self.allowed_building_devices
    }

    #[inline]
    pub fn buffer_device_addr(&self) -> u64 {
        self.buffer_device_addr
    }

    #[inline]
    pub fn buffer_size(&self) -> u64 {
        self.buffer.size()
    }

    #[inline]
    pub fn instance_buffer(&self) -> Arc<TopLevelAccelerationStructureInstanceBuffer> {
        self.instance_buffer.clone()
    }

    pub fn new(
        memory_pool: Arc<MemoryPool>,
        allowed_building_devices: AllowedBuildingDevice,
        blas_decl: TopLevelBLASGroupDecl,
        max_instances: u32,
        instance_buffer_usage: BufferUsage,
        sharing: Option<&[std::sync::Weak<QueueFamily>]>,
        debug_name: Option<&str>,
    ) -> VulkanResult<Arc<Self>> {
        let device = memory_pool.get_parent_memory_heap().get_parent_device();

        let Some(as_ext) = device.ash_ext_acceleration_structure_khr() else {
            return Err(VulkanError::MissingExtension(String::from(
                "VK_KHR_acceleration_structure",
            )));
        };

        if !memory_pool.features().device_addressable() {
            return Err(VulkanError::Framework(
                FrameworkError::MemoryPoolNotAddressable,
            ));
        }

        let instance_buffer = TopLevelAccelerationStructureInstanceBuffer::new(
            memory_pool.clone(),
            instance_buffer_usage,
            max_instances,
            sharing,
            &debug_name,
        )?;

        let tlas_mix_buffer_size = TopLevelAccelerationStructure::query_minimum_buffer_size(
            allowed_building_devices,
            &[&blas_decl],
            max_instances,
            instance_buffer.clone(),
        )?;

        let build_scratch_buffer_size = Self::query_minimum_build_scratch_buffer_size(
            allowed_building_devices,
            &[&blas_decl],
            max_instances,
            instance_buffer.clone(),
        )?;

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

        let blas_decl = smallvec::smallvec![blas_decl];

        let device_build_scratch_buffer =
            DeviceScratchBuffer::new(memory_pool, build_scratch_buffer_size)?;

        Ok(Arc::new(Self {
            blas_decl,
            max_instances,
            handle,
            buffer,
            buffer_device_addr,
            instance_buffer,
            allowed_building_devices,
            device_build_scratch_buffer,
        }))
    }

    pub(crate) fn ash_build_info(
        &self,
        primitive_offset: u32,
        primitive_count: u32,
    ) -> VulkanResult<(
        smallvec::SmallVec<[AccelerationStructureGeometryKHR<'_>; 1]>,
        smallvec::SmallVec<
            [smallvec::SmallVec<[ash::vk::AccelerationStructureBuildRangeInfoKHR; 1]>; 1],
        >,
    )> {
        let tlas_max_instances = self.max_instances() as u64;
        let selected_instances_max_index =
            (primitive_offset.to_owned() as u64) + (primitive_count.to_owned() as u64);
        assert!(tlas_max_instances >= selected_instances_max_index);

        let geometries = self.ash_geometry();

        let range_infos: smallvec::SmallVec<
            [smallvec::SmallVec<[ash::vk::AccelerationStructureBuildRangeInfoKHR; 1]>; 1],
        > = smallvec::smallvec![smallvec::smallvec![
            ash::vk::AccelerationStructureBuildRangeInfoKHR::default()
                .primitive_offset(primitive_offset.to_owned())
                .primitive_count(primitive_count.to_owned())
        ]];

        // TODO: ash::vk::AccelerationStructureBuildGeometryInfoKHR

        Ok((geometries, range_infos))
    }
}

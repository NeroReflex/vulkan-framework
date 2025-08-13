use std::sync::Arc;

use ash::vk::{AccelerationStructureGeometryKHR, DeviceOrHostAddressConstKHR};

use crate::{
    acceleration_structure::{scratch_buffer::DeviceScratchBuffer, AllowedBuildingDevice},
    buffer::{AllocatedBuffer, Buffer, BufferTrait, BufferUsage, ConcreteBufferDescriptor},
    device::DeviceOwned,
    instance::InstanceOwned,
    memory_heap::{MemoryHostVisibility, MemoryType},
    memory_management::{MemoryManagementTags, MemoryManagerTrait},
    memory_pool::{MemoryPoolBacked, MemoryPoolFeatures},
    prelude::{VulkanError, VulkanResult},
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

    blas_decl: TopLevelBLASGroupDecl,
    max_instances: u32,
}

impl TopLevelAccelerationStructureInstanceBuffer {
    #[inline(always)]
    fn size_from_definition(blas_decl: &TopLevelBLASGroupDecl, max_instances: u32) -> u64 {
        // TODO: if array_of_pointers act accordingly
        match blas_decl.array_of_pointers() {
            true => todo!(),
            false => {
                (core::mem::size_of::<ash::vk::AccelerationStructureInstanceKHR>() as u64)
                    * (max_instances as u64)
            }
        }
    }

    #[inline(always)]
    pub fn template(
        blas_decl: &TopLevelBLASGroupDecl,
        max_instances: u32,
        usage: BufferUsage,
    ) -> ConcreteBufferDescriptor {
        ConcreteBufferDescriptor::new(
            BufferUsage::from(
                <BufferUsage as Into<ash::vk::BufferUsageFlags>>::into(usage).as_raw()
                    | (ash::vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                    //| ash::vk::BufferUsageFlags::STORAGE_BUFFER
                    | ash::vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS)
                        .as_raw(),
            ),
            Self::size_from_definition(blas_decl, max_instances),
        )
    }

    pub fn new(
        blas_decl: TopLevelBLASGroupDecl,
        max_instances: u32,
        buffer: Arc<AllocatedBuffer>,
    ) -> VulkanResult<Self> {
        // Check if the transform buffer is suitable to be used as such
        {
            let usage = buffer.buffer().descriptor().ash_usage();
            if !buffer
                .get_backing_memory_pool()
                .features()
                .device_addressable()
            {
                panic!("Buffer used as an acceleration structure instance buffer was created from a memory pool that is not device addressable");
            } else if !usage.contains(
                ash::vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
            ) {
                panic!("Buffer used as an acceleration structure instance buffer was not created with ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR");
            } else if !usage.contains(ash::vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS) {
                panic!("Buffer used as an acceleration structure instance buffer was not created with SHADER_DEVICE_ADDRESS");
            } else if buffer.size() < Self::size_from_definition(&blas_decl, max_instances) {
                panic!("Buffer used as an acceleration structure instance buffer is too small");
            }
        }

        let info = ash::vk::BufferDeviceAddressInfo::default().buffer(buffer.ash_handle());
        let buffer_device_addr = unsafe {
            buffer
                .get_parent_device()
                .ash_handle()
                .get_buffer_device_address(&info)
        };

        Ok(Self {
            buffer,
            buffer_device_addr,

            blas_decl,
            max_instances,
        })
    }

    #[inline(always)]
    pub fn max_instances(&self) -> u32 {
        self.max_instances.to_owned()
    }

    #[inline(always)]
    pub fn blas_decl(&self) -> &TopLevelBLASGroupDecl {
        &self.blas_decl
    }

    #[inline(always)]
    pub fn buffer(&self) -> Arc<AllocatedBuffer> {
        self.buffer.clone()
    }

    #[inline(always)]
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

    instance_buffer: TopLevelAccelerationStructureInstanceBuffer,

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
        instance_buffer: &TopLevelAccelerationStructureInstanceBuffer,
    ) -> smallvec::SmallVec<[ash::vk::AccelerationStructureGeometryKHR<'a>; 1]> {
        blas_decl
            .iter()
            .map(|g| {
                let mut data = g.ash_geometry();

                assert_eq!(data.geometry_type, ash::vk::GeometryTypeKHR::INSTANCES);
                assert_eq!(
                    unsafe { data.geometry.instances.array_of_pointers },
                    g.array_of_pointers() as u32
                );

                data.geometry.instances.data = DeviceOrHostAddressConstKHR {
                    device_address: instance_buffer.buffer_device_addr(),
                };

                data
            })
            .collect::<smallvec::SmallVec<_>>()
    }

    pub(crate) fn ash_geometry<'a>(
        &'a self,
    ) -> smallvec::SmallVec<[ash::vk::AccelerationStructureGeometryKHR<'a>; 1]> {
        Self::static_ash_geometry(
            self.blas_decl
                .iter()
                .collect::<smallvec::SmallVec<[&'a TopLevelBLASGroupDecl; 1]>>()
                .as_slice(),
            self.instance_buffer(),
        )
    }

    pub(crate) fn geometry_info<'a>(
        geometries: &'a [AccelerationStructureGeometryKHR],
    ) -> ash::vk::AccelerationStructureBuildGeometryInfoKHR<'a> {
        ash::vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .geometries(geometries)
            .flags(Self::static_build_flags())
            .ty(ash::vk::AccelerationStructureTypeKHR::TOP_LEVEL)
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
        instance_buffer: &TopLevelAccelerationStructureInstanceBuffer,
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
        let geometry_info = Self::geometry_info(geometries.as_slice());

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
        instance_buffer: &TopLevelAccelerationStructureInstanceBuffer,
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
        let geometry_info = Self::geometry_info(geometries.as_slice());

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

    #[inline(always)]
    pub(crate) fn ash_handle(&self) -> ash::vk::AccelerationStructureKHR {
        self.handle
    }

    #[inline(always)]
    pub(crate) fn device_build_scratch_buffer(&self) -> Arc<DeviceScratchBuffer> {
        self.device_build_scratch_buffer.clone()
    }

    #[inline(always)]
    pub fn max_instances(&self) -> u32 {
        self.max_instances.to_owned()
    }

    #[inline(always)]
    pub fn blas_decl(&self) -> &[TopLevelBLASGroupDecl] {
        self.blas_decl.as_slice()
    }

    #[inline(always)]
    pub fn allowed_building_devices(&self) -> AllowedBuildingDevice {
        self.allowed_building_devices
    }

    #[inline(always)]
    pub fn buffer_device_addr(&self) -> u64 {
        self.buffer_device_addr
    }

    #[inline(always)]
    pub fn buffer_size(&self) -> u64 {
        self.buffer.size()
    }

    #[inline(always)]
    pub fn instance_buffer(&self) -> &TopLevelAccelerationStructureInstanceBuffer {
        &self.instance_buffer
    }

    fn create_tlas_buffer(
        memory_manager: &mut dyn MemoryManagerTrait,
        buffer_size: u64,
        allocation_tags: MemoryManagementTags,
        sharing: Option<&[std::sync::Weak<QueueFamily>]>,
        debug_name: &Option<&str>,
    ) -> VulkanResult<(Arc<AllocatedBuffer>, u64)> {
        let device = memory_manager.get_parent_device();

        // TODO: change index buffer sizes
        let tlas_buffer_debug_name = debug_name.map(|name| format!("{name}_tlas_buffer"));
        let tlas_buffer = Buffer::new(
            device.clone(),
            ConcreteBufferDescriptor::new(
                BufferUsage::from(
                    (ash::vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
                        | ash::vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS)
                        .as_raw(),
                ),
                buffer_size,
            ),
            sharing,
            match &tlas_buffer_debug_name {
                Some(name) => Some(name.as_str()),
                None => None,
            },
        )?;

        let buffer = memory_manager.allocate_resources(
            &MemoryType::device_local_and_host_visible(),
            &MemoryPoolFeatures::new(true),
            vec![tlas_buffer.into()],
            allocation_tags,
        )?[0]
            .buffer();

        let info = ash::vk::BufferDeviceAddressInfo::default().buffer(buffer.ash_handle());
        let buffer_device_addr = unsafe { device.ash_handle().get_buffer_device_address(&info) };

        Ok((buffer, buffer_device_addr))
    }

    #[inline(always)]
    pub(crate) fn static_build_flags() -> ash::vk::BuildAccelerationStructureFlagsKHR {
        ash::vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE
    }

    #[inline(always)]
    pub(crate) fn build_flags(&self) -> ash::vk::BuildAccelerationStructureFlagsKHR {
        Self::static_build_flags()
    }

    pub fn new(
        memory_manager: &mut dyn MemoryManagerTrait,
        allowed_building_devices: AllowedBuildingDevice,
        instance_buffer: TopLevelAccelerationStructureInstanceBuffer,
        allocation_tags: MemoryManagementTags,
        sharing: Option<&[std::sync::Weak<QueueFamily>]>,
        debug_name: Option<&str>,
    ) -> VulkanResult<Arc<Self>> {
        let device = memory_manager.get_parent_device();

        let Some(as_ext) = device.ash_ext_acceleration_structure_khr() else {
            return Err(VulkanError::MissingExtension(String::from(
                "VK_KHR_acceleration_structure",
            )));
        };

        let blas_decl = instance_buffer.blas_decl();
        let max_instances = instance_buffer.max_instances();

        let tlas_min_buffer_size = TopLevelAccelerationStructure::query_minimum_buffer_size(
            allowed_building_devices,
            &[blas_decl],
            max_instances,
            &instance_buffer,
        )?;

        let build_scratch_buffer_size = Self::query_minimum_build_scratch_buffer_size(
            allowed_building_devices,
            &[blas_decl],
            max_instances,
            &instance_buffer,
        )?;

        let (buffer, buffer_device_addr) = Self::create_tlas_buffer(
            memory_manager,
            tlas_min_buffer_size,
            allocation_tags.clone(),
            sharing,
            &debug_name,
        )?;

        // If deviceAddress is not zero, createFlags must include VK_ACCELERATION_STRUCTURE_CREATE_DEVICE_ADDRESS_CAPTURE_REPLAY_BIT_KHR
        let create_info = ash::vk::AccelerationStructureCreateInfoKHR::default()
            .buffer(buffer.ash_handle())
            .offset(0)
            .size(buffer.size())
            .ty(ash::vk::AccelerationStructureTypeKHR::TOP_LEVEL)
            .create_flags(ash::vk::AccelerationStructureCreateFlagsKHR::empty());

        let handle = unsafe {
            as_ext.create_acceleration_structure(
                &create_info,
                device.get_parent_instance().get_alloc_callbacks(),
            )
        }?;

        let blas_decl = smallvec::smallvec![*blas_decl];

        let device_build_scratch_buffer =
            DeviceScratchBuffer::new(memory_manager, build_scratch_buffer_size, allocation_tags)?;

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
        let selected_instances_max_index =
            (primitive_offset.to_owned() as u64) + (primitive_count.to_owned() as u64);
        assert!((self.max_instances() as u64) >= selected_instances_max_index);

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

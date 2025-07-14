use std::sync::Arc;

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

pub struct DeviceScratchBuffer {
    buffer: Arc<dyn BufferTrait>,
    buffer_device_addr: u64,
}

impl DeviceOwned for DeviceScratchBuffer {
    fn get_parent_device(&self) -> Arc<Device> {
        self.buffer.get_parent_device()
    }
}

impl BufferTrait for DeviceScratchBuffer {
    #[inline]
    fn size(&self) -> u64 {
        self.buffer.size()
    }

    #[inline]
    fn native_handle(&self) -> u64 {
        self.buffer.native_handle()
    }
}

impl DeviceScratchBuffer {
    #[inline]
    pub(crate) fn addr(&self) -> ash::vk::DeviceOrHostAddressKHR {
        ash::vk::DeviceOrHostAddressKHR {
            device_address: self.buffer_device_addr,
        }
    }

    pub fn new(memory_pool: Arc<MemoryPool>, size: u64) -> VulkanResult<Arc<Self>> {
        let buffer = AllocatedBuffer::new(
            memory_pool.clone(),
            Buffer::new(
                memory_pool
                    .get_parent_memory_heap()
                    .get_parent_device()
                    .clone(),
                ConcreteBufferDescriptor::new(
                    BufferUsage::Unmanaged(
                        (ash::vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                            | ash::vk::BufferUsageFlags::STORAGE_BUFFER)
                            .as_raw(),
                    ),
                    size,
                ),
                None,
                None,
            )?,
        )?;
        let info = ash::vk::BufferDeviceAddressInfo::default().buffer(buffer.ash_handle());

        let buffer_device_addr = unsafe {
            memory_pool
                .get_parent_memory_heap()
                .get_parent_device()
                .ash_handle()
                .get_buffer_device_address(&info)
        };

        Ok(Arc::new(Self {
            buffer,
            buffer_device_addr,
        }))
    }
}

pub struct HostScratchBuffer {
    buffer: Mutex<Vec<u8>>,
}

impl HostScratchBuffer {
    pub(crate) fn address(&self) -> ash::vk::DeviceOrHostAddressKHR {
        #[cfg(feature = "better_mutex")]
        {
            let mut lck = self.buffer.lock();

            ash::vk::DeviceOrHostAddressKHR {
                host_address: lck.as_mut_slice().as_mut_ptr() as *mut std::ffi::c_void,
            }
        }

        #[cfg(not(feature = "better_mutex"))]
        {
            match self.buffer.lock() {
                Ok(mut lck) => ash::vk::DeviceOrHostAddressKHR {
                    host_address: lck.as_mut_slice().as_mut_ptr() as *mut std::ffi::c_void,
                },
                Err(_err) => {
                    todo!()
                }
            }
        }
    }

    pub fn new(size: u64) -> Arc<Self> {
        #[cfg(feature = "better_mutex")]
        {
            Arc::new(Self {
                buffer: const_mutex(Vec::<u8>::with_capacity(size as usize)),
            })
        }

        #[cfg(not(feature = "better_mutex"))]
        {
            Arc::new(Self {
                buffer: Mutex::new(Vec::<u8>::with_capacity(size as usize)),
            })
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
    pub(crate) fn ash_index_type(&self) -> ash::vk::IndexType {
        match self {
            VertexIndexing::None => ash::vk::IndexType::NONE_KHR,
            VertexIndexing::UInt16 => ash::vk::IndexType::UINT16,
            VertexIndexing::UInt32 => ash::vk::IndexType::UINT32,
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct BottomLevelTrianglesGroupDecl {
    vertex_indexing: VertexIndexing,
    max_triangles: u32,
    vertex_stride: u64,
    vertex_format: AttributeType,
}

impl BottomLevelTrianglesGroupDecl {
    pub fn new(
        vertex_indexing: VertexIndexing,
        max_triangles: u32,
        vertex_stride: u64,
        vertex_format: AttributeType,
    ) -> Self {
        Self {
            vertex_indexing,
            max_triangles,
            vertex_stride,
            vertex_format,
        }
    }

    pub fn vertex_indexing(&self) -> VertexIndexing {
        self.vertex_indexing
    }

    pub fn max_triangles(&self) -> u32 {
        self.max_triangles
    }

    pub fn max_vertices(&self) -> u32 {
        self.max_triangles() * 3u32
    }

    pub fn vertex_stride(&self) -> u64 {
        self.vertex_stride
    }

    pub fn vertex_format(&self) -> AttributeType {
        self.vertex_format
    }

    pub(crate) fn ash_geometry(&self) -> ash::vk::AccelerationStructureGeometryKHR {
        ash::vk::AccelerationStructureGeometryKHR::default()
            // TODO: .flags()
            .geometry_type(ash::vk::GeometryTypeKHR::TRIANGLES)
            .geometry(ash::vk::AccelerationStructureGeometryDataKHR {
                triangles: ash::vk::AccelerationStructureGeometryTrianglesDataKHR::default()
                    .index_type(self.vertex_indexing().ash_index_type())
                    .max_vertex(self.max_vertices())
                    .vertex_format(self.vertex_format.ash_format())
                    .vertex_stride(self.vertex_stride),
            })
    }
}

pub struct BottomLevelTrianglesGroupData {
    decl: BottomLevelTrianglesGroupDecl,

    index_buffer: Option<Arc<AllocatedBuffer>>,
    vertex_buffer: Arc<AllocatedBuffer>,
    transform_buffer: Arc<AllocatedBuffer>,

    primitive_offset: u32,
    primitive_count: u32,
    first_vertex: u32,
    transform_offset: u32,
}

impl BottomLevelTrianglesGroupData {
    pub fn new(
        decl: BottomLevelTrianglesGroupDecl,
        index_buffer: Option<Arc<AllocatedBuffer>>,
        vertex_buffer: Arc<AllocatedBuffer>,
        transform_buffer: Arc<AllocatedBuffer>,
        primitive_offset: u32,
        primitive_count: u32,
        first_vertex: u32,
        transform_offset: u32,
    ) -> Self {
        Self {
            decl,
            index_buffer,
            vertex_buffer,
            transform_buffer,
            primitive_offset,
            primitive_count,
            first_vertex,
            transform_offset,
        }
    }

    pub fn decl(&self) -> BottomLevelTrianglesGroupDecl {
        self.decl
    }

    pub fn index_buffer(&self) -> Option<Arc<AllocatedBuffer>> {
        self.index_buffer.clone()
    }

    pub fn vertex_buffer(&self) -> Arc<AllocatedBuffer> {
        self.vertex_buffer.clone()
    }

    pub fn transform_buffer(&self) -> Arc<AllocatedBuffer> {
        self.transform_buffer.clone()
    }

    pub fn primitive_offset(&self) -> u32 {
        self.primitive_offset
    }

    pub fn primitive_count(&self) -> u32 {
        self.primitive_count
    }

    pub fn first_vertex(&self) -> u32 {
        self.first_vertex
    }

    pub fn transform_offset(&self) -> u32 {
        self.transform_offset
    }
}

pub struct BottomLevelAccelerationStructure {
    handle: ash::vk::AccelerationStructureKHR,
    buffer: Arc<dyn BufferTrait>,
    device_addr: u64,
    //builder: Arc<BottomLevelAccelerationStructureBuilder>,
}

impl Drop for BottomLevelAccelerationStructure {
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

impl BottomLevelAccelerationStructure {
    /*
     * This function allows the user to estimate a minimum size for provided geometry info.
     *
     * If the operation is successful the tuple returned will be:
     *   - BLAS (minimum) size
     *   - BLAS build (minimum) scratch buffer size
     *   - BLAS update (minimum) scratch buffer size
     */
    pub fn query_minimum_sizes(
        device: Arc<Device>,
        allowed_building_devices: AllowedBuildingDevice,
        geometries_decl: &[BottomLevelTrianglesGroupDecl],
    ) -> VulkanResult<(u64, u64, u64)> {
        let geometries = geometries_decl
            .iter()
            .map(|g| g.ash_geometry())
            .collect::<Vec<ash::vk::AccelerationStructureGeometryKHR>>();
        let max_primitives_count = geometries_decl
            .iter()
            .map(|g| g.max_triangles())
            .collect::<Vec<u32>>();

        // From vulkan specs: https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkGetAccelerationStructureBuildSizesKHR.html
        // The srcAccelerationStructure, dstAccelerationStructure, and mode members of pBuildInfo are ignored.
        // Any VkDeviceOrHostAddressKHR members of pBuildInfo are ignored by this command,
        // except that the hostAddress member of VkAccelerationStructureGeometryTrianglesDataKHR::transformData
        // will be examined to check if it is NULL.
        let geometry_info = ash::vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .geometries(geometries.as_slice())
            .flags(ash::vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
            .ty(ash::vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL);

        match device.ash_ext_acceleration_structure_khr() {
            Some(as_ext) => {
                let mut build_sizes = ash::vk::AccelerationStructureBuildSizesInfoKHR::default();

                unsafe {
                    as_ext.get_acceleration_structure_build_sizes(
                        allowed_building_devices.ash_flags(),
                        &geometry_info,
                        max_primitives_count.as_slice(),
                        &mut build_sizes,
                    )
                };

                Ok((
                    build_sizes.acceleration_structure_size,
                    build_sizes.build_scratch_size,
                    build_sizes.update_scratch_size,
                ))
            }
            None => Err(VulkanError::MissingExtension(String::from(
                "VK_KHR_acceleration_structure",
            ))),
        }
    }

    pub(crate) fn ash_handle(&self) -> ash::vk::AccelerationStructureKHR {
        self.handle
    }

    pub fn device_addr(&self) -> u64 {
        self.device_addr
    }

    pub fn buffer_size(&self) -> u64 {
        self.buffer.size()
    }

    pub fn new(
        memory_pool: Arc<MemoryPool>,
        //builder: Arc<BottomLevelAccelerationStructureBuilder>,
        buffer_size: u64,
    ) -> VulkanResult<Arc<Self>> {
        let device = memory_pool.get_parent_memory_heap().get_parent_device();

        if !memory_pool.features().device_addressable() {
            return Err(VulkanError::Framework(FrameworkError::Unknown(Some(
                String::from("Missing feature on MemoryPool: device_addressable need to be set"),
            ))));
        }

        // TODO: review sharing mode and debug name
        let buffer = AllocatedBuffer::new(
            memory_pool,
            Buffer::new(
                device.clone(),
                ConcreteBufferDescriptor::new(
                    BufferUsage::Unmanaged(
                        (ash::vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
                            | ash::vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS | ash::vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR)
                            .as_raw(),
                    ),
                    buffer_size,
                ),
                None,
                None,
            )?
        )?;

        let info = ash::vk::BufferDeviceAddressInfo::default().buffer(buffer.ash_handle());

        let _buffer_device_addr = unsafe { device.ash_handle().get_buffer_device_address(&info) };

        /*let scratch_data = ash::vk::DeviceOrHostAddressKHR {

        };*/

        match device.ash_ext_acceleration_structure_khr() {
            Some(as_ext) => {
                // If deviceAddress is not zero, createFlags must include VK_ACCELERATION_STRUCTURE_CREATE_DEVICE_ADDRESS_CAPTURE_REPLAY_BIT_KHR
                let create_info = ash::vk::AccelerationStructureCreateInfoKHR::default()
                    .buffer(buffer.ash_handle())
                    .offset(0)
                    .size(buffer.size())
                    .ty(ash::vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
                    .create_flags(ash::vk::AccelerationStructureCreateFlagsKHR::empty());
                //.device_address(buffer_device_addr)

                match unsafe {
                    as_ext.create_acceleration_structure(
                        &create_info,
                        device.get_parent_instance().get_alloc_callbacks(),
                    )
                } {
                    Ok(handle) => {
                        let info = ash::vk::AccelerationStructureDeviceAddressInfoKHR::default()
                            .acceleration_structure(handle);
                        Ok(Arc::new(Self {
                            handle,
                            buffer,
                            device_addr: unsafe {
                                as_ext.get_acceleration_structure_device_address(&info)
                            },
                        }))
                    }
                    Err(err) => Err(VulkanError::Vulkan(
                        err.as_raw(),
                        Some(format!("Error creating acceleration structure: {}", err)),
                    )),
                }
            }
            None => Err(VulkanError::MissingExtension(String::from(
                "VK_KHR_acceleration_structure",
            ))),
        }
    }
}

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

pub struct TopLevelBLASGroupData {
    decl: TopLevelBLASGroupDecl,

    instances_buffer: Arc<AllocatedBuffer>,

    primitive_offset: u32,
    primitive_count: u32,
    first_vertex: u32,
    transform_offset: u32,
}

impl TopLevelBLASGroupData {
    pub fn new(
        decl: TopLevelBLASGroupDecl,
        instances_buffer: Arc<AllocatedBuffer>,
        primitive_offset: u32,
        primitive_count: u32,
        first_vertex: u32,
        transform_offset: u32,
    ) -> Self {
        Self {
            decl,
            instances_buffer,
            primitive_offset,
            primitive_count,
            first_vertex,
            transform_offset,
        }
    }

    pub fn decl(&self) -> TopLevelBLASGroupDecl {
        self.decl
    }

    pub fn primitive_offset(&self) -> u32 {
        self.primitive_offset
    }

    pub fn primitive_count(&self) -> u32 {
        self.primitive_count
    }

    pub fn first_vertex(&self) -> u32 {
        self.first_vertex
    }

    pub fn transform_offset(&self) -> u32 {
        self.transform_offset
    }

    pub fn instances_buffer(&self) -> Arc<AllocatedBuffer> {
        self.instances_buffer.clone()
    }
}

pub struct TopLevelAccelerationStructure {
    handle: ash::vk::AccelerationStructureKHR,
    buffer: Arc<dyn BufferTrait>,
    buffer_device_addr: u64,
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
     * If the operation is successful the tuple returned will be:
     *   - BLAS (minimum) size
     *   - BLAS build (minimum) scratch buffer size
     *   - BLAS update (minimum) scratch buffer size
     */
    pub fn query_minimum_sizes(
        device: Arc<Device>,
        allowed_building_devices: AllowedBuildingDevice,
        geometries_decl: &[TopLevelBLASGroupDecl],
    ) -> VulkanResult<(u64, u64, u64)> {
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

        match device.ash_ext_acceleration_structure_khr() {
            Some(as_ext) => {
                let mut build_sizes = ash::vk::AccelerationStructureBuildSizesInfoKHR::default();

                unsafe {
                    as_ext.get_acceleration_structure_build_sizes(
                        allowed_building_devices.ash_flags(),
                        &geometry_info,
                        max_primitives_count.as_slice(),
                        &mut build_sizes,
                    )
                };

                Ok((
                    build_sizes.acceleration_structure_size,
                    build_sizes.build_scratch_size,
                    build_sizes.update_scratch_size,
                ))
            }
            None => Err(VulkanError::MissingExtension(String::from(
                "VK_KHR_acceleration_structure",
            ))),
        }
    }

    pub(crate) fn ash_handle(&self) -> ash::vk::AccelerationStructureKHR {
        self.handle
    }

    pub fn buffer_device_addr(&self) -> u64 {
        self.buffer_device_addr
    }

    pub fn buffer_size(&self) -> u64 {
        self.buffer.size()
    }

    pub fn new(
        memory_pool: Arc<MemoryPool>,
        //builder: Arc<BottomLevelAccelerationStructureBuilder>,
        buffer_size: u64,
    ) -> VulkanResult<Arc<Self>> {
        let device = memory_pool.get_parent_memory_heap().get_parent_device();

        if !memory_pool.features().device_addressable() {
            return Err(VulkanError::Framework(FrameworkError::Unknown(Some(
                String::from("Missing feature on MemoryPool: device_addressable need to be set"),
            ))));
        }

        // TODO: review sharing mode and debug name
        let buffer = AllocatedBuffer::new(
            memory_pool,
            Buffer::new(
                device.clone(),
                ConcreteBufferDescriptor::new(
                    BufferUsage::Unmanaged(
                        (ash::vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
                            | ash::vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS)
                            .as_raw(),
                    ),
                    buffer_size,
                ),
                None,
                None,
            )?,
        )?;

        let info = ash::vk::BufferDeviceAddressInfo::default().buffer(buffer.ash_handle());

        let buffer_device_addr = unsafe { device.ash_handle().get_buffer_device_address(&info) };

        /*let scratch_data = ash::vk::DeviceOrHostAddressKHR {

        };*/

        match device.ash_ext_acceleration_structure_khr() {
            Some(as_ext) => {
                // If deviceAddress is not zero, createFlags must include VK_ACCELERATION_STRUCTURE_CREATE_DEVICE_ADDRESS_CAPTURE_REPLAY_BIT_KHR
                let create_info = ash::vk::AccelerationStructureCreateInfoKHR::default()
                    .buffer(buffer.ash_handle())
                    .offset(0)
                    .size(buffer.size())
                    .ty(ash::vk::AccelerationStructureTypeKHR::TOP_LEVEL)
                    .create_flags(ash::vk::AccelerationStructureCreateFlagsKHR::empty());
                //.device_address(buffer_device_addr)

                match unsafe {
                    as_ext.create_acceleration_structure(
                        &create_info,
                        device.get_parent_instance().get_alloc_callbacks(),
                    )
                } {
                    Ok(handle) => Ok(Arc::new(Self {
                        handle,
                        buffer,
                        buffer_device_addr,
                    })),
                    Err(err) => Err(VulkanError::Vulkan(
                        err.as_raw(),
                        Some(format!("Error creating acceleration structure: {}", err)),
                    )),
                }
            }
            None => Err(VulkanError::MissingExtension(String::from(
                "VK_KHR_acceleration_structure",
            ))),
        }
    }
}

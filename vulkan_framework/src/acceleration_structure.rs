use std::sync::{Arc, Mutex};

use crate::{
    buffer::{Buffer, BufferTrait, BufferUsage, ConcreteBufferDescriptor},
    device::{Device, DeviceOwned},
    graphics_pipeline::AttributeType,
    instance::InstanceOwned,
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

pub struct TopLevelAccelerationStructureBuilder {
    device: Arc<Device>,

    geometries: smallvec::SmallVec<[ash::vk::AccelerationStructureGeometryKHR; 32]>,
    geometry_info: ash::vk::AccelerationStructureBuildGeometryInfoKHR,

    allowed_building_devices: AllowedBuildingDevice,
    acceleration_structure_size: u64,
    build_scratch_buffer_size: u64,
    update_scratch_buffer_size: u64,
}

impl TopLevelAccelerationStructureBuilder {
    
    pub fn acceleration_structure_size(&self) -> u64 {
        self.acceleration_structure_size
    }

    pub fn build_scratch_buffer_size(&self) -> u64 {
        self.build_scratch_buffer_size
    }

    pub fn update_scratch_buffer_size(&self) -> u64 {
        self.update_scratch_buffer_size
    }

    pub(crate) fn ash_structures(
        &self,
    ) -> (
        smallvec::SmallVec<[ash::vk::AccelerationStructureGeometryKHR; 32]>,
        ash::vk::AccelerationStructureBuildGeometryInfoKHR,
    ) {
        let geometries = self.geometries.clone();
        let geometry_info = ash::vk::AccelerationStructureBuildGeometryInfoKHR::builder()
            .geometries(geometries.as_slice())
            .flags(self.geometry_info.flags)
            .ty(self.geometry_info.ty)
            .build();

        (geometries, geometry_info)
    }

    pub fn allowed_building_devices(&self) -> AllowedBuildingDevice {
        self.allowed_building_devices
    }

    pub fn new(
        device: Arc<Device>,
        max_blas: u32,
        
        
        vertex_stride: u64,
        vertex_format: AttributeType,
        allowed_building_devices: AllowedBuildingDevice,
    ) -> VulkanResult<Arc<Self>> {
        let geometries = (0..max_blas).into_iter().map(|idx| {
            ash::vk::AccelerationStructureGeometryKHR::builder()
                .geometry_type(ash::vk::GeometryTypeKHR::INSTANCES)
                .flags(ash::vk::GeometryFlagsKHR::OPAQUE)
                .geometry(ash::vk::AccelerationStructureGeometryDataKHR {
                    instances: ash::vk::AccelerationStructureGeometryInstancesDataKHR::builder()
                        .array_of_pointers(false)
                        
                        .build()
                })
                .build()
        }).collect::<smallvec::SmallVec<[ash::vk::AccelerationStructureGeometryKHR; 32]>>();


        // From vulkan specs: https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkGetAccelerationStructureBuildSizesKHR.html
        // The srcAccelerationStructure, dstAccelerationStructure, and mode members of pBuildInfo are ignored.
        // Any VkDeviceOrHostAddressKHR members of pBuildInfo are ignored by this command,
        // except that the hostAddress member of VkAccelerationStructureGeometryTrianglesDataKHR::transformData
        // will be examined to check if it is NULL.
        let geometry_info = ash::vk::AccelerationStructureBuildGeometryInfoKHR::builder()
            .geometries(geometries.as_slice())
            .flags(ash::vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
            .ty(ash::vk::AccelerationStructureTypeKHR::TOP_LEVEL)
            .build();

        match device.ash_ext_acceleration_structure_khr() {
            Some(as_ext) => {
                let build_sizes = unsafe {
                    as_ext.get_acceleration_structure_build_sizes(
                        allowed_building_devices.ash_flags(),
                        &geometry_info,
                        &[max_blas],
                    )
                };

                //assert_eq!(build_sizes.len(), 1);

                /*let create_info = ash::vk::AccelerationStructureCreateInfoKHR::builder()
                .build();*/

                Ok(Arc::new(Self {
                    device,
                    geometries,
                    geometry_info,
                    
                    allowed_building_devices,
                    acceleration_structure_size: build_sizes.acceleration_structure_size,
                    build_scratch_buffer_size: build_sizes.build_scratch_size,
                    update_scratch_buffer_size: build_sizes.update_scratch_size,
                }))
            }
            None => Err(VulkanError::MissingExtension(String::from(
                "VK_KHR_acceleration_structure",
            ))),
        }
    }
}

pub struct DeviceScratchBuffer {
    buffer: Arc<Buffer>,
    buffer_device_addr: u64,
}

impl DeviceOwned for DeviceScratchBuffer {
    fn get_parent_device(&self) -> Arc<Device> {
        self.buffer.get_parent_device()
    }
}

impl BufferTrait for DeviceScratchBuffer {
    fn size(&self) -> u64 {
        self.buffer.size()
    }

    fn native_handle(&self) -> u64 {
        self.buffer.native_handle()
    }
}

impl DeviceScratchBuffer {
    pub(crate) fn addr(&self) -> ash::vk::DeviceOrHostAddressKHR {
        ash::vk::DeviceOrHostAddressKHR {
            device_address: self.buffer_device_addr,
        }
    }

    pub fn new(memory_pool: Arc<MemoryPool>, size: u64) -> VulkanResult<Arc<Self>> {
        match Buffer::new(
            memory_pool.clone(),
            ConcreteBufferDescriptor::new(
                BufferUsage::Unmanaged(
                    (
                        ash::vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS |
                        ash::vk::BufferUsageFlags::STORAGE_BUFFER
                    ).as_raw(),
                ),
                size,
            ),
            None,
            None,
        ) {
            Ok(buffer) => {
                let info = ash::vk::BufferDeviceAddressInfo::builder().buffer(buffer.ash_handle());

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
            Err(err) => Err(err),
        }
    }
}

pub struct HostScratchBuffer {
    buffer: Mutex<Vec<u8>>,
}

impl HostScratchBuffer {
    pub(crate) fn address(&self) -> ash::vk::DeviceOrHostAddressKHR {
        match self.buffer.lock() {
            Ok(mut lck) => ash::vk::DeviceOrHostAddressKHR {
                host_address: lck.as_mut_slice().as_mut_ptr() as *mut std::ffi::c_void,
            },
            Err(err) => {
                todo!()
            }
        }
    }

    pub fn new(size: u64) -> Arc<Self> {
        Arc::new(Self {
            buffer: Mutex::new(Vec::<u8>::with_capacity(size as usize)),
        })
    }
}

#[derive(Copy, Clone, PartialEq, Eq)]
pub struct BottomLevelTrianglesGroupDecl {
    max_triangles: u32,
    vertex_stride: u64,
    vertex_format: AttributeType,
}

impl BottomLevelTrianglesGroupDecl {
    pub fn new(
        max_triangles: u32,
        vertex_stride: u64,
        vertex_format: AttributeType,
    ) -> Self {
        Self {
            max_triangles,
            vertex_stride,
            vertex_format,
        }
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
        ash::vk::AccelerationStructureGeometryKHR::builder()
            // TODO: .flags()
            .geometry_type(ash::vk::GeometryTypeKHR::TRIANGLES)
            .geometry(
                ash::vk::AccelerationStructureGeometryDataKHR {
                    triangles: ash::vk::AccelerationStructureGeometryTrianglesDataKHR::builder()
                        .index_type(ash::vk::IndexType::UINT32)
                        .max_vertex(self.max_vertices())
                        .vertex_format(self.vertex_format.ash_format())
                        .vertex_stride(self.vertex_stride)
                        .build(),
                }
            )
            .build()
    }
}

pub struct BottomLevelTrianglesGroupData {
    decl: BottomLevelTrianglesGroupDecl,

    index_buffer: Arc<Buffer>,
    vertex_buffer: Arc<Buffer>,
    transform_buffer: Arc<Buffer>,
    
    primitive_offset: u32,
    primitive_count: u32,
    first_vertex: u32,
    transform_offset: u32,
}

impl BottomLevelTrianglesGroupData {
    pub fn new(
        decl: BottomLevelTrianglesGroupDecl,
        index_buffer: Arc<Buffer>,
        vertex_buffer: Arc<Buffer>,
        transform_buffer: Arc<Buffer>,
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

    pub fn index_buffer(&self) -> Arc<Buffer> {
        self.index_buffer.clone()
    }

    pub fn vertex_buffer(&self) -> Arc<Buffer> {
        self.vertex_buffer.clone()
    }

    pub fn transform_buffer(&self) -> Arc<Buffer> {
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
    buffer: Arc<Buffer>,
    buffer_device_addr: u64,
    //builder: Arc<BottomLevelAccelerationStructureBuilder>,
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
        geometries_decl: &[BottomLevelTrianglesGroupDecl]
    ) -> VulkanResult<(u64, u64, u64)> {
        let geometries = geometries_decl.iter().map(|g| g.ash_geometry()).collect::<Vec<ash::vk::AccelerationStructureGeometryKHR>>();
        let max_primitives_count = geometries_decl.iter().map(|g| g.max_triangles()).collect::<Vec<u32>>();

        // From vulkan specs: https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkGetAccelerationStructureBuildSizesKHR.html
        // The srcAccelerationStructure, dstAccelerationStructure, and mode members of pBuildInfo are ignored.
        // Any VkDeviceOrHostAddressKHR members of pBuildInfo are ignored by this command,
        // except that the hostAddress member of VkAccelerationStructureGeometryTrianglesDataKHR::transformData
        // will be examined to check if it is NULL.
        let geometry_info = ash::vk::AccelerationStructureBuildGeometryInfoKHR::builder()
            .geometries(geometries.as_slice())
            .flags(ash::vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
            .ty(ash::vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
            .build();

        match device.ash_ext_acceleration_structure_khr() {
            Some(as_ext) => {
                let build_sizes = unsafe {
                    as_ext.get_acceleration_structure_build_sizes(
                        allowed_building_devices.ash_flags(),
                        &geometry_info,
                        max_primitives_count.as_slice(),
                    )
                };

                Ok(
                    (
                        build_sizes.acceleration_structure_size,
                        build_sizes.build_scratch_size,
                        build_sizes.update_scratch_size,
                    )
                )
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

    //pub fn builder(&self) -> Arc<BottomLevelAccelerationStructureBuilder> {
    //    self.builder.clone()
    //}

    /*pub fn build(&self, scratch_buffer: Arc<HostScratchBuffer>) -> VulkanResult<()> {
        let allowed_building_devices = self.builder.allowed_building_devices();
        if allowed_building_devices == AllowedBuildingDevice::DeviceOnly {
            // TODO: error
        }

        match self
            .buffer
            .get_parent_device()
            .ash_ext_acceleration_structure_khr()
        {
            Some(as_ext) => {
                let (/*mut*/ geometries, geometry_info) = self.builder.ash_structures();

                /*for geometry in geometries.iter_mut() {

                }*/

                let geometry_info = ash::vk::AccelerationStructureBuildGeometryInfoKHR::builder()
                    .geometries(geometries.as_slice())
                    .flags(geometry_info.flags)
                    .ty(geometry_info.ty)
                    .mode(ash::vk::BuildAccelerationStructureModeKHR::BUILD)
                    .src_acceleration_structure(ash::vk::AccelerationStructureKHR::null())
                    .dst_acceleration_structure(self.handle)
                    .scratch_data(scratch_buffer.address())
                    .build();

                let range_infos: smallvec::SmallVec<
                    [ash::vk::AccelerationStructureBuildRangeInfoKHR; 1],
                > = smallvec::smallvec![
                    ash::vk::AccelerationStructureBuildRangeInfoKHR::builder().build()
                ];

                let range_info_0 = range_infos.as_slice();

                match unsafe {
                    as_ext.build_acceleration_structures(
                        ash::vk::DeferredOperationKHR::null(),
                        &[geometry_info],
                        &[range_info_0],
                    )
                } {
                    Ok(_) => Ok(()),
                    Err(err) => Err(VulkanError::Vulkan(
                        err.as_raw(),
                        Some(format!("Error building the BLAS: {}", err.to_string())),
                    )),
                }
            }
            None => Err(VulkanError::MissingExtension(String::from(
                "VK_KHR_acceleration_structure",
            ))),
        }
    }*/

    pub fn new(
        memory_pool: Arc<MemoryPool>,
        //builder: Arc<BottomLevelAccelerationStructureBuilder>,
        buffer_size: u64
    ) -> VulkanResult<Arc<Self>> {
        let device = memory_pool.get_parent_memory_heap().get_parent_device();

        if !memory_pool.features().device_addressable() {
            return Err(VulkanError::Framework(FrameworkError::Unknown(Some(
                String::from("Missing feature on MemoryPool: device_addressable need to be set"),
            ))));
        }

        // TODO: review sharing mode and debug name
        match Buffer::new(
            memory_pool,
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
        ) {
            Ok(buffer) => {
                let info = ash::vk::BufferDeviceAddressInfo::builder().buffer(buffer.ash_handle());

                let buffer_device_addr =
                    unsafe { device.ash_handle().get_buffer_device_address(&info) };

                /*let scratch_data = ash::vk::DeviceOrHostAddressKHR {

                };*/

                match device.ash_ext_acceleration_structure_khr() {
                    Some(as_ext) => {
                        // If deviceAddress is not zero, createFlags must include VK_ACCELERATION_STRUCTURE_CREATE_DEVICE_ADDRESS_CAPTURE_REPLAY_BIT_KHR
                        let create_info = ash::vk::AccelerationStructureCreateInfoKHR::builder()
                            .buffer(buffer.ash_handle())
                            .offset(0)
                            .size(buffer.size())
                            .ty(ash::vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
                            .create_flags(ash::vk::AccelerationStructureCreateFlagsKHR::empty())
                            //.device_address(buffer_device_addr)
                            .build();

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
                                Some(format!(
                                    "Error creating acceleration structure: {}",
                                    err.to_string()
                                )),
                            )),
                        }
                    }
                    None => Err(VulkanError::MissingExtension(String::from(
                        "VK_KHR_acceleration_structure",
                    ))),
                }
            }
            Err(err) => Err(err),
        }
    }
}

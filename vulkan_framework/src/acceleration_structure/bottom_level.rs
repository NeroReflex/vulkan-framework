use std::sync::Arc;

use ash::vk::{BuildAccelerationStructureModeKHR, DeviceOrHostAddressConstKHR};

use crate::{
    acceleration_structure::{
        scratch_buffer::DeviceScratchBuffer, AllowedBuildingDevice, VertexIndexing,
    },
    buffer::{AllocatedBuffer, Buffer, BufferTrait, BufferUsage, ConcreteBufferDescriptor},
    device::{Device, DeviceOwned},
    graphics_pipeline::AttributeType,
    instance::InstanceOwned,
    memory_heap::MemoryHeapOwned,
    memory_pool::MemoryPool,
    prelude::{FrameworkError, VulkanError, VulkanResult},
};

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct BottomLevelTrianglesGroupDecl {
    vertex_indexing: VertexIndexing,
    max_triangles: u32,
    per_vertex_user_stride: u64,
    vertex_format: AttributeType,
}

impl BottomLevelTrianglesGroupDecl {
    /**
     * Define how geometry is stored in memory
     *
     * @param vertex_indexing the vertex indexing buffer format: None means no indexing
     * @param max_triangles the maximum number of triangles
     * @param vertex_format the in-memory format of the vertex position
     * @param per_vertex_user_stride the [unused] space (in bytes) that follows each vertex definition
     */
    pub fn new(
        vertex_indexing: VertexIndexing,
        max_triangles: u32,
        vertex_format: AttributeType,
        per_vertex_user_stride: u64,
    ) -> Self {
        Self {
            vertex_indexing,
            max_triangles,
            per_vertex_user_stride,
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
        self.per_vertex_user_stride
            + match self.vertex_format {
                AttributeType::Float => 4u64,
                AttributeType::Vec1 => 4u64,
                AttributeType::Vec2 => 4u64 * 2,
                AttributeType::Vec3 => 4u64 * 3,
                AttributeType::Vec4 => 4u64 * 4,
                AttributeType::Uint => todo!(),
                AttributeType::Uvec1 => todo!(),
                AttributeType::Uvec2 => todo!(),
                AttributeType::Uvec3 => todo!(),
                AttributeType::Uvec4 => todo!(),
                AttributeType::Sint => todo!(),
                AttributeType::Ivec1 => todo!(),
                AttributeType::Ivec2 => todo!(),
                AttributeType::Ivec3 => todo!(),
                AttributeType::Ivec4 => todo!(),
            }
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
                    .vertex_stride(self.vertex_stride()),
            })
    }
}

pub struct BottomLevelAccelerationStructureIndexBuffer {
    buffer: Arc<AllocatedBuffer>,
    buffer_device_addr: u64,
}

impl BottomLevelAccelerationStructureIndexBuffer {
    pub fn new(
        memory_pool: Arc<MemoryPool>,
        buffer_size: u64,
        debug_name: &Option<&str>,
    ) -> VulkanResult<Arc<Self>> {
        let device = memory_pool.get_parent_memory_heap().get_parent_device();

        // TODO: change index buffer sizes
        let index_buffer_debug_name = debug_name.map(|name| format!("{name}_index_buffer"));
        let index_buffer = Buffer::new(
            device.clone(),
            ConcreteBufferDescriptor::new(
                BufferUsage::Unmanaged(
                    (ash::vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                        | ash::vk::BufferUsageFlags::INDEX_BUFFER
                        | ash::vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS)
                        .as_raw(),
                ),
                buffer_size,
            ),
            None,
            match &index_buffer_debug_name {
                Some(name) => Some(name.as_str()),
                None => None,
            },
        )?;

        let buffer = AllocatedBuffer::new(memory_pool.clone(), index_buffer)?;

        let info = ash::vk::BufferDeviceAddressInfo::default().buffer(buffer.ash_handle());
        let buffer_device_addr = unsafe { device.ash_handle().get_buffer_device_address(&info) };

        Ok(Arc::new(Self {
            buffer,
            buffer_device_addr,
        }))
    }

    pub fn buffer(&self) -> Arc<AllocatedBuffer> {
        self.buffer.clone()
    }

    pub fn buffer_device_addr(&self) -> u64 {
        self.buffer_device_addr.to_owned()
    }
}

pub struct BottomLevelAccelerationStructureVertexBuffer {
    buffer: Arc<AllocatedBuffer>,
    buffer_device_addr: u64,
}

impl BottomLevelAccelerationStructureVertexBuffer {
    pub fn new(
        memory_pool: Arc<MemoryPool>,
        buffer_size: u64,
        debug_name: &Option<&str>,
    ) -> VulkanResult<Arc<Self>> {
        let device = memory_pool.get_parent_memory_heap().get_parent_device();

        // TODO: change vertex buffer sizes
        let vertex_buffer_debug_name = debug_name.map(|name| format!("{name}_vertex_buffer"));
        let vertex_buffer = Buffer::new(
            device.clone(),
            ConcreteBufferDescriptor::new(
                BufferUsage::Unmanaged(
                    (ash::vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                        | ash::vk::BufferUsageFlags::VERTEX_BUFFER
                        | ash::vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS)
                        .as_raw(),
                ),
                buffer_size,
            ),
            None,
            match &vertex_buffer_debug_name {
                Some(name) => Some(name.as_str()),
                None => None,
            },
        )?;

        let buffer = AllocatedBuffer::new(memory_pool.clone(), vertex_buffer)?;

        let info = ash::vk::BufferDeviceAddressInfo::default().buffer(buffer.ash_handle());
        let buffer_device_addr = unsafe { device.ash_handle().get_buffer_device_address(&info) };

        Ok(Arc::new(Self {
            buffer,
            buffer_device_addr,
        }))
    }

    pub fn buffer(&self) -> Arc<AllocatedBuffer> {
        self.buffer.clone()
    }

    pub fn buffer_device_addr(&self) -> u64 {
        self.buffer_device_addr.to_owned()
    }

    /*
     * This function allows the user to estimate a minimum size for provided geometry info.
     *
     * If the operation is successful the tuple returned will be the BLAS build (minimum) scratch buffer size
     */
    pub fn query_minimum_build_scratch_buffer_size(
        &self,
        allowed_building_devices: AllowedBuildingDevice,
        geometries_decl: &[&BottomLevelTrianglesGroupDecl],
    ) -> VulkanResult<u64> {
        let device = self.buffer().get_parent_device();

        let Some(as_ext) = device.ash_ext_acceleration_structure_khr() else {
            return Err(VulkanError::MissingExtension(String::from(
                "VK_KHR_acceleration_structure",
            )));
        };

        let geometries = geometries_decl
            .iter()
            .map(|g| {
                let mut data = g.ash_geometry();

                data.geometry.triangles.vertex_data = DeviceOrHostAddressConstKHR {
                    device_address: self.buffer_device_addr(),
                };

                data
            })
            .collect::<smallvec::SmallVec<[ash::vk::AccelerationStructureGeometryKHR; 4]>>();
        let max_primitives_count = geometries_decl
            .iter()
            .map(|g| g.max_triangles())
            .collect::<smallvec::SmallVec<[u32; 4]>>();

        // See https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkGetAccelerationStructureBuildSizesKHR.html
        let geometry_info = ash::vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .geometries(geometries.as_slice())
            .flags(ash::vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
            .ty(ash::vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
            .mode(BuildAccelerationStructureModeKHR::BUILD);

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
}

pub struct BottomLevelAccelerationStructure {
    triangles_decl: BottomLevelTrianglesGroupDecl,

    handle: ash::vk::AccelerationStructureKHR,
    acceleration_structure_device_addr: u64,

    blas_buffer: Arc<AllocatedBuffer>,
    blas_buffer_device_addr: u64,

    vertex_buffer: Arc<BottomLevelAccelerationStructureVertexBuffer>,

    index_buffer: Option<Arc<BottomLevelAccelerationStructureIndexBuffer>>,

    allowed_building_devices: AllowedBuildingDevice,

    device_build_scratch_buffer: Arc<DeviceScratchBuffer>,
    // TODO: device_update_scratch_buffer: Arc<DeviceScratchBuffer>,

    //builder: Arc<BottomLevelAccelerationStructureBuilder>,
}

impl Drop for BottomLevelAccelerationStructure {
    fn drop(&mut self) {
        let device = self.blas_buffer.get_parent_device();
        match self
            .blas_buffer
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
     * If the operation is successful the tuple returned will be the BLAS (minimum) size.
     */
    pub fn query_minimum_buffer_size(
        device: Arc<Device>,
        allowed_building_devices: AllowedBuildingDevice,
        geometries_decl: &[&BottomLevelTrianglesGroupDecl],
    ) -> VulkanResult<u64> {
        let Some(as_ext) = device.ash_ext_acceleration_structure_khr() else {
            return Err(VulkanError::MissingExtension(String::from(
                "VK_KHR_acceleration_structure",
            )));
        };

        let geometries = geometries_decl
            .iter()
            .map(|g| g.ash_geometry())
            .collect::<smallvec::SmallVec<[ash::vk::AccelerationStructureGeometryKHR; 4]>>();
        let max_primitives_count = geometries_decl
            .iter()
            .map(|g| g.max_triangles())
            .collect::<smallvec::SmallVec<[u32; 4]>>();

        // From vulkan specs: https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkGetAccelerationStructureBuildSizesKHR.html
        // The srcAccelerationStructure, dstAccelerationStructure, and mode members of pBuildInfo are ignored.
        // Any VkDeviceOrHostAddressKHR members of pBuildInfo are ignored by this command,
        // except that the hostAddress member of VkAccelerationStructureGeometryTrianglesDataKHR::transformData
        // will be examined to check if it is NULL.
        let geometry_info = ash::vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .geometries(geometries.as_slice())
            .flags(ash::vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
            .ty(ash::vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL);

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

    pub(crate) fn ash_handle(&self) -> ash::vk::AccelerationStructureKHR {
        self.handle
    }

    pub fn triangles_decl(&self) -> &BottomLevelTrianglesGroupDecl {
        &self.triangles_decl
    }

    pub fn device_addr(&self) -> u64 {
        self.acceleration_structure_device_addr.to_owned()
    }

    pub fn vertex_buffer(&self) -> Arc<BottomLevelAccelerationStructureVertexBuffer> {
        self.vertex_buffer.clone()
    }

    pub fn index_buffer(&self) -> Option<Arc<BottomLevelAccelerationStructureIndexBuffer>> {
        self.index_buffer.clone()
    }

    pub fn buffer_size(&self) -> u64 {
        self.blas_buffer.size()
    }

    pub fn allowed_building_devices(&self) -> AllowedBuildingDevice {
        self.allowed_building_devices.to_owned()
    }

    pub(crate) fn device_build_scratch_buffer(&self) -> Arc<DeviceScratchBuffer> {
        self.device_build_scratch_buffer.clone()
    }

    fn create_blas_buffer(
        memory_pool: Arc<MemoryPool>,
        buffer_size: u64,
        debug_name: &Option<&str>,
    ) -> VulkanResult<(Arc<AllocatedBuffer>, u64)> {
        let device = memory_pool.get_parent_memory_heap().get_parent_device();

        // TODO: change index buffer sizes
        let blas_buffer_debug_name = debug_name.map(|name| format!("{name}_blas_buffer"));
        let blas_buffer = Buffer::new(
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
            match &blas_buffer_debug_name {
                Some(name) => Some(name.as_str()),
                None => None
            },
        )?;

        let buffer = AllocatedBuffer::new(memory_pool.clone(), blas_buffer)?;

        let info = ash::vk::BufferDeviceAddressInfo::default().buffer(buffer.ash_handle());
        let buffer_device_addr = unsafe { device.ash_handle().get_buffer_device_address(&info) };

        Ok((buffer, buffer_device_addr))
    }

    pub fn new(
        memory_pool: Arc<MemoryPool>,
        //builder: Arc<BottomLevelAccelerationStructureBuilder>,
        triangles_decl: BottomLevelTrianglesGroupDecl,
        allowed_building_devices: AllowedBuildingDevice,
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

        // TODO: review sharing mode for buffers

        // WARNING: this sets the maximum number of vertices equals to the maximum number of vertices,
        // effectively negating the benefit of having an index buffer.
        let vertex_buffer = BottomLevelAccelerationStructureVertexBuffer::new(
            memory_pool.clone(),
            triangles_decl.vertex_stride() * 3u64 * (triangles_decl.max_vertices() as u64),
            &debug_name,
        )?;

        let index_buffer = match triangles_decl.vertex_indexing() {
            VertexIndexing::None => None,
            index_type => Some(BottomLevelAccelerationStructureIndexBuffer::new(
                memory_pool.clone(),
                index_type.size() * (triangles_decl.max_vertices() as u64),
                &debug_name,
            )?),
        };

        let geometries_decl = [&triangles_decl];
        let blas_buffer_size = Self::query_minimum_buffer_size(
            device.clone(),
            allowed_building_devices,
            &geometries_decl,
        )?;

        let (blas_buffer, blas_buffer_device_addr) =
            Self::create_blas_buffer(memory_pool.clone(), blas_buffer_size, &debug_name)?;

        let build_scratch_buffer_size = vertex_buffer
            .query_minimum_build_scratch_buffer_size(allowed_building_devices, &geometries_decl)?;
        let device_build_scratch_buffer =
            DeviceScratchBuffer::new(memory_pool.clone(), build_scratch_buffer_size)?;

        // If deviceAddress is not zero, createFlags must include VK_ACCELERATION_STRUCTURE_CREATE_DEVICE_ADDRESS_CAPTURE_REPLAY_BIT_KHR
        let create_info = ash::vk::AccelerationStructureCreateInfoKHR::default()
            .buffer(blas_buffer.ash_handle())
            .offset(0)
            .size(blas_buffer.size())
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
                let acceleration_structure_device_addr =
                    unsafe { as_ext.get_acceleration_structure_device_address(&info) };
                Ok(Arc::new(Self {
                    triangles_decl,
                    handle,
                    acceleration_structure_device_addr,
                    blas_buffer,
                    blas_buffer_device_addr,
                    vertex_buffer,
                    index_buffer,
                    allowed_building_devices,
                    device_build_scratch_buffer,
                }))
            }
            Err(err) => Err(VulkanError::Vulkan(
                err.as_raw(),
                Some(format!("Error creating acceleration structure: {}", err)),
            )),
        }
    }
}

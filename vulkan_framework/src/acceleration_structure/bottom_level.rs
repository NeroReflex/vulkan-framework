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
    queue_family::QueueFamily,
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

    #[inline]
    pub fn vertex_indexing(&self) -> VertexIndexing {
        self.vertex_indexing
    }

    #[inline]
    pub fn max_triangles(&self) -> u32 {
        self.max_triangles
    }

    #[inline]
    pub fn max_vertices(&self) -> u32 {
        self.max_triangles() * 3u32
    }

    #[inline]
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

    #[inline]
    pub fn vertex_format(&self) -> AttributeType {
        self.vertex_format
    }

    pub(crate) fn ash_geometry(&self) -> ash::vk::AccelerationStructureGeometryKHR {
        ash::vk::AccelerationStructureGeometryKHR::default()
            // TODO: .flags(ash::vk::GeometryFlagsKHR::NO_DUPLICATE_ANY_HIT_INVOCATION | ash::vk::GeometryFlagsKHR::OPAQUE)
            .geometry_type(ash::vk::GeometryTypeKHR::TRIANGLES)
            .geometry(ash::vk::AccelerationStructureGeometryDataKHR {
                triangles: ash::vk::AccelerationStructureGeometryTrianglesDataKHR::default()
                    .index_type(self.vertex_indexing().ash_index_type())
                    // this is very incredibly stupid: the vulkan specification states:
                    // maxVertex is the number of vertices in vertexData minus one
                    // See https://registry.khronos.org/vulkan/specs/latest/man/html/VkAccelerationStructureGeometryTrianglesDataKHR.html
                    .max_vertex(self.max_vertices() - 1)
                    .vertex_format(self.vertex_format.ash_format())
                    .vertex_stride(self.vertex_stride()),
            })
    }
}

pub struct BottomLevelAccelerationStructureTransformBuffer {
    buffer: Arc<AllocatedBuffer>,
    buffer_device_addr: u64,
}

impl BottomLevelAccelerationStructureTransformBuffer {
    pub fn new(
        memory_pool: Arc<MemoryPool>,
        usage: BufferUsage,
        max_instances: u64,
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
                        | ash::vk::BufferUsageFlags::INDEX_BUFFER
                        | ash::vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS)
                        .as_raw(),
                ),
                (core::mem::size_of::<ash::vk::AccelerationStructureInstanceKHR>() as u64)
                    * max_instances,
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

pub struct BottomLevelAccelerationStructureIndexBuffer {
    buffer: Arc<AllocatedBuffer>,
    buffer_device_addr: u64,
}

impl BottomLevelAccelerationStructureIndexBuffer {
    pub fn new(
        memory_pool: Arc<MemoryPool>,
        usage: BufferUsage,
        buffer_size: u64,
        sharing: Option<&[std::sync::Weak<QueueFamily>]>,
        debug_name: &Option<&str>,
    ) -> VulkanResult<Arc<Self>> {
        let device = memory_pool.get_parent_memory_heap().get_parent_device();

        // TODO: change index buffer sizes
        let index_buffer_debug_name = debug_name.map(|name| format!("{name}_index_buffer"));
        let index_buffer = Buffer::new(
            device.clone(),
            ConcreteBufferDescriptor::new(
                BufferUsage::Unmanaged(
                    usage.ash_usage().as_raw() |
                    (ash::vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                        | ash::vk::BufferUsageFlags::INDEX_BUFFER
                        | ash::vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS)
                        .as_raw(),
                ),
                buffer_size,
            ),
            sharing,
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

    #[inline]
    pub fn buffer(&self) -> Arc<AllocatedBuffer> {
        self.buffer.clone()
    }

    #[inline]
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
        usage: BufferUsage,
        sharing: Option<&[std::sync::Weak<QueueFamily>]>,
        debug_name: &Option<&str>,
    ) -> VulkanResult<Arc<Self>> {
        let device = memory_pool.get_parent_memory_heap().get_parent_device();

        // TODO: change vertex buffer sizes
        let vertex_buffer_debug_name = debug_name.map(|name| format!("{name}_vertex_buffer"));
        let vertex_buffer = Buffer::new(
            device.clone(),
            ConcreteBufferDescriptor::new(
                BufferUsage::Unmanaged(
                    usage.ash_usage().as_raw() |
                    (ash::vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                        | ash::vk::BufferUsageFlags::VERTEX_BUFFER
                        | ash::vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS)
                        .as_raw(),
                ),
                buffer_size,
            ),
            sharing,
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

    #[inline]
    pub fn buffer(&self) -> Arc<AllocatedBuffer> {
        self.buffer.clone()
    }

    #[inline]
    pub fn buffer_device_addr(&self) -> u64 {
        self.buffer_device_addr.to_owned()
    }
}

pub struct BottomLevelAccelerationStructure {
    max_instances: u64,
    triangles_decl: smallvec::SmallVec<[BottomLevelTrianglesGroupDecl; 1]>,

    handle: ash::vk::AccelerationStructureKHR,
    acceleration_structure_device_addr: u64,

    blas_buffer: Arc<AllocatedBuffer>,
    blas_buffer_device_addr: u64,

    vertex_buffer: Arc<BottomLevelAccelerationStructureVertexBuffer>,
    index_buffer: Arc<BottomLevelAccelerationStructureIndexBuffer>,
    transform_buffer: Arc<BottomLevelAccelerationStructureTransformBuffer>,

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

    pub(crate) fn static_ash_geometry<'a>(
        geometries_decl: &[&'a BottomLevelTrianglesGroupDecl],
        vertex_buffer: Arc<BottomLevelAccelerationStructureVertexBuffer>,
        index_buffer: Arc<BottomLevelAccelerationStructureIndexBuffer>,
        instance_buffer: Arc<BottomLevelAccelerationStructureTransformBuffer>,
    ) -> smallvec::SmallVec<[ash::vk::AccelerationStructureGeometryKHR<'a>; 1]> {
        geometries_decl
            .iter()
            .map(|g| {
                let mut data = g.ash_geometry();

                data.geometry.triangles.vertex_data = DeviceOrHostAddressConstKHR {
                    device_address: vertex_buffer.buffer_device_addr(),
                };

                data.geometry.triangles.index_data = DeviceOrHostAddressConstKHR {
                    device_address: index_buffer.buffer_device_addr(),
                };

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
        self.triangles_decl
            .iter()
            .map(|g| {
                let mut data = g.ash_geometry();

                data.geometry.triangles.vertex_data = DeviceOrHostAddressConstKHR {
                    device_address: self.vertex_buffer().buffer_device_addr(),
                };

                data.geometry.triangles.index_data = DeviceOrHostAddressConstKHR {
                    device_address: self.index_buffer().buffer_device_addr(),
                };

                data.geometry.triangles.transform_data = DeviceOrHostAddressConstKHR {
                    device_address: self.instance_buffer().buffer_device_addr(),
                };

                data
            })
            .collect::<smallvec::SmallVec<_>>()
    }

    /*
     * This function allows the user to estimate a minimum size for provided geometry info.
     *
     * If the operation is successful the tuple returned will be the BLAS build (minimum) scratch buffer size
     */
    pub fn query_minimum_build_scratch_buffer_size(
        geometries_decl: &[&BottomLevelTrianglesGroupDecl],
        allowed_building_devices: AllowedBuildingDevice,
        vertex_buffer: Arc<BottomLevelAccelerationStructureVertexBuffer>,
        index_buffer: Arc<BottomLevelAccelerationStructureIndexBuffer>,
        instance_buffer: Arc<BottomLevelAccelerationStructureTransformBuffer>,
    ) -> VulkanResult<u64> {
        let device = vertex_buffer.buffer().get_parent_device();

        let Some(as_ext) = device.ash_ext_acceleration_structure_khr() else {
            return Err(VulkanError::MissingExtension(String::from(
                "VK_KHR_acceleration_structure",
            )));
        };

        let geometries = Self::static_ash_geometry(
            geometries_decl,
            vertex_buffer,
            index_buffer,
            instance_buffer,
        );

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

        let mut blas_size_info = ash::vk::AccelerationStructureBuildSizesInfoKHR::default();

        unsafe {
            as_ext.get_acceleration_structure_build_sizes(
                allowed_building_devices.ash_flags(),
                &geometry_info,
                max_primitives_count.as_slice(),
                &mut blas_size_info,
            )
        };

        match device.ray_tracing_info() {
            Some(rt_info) => Ok((blas_size_info.build_scratch_size
                + ((rt_info.min_acceleration_structure_scratch_offset_alignment() as u64) - 1u64))
                & !((rt_info.min_acceleration_structure_scratch_offset_alignment() as u64) - 1u64)),
            None => Ok(blas_size_info.build_scratch_size),
        }
    }

    #[inline]
    pub(crate) fn ash_handle(&self) -> ash::vk::AccelerationStructureKHR {
        self.handle
    }

    #[inline]
    pub fn triangles_decl(&self) -> &[BottomLevelTrianglesGroupDecl] {
        self.triangles_decl.as_slice()
    }

    #[inline]
    pub fn max_instances(&self) -> u64 {
        self.max_instances
    }

    #[inline]
    pub fn device_addr(&self) -> u64 {
        self.acceleration_structure_device_addr.to_owned()
    }

    #[inline]
    pub fn vertex_buffer(&self) -> Arc<BottomLevelAccelerationStructureVertexBuffer> {
        self.vertex_buffer.clone()
    }

    #[inline]
    pub fn index_buffer(&self) -> Arc<BottomLevelAccelerationStructureIndexBuffer> {
        self.index_buffer.clone()
    }

    #[inline]
    pub fn instance_buffer(&self) -> Arc<BottomLevelAccelerationStructureTransformBuffer> {
        self.transform_buffer.clone()
    }

    #[inline]
    pub fn buffer_size(&self) -> u64 {
        self.blas_buffer.size()
    }

    #[inline]
    pub fn allowed_building_devices(&self) -> AllowedBuildingDevice {
        self.allowed_building_devices.to_owned()
    }

    #[inline]
    pub(crate) fn device_build_scratch_buffer(&self) -> Arc<DeviceScratchBuffer> {
        self.device_build_scratch_buffer.clone()
    }

    fn create_blas_buffer(
        memory_pool: Arc<MemoryPool>,
        buffer_size: u64,
        sharing: Option<&[std::sync::Weak<QueueFamily>]>,
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
            sharing,
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
        triangles_decl: BottomLevelTrianglesGroupDecl,
        max_instances: u64,
        allowed_building_devices: AllowedBuildingDevice,
        vertex_buffer_usage: BufferUsage,
        index_buffer_usage: BufferUsage,
        transform_buffer_usage: BufferUsage,
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
            vertex_buffer_usage,
            sharing.clone(),
            &debug_name,
        )?;

        let index_buffer = BottomLevelAccelerationStructureIndexBuffer::new(
            memory_pool.clone(),
            index_buffer_usage,
            triangles_decl.vertex_indexing().size() * (triangles_decl.max_vertices() as u64),
            sharing.clone(),
            &debug_name,
        )?;

        let transform_buffer = BottomLevelAccelerationStructureTransformBuffer::new(
            memory_pool.clone(),
            transform_buffer_usage,
            max_instances,
            sharing.clone(),
            &debug_name,
        )?;

        let geometries_decl = [&triangles_decl];
        let blas_buffer_size = Self::query_minimum_buffer_size(
            device.clone(),
            allowed_building_devices,
            &geometries_decl,
        )?;

        let (blas_buffer, blas_buffer_device_addr) = Self::create_blas_buffer(
            memory_pool.clone(),
            blas_buffer_size,
            sharing.clone(),
            &debug_name,
        )?;

        let build_scratch_buffer_size = Self::query_minimum_build_scratch_buffer_size(
            &geometries_decl,
            allowed_building_devices,
            vertex_buffer.clone(),
            index_buffer.clone(),
            transform_buffer.clone(),
        )?;

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

                let triangles_decl = smallvec::smallvec![triangles_decl];
                Ok(Arc::new(Self {
                    max_instances,
                    triangles_decl,
                    handle,
                    acceleration_structure_device_addr,
                    blas_buffer,
                    blas_buffer_device_addr,
                    vertex_buffer,
                    index_buffer,
                    transform_buffer,
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

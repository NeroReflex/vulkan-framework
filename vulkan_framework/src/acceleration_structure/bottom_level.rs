use std::sync::Arc;

use ash::vk::{
    AccelerationStructureGeometryKHR, BuildAccelerationStructureModeKHR,
    DeviceOrHostAddressConstKHR,
};

use crate::{
    acceleration_structure::{
        scratch_buffer::DeviceScratchBuffer, AllowedBuildingDevice, VertexIndexing,
    },
    buffer::{AllocatedBuffer, Buffer, BufferTrait, BufferUsage, ConcreteBufferDescriptor},
    device::DeviceOwned,
    graphics_pipeline::AttributeType,
    instance::InstanceOwned,
    memory_heap::MemoryHeapOwned,
    memory_pool::{MemoryPool, MemoryPoolBacked},
    prelude::{FrameworkError, VulkanError, VulkanResult},
    queue_family::QueueFamily,
};

/// This is used to describe the vertex buffer of a bottom level AS
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct BottomLevelVerticesTopologyDecl {
    max_vertices: u32,
    per_vertex_user_stride: u64,
    vertex_format: AttributeType,
}

impl BottomLevelVerticesTopologyDecl {
    /**
     * Define how geometry is stored in memory
     *
     * @param vertex_indexing the vertex indexing buffer format: None means no indexing
     * @param max_triangles the maximum number of triangles
     * @param vertex_format the in-memory format of the vertex position
     * @param per_vertex_user_stride the [unused] space (in bytes) that follows each vertex definition
     */
    pub fn new(
        max_vertices: u32,
        vertex_format: AttributeType,
        per_vertex_user_stride: u64,
    ) -> Self {
        Self {
            max_vertices,
            per_vertex_user_stride,
            vertex_format,
        }
    }

    #[inline]
    pub fn max_vertices(&self) -> u32 {
        self.max_vertices
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
}

/// This is used to describe the index buffer of a bottom level AS
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct BottomLevelTrianglesGroupDecl {
    vertex_indexing: VertexIndexing,
    max_triangles: u32,
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
    pub fn new(vertex_indexing: VertexIndexing, max_triangles: u32) -> Self {
        Self {
            vertex_indexing,
            max_triangles,
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
}

pub struct BottomLevelAccelerationStructureTransformBuffer {
    buffer: Arc<AllocatedBuffer>,
    buffer_device_addr: u64,
}

const IDENTITY_MATRIX: ash::vk::TransformMatrixKHR = ash::vk::TransformMatrixKHR {
    matrix: [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
};

impl BottomLevelAccelerationStructureTransformBuffer {
    pub fn new(
        memory_pool: Arc<MemoryPool>,
        usage: BufferUsage,
        sharing: Option<&[std::sync::Weak<QueueFamily>]>,
        debug_name: &Option<&str>,
    ) -> VulkanResult<Arc<Self>> {
        let device = memory_pool.get_parent_memory_heap().get_parent_device();

        // TODO: change index buffer sizes
        let transform_buffer_debug_name = debug_name.map(|name| format!("{name}_transform_buffer"));
        let transform_buffer = Buffer::new(
            device.clone(),
            ConcreteBufferDescriptor::new(
                BufferUsage::from(
                    <BufferUsage as Into<ash::vk::BufferUsageFlags>>::into(usage).as_raw() |
                    (ash::vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                        | ash::vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS)
                        .as_raw(),
                ),
                core::mem::size_of::<ash::vk::TransformMatrixKHR>() as u64,
            ),
            sharing,
            match &transform_buffer_debug_name {
                Some(name) => Some(name.as_str()),
                None => None,
            },
        )?;

        let buffer = AllocatedBuffer::new(memory_pool.clone(), transform_buffer)?;

        // preload the identity matrix for convenience
        memory_pool
            .write_raw_data(buffer.allocation_offset(), &[IDENTITY_MATRIX])
            .unwrap();

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

    triangles_decl: BottomLevelTrianglesGroupDecl,
}

#[derive(Clone)]
pub enum BottomLevelIndexBufferSpecifier {
    Preallocated(Arc<BottomLevelAccelerationStructureIndexBuffer>),
    Allocate(BufferUsage, BottomLevelTrianglesGroupDecl),
}

impl MemoryPoolBacked for BottomLevelAccelerationStructureIndexBuffer {
    fn get_backing_memory_pool(&self) -> Arc<MemoryPool> {
        self.buffer.get_backing_memory_pool()
    }

    fn allocation_offset(&self) -> u64 {
        self.buffer.allocation_offset()
    }

    fn allocation_size(&self) -> u64 {
        self.buffer.allocation_size()
    }
}

impl BottomLevelAccelerationStructureIndexBuffer {
    pub fn new(
        memory_pool: Arc<MemoryPool>,
        usage: BufferUsage,
        triangles_decl: BottomLevelTrianglesGroupDecl,
        sharing: Option<&[std::sync::Weak<QueueFamily>]>,
        debug_name: &Option<&str>,
    ) -> VulkanResult<Arc<Self>> {
        let device = memory_pool.get_parent_memory_heap().get_parent_device();
        let index_buffer_debug_name = debug_name.map(|name| format!("{name}_index_buffer"));
        let index_buffer = Buffer::new(
            device.clone(),
            ConcreteBufferDescriptor::new(
                BufferUsage::from(
                    <BufferUsage as Into<ash::vk::BufferUsageFlags>>::into(usage).as_raw() |
                    (ash::vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                        | ash::vk::BufferUsageFlags::INDEX_BUFFER
                        | ash::vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS)
                        .as_raw(),
                ),
                // this does not include vertex stride size because it's not part of the index buffer
                triangles_decl.vertex_indexing().size()
                        * (triangles_decl.max_vertices() as u64),
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
            triangles_decl,
        }))
    }

    #[inline]
    pub fn buffer(&self) -> Arc<AllocatedBuffer> {
        self.buffer.clone()
    }

    #[inline]
    pub fn size(&self) -> u64 {
        self.buffer.size()
    }

    #[inline]
    pub fn buffer_device_addr(&self) -> u64 {
        self.buffer_device_addr.to_owned()
    }

    #[inline]
    pub fn triangles_decl(&self) -> &BottomLevelTrianglesGroupDecl {
        &self.triangles_decl
    }
}

#[derive(Clone)]
pub enum BottomLevelVertexBufferSpecifier {
    Preallocated(Arc<BottomLevelAccelerationStructureVertexBuffer>),
    Allocate(BufferUsage, BottomLevelVerticesTopologyDecl),
}

pub struct BottomLevelAccelerationStructureVertexBuffer {
    buffer: Arc<AllocatedBuffer>,
    buffer_device_addr: u64,

    triangles_topology: BottomLevelVerticesTopologyDecl,
}

impl MemoryPoolBacked for BottomLevelAccelerationStructureVertexBuffer {
    #[inline]
    fn get_backing_memory_pool(&self) -> Arc<MemoryPool> {
        self.buffer.get_backing_memory_pool()
    }

    #[inline]
    fn allocation_offset(&self) -> u64 {
        self.buffer.allocation_offset()
    }

    #[inline]
    fn allocation_size(&self) -> u64 {
        self.buffer.allocation_size()
    }
}

impl BottomLevelAccelerationStructureVertexBuffer {
    pub fn new(
        memory_pool: Arc<MemoryPool>,
        triangles_topology: BottomLevelVerticesTopologyDecl,
        usage: BufferUsage,
        sharing: Option<&[std::sync::Weak<QueueFamily>]>,
        debug_name: &Option<&str>,
    ) -> VulkanResult<Arc<Self>> {
        let device = memory_pool.get_parent_memory_heap().get_parent_device();
        let buffer_size = triangles_topology.vertex_stride()
            * triangles_topology.max_vertices() as u64;

        // TODO: change vertex buffer sizes
        let vertex_buffer_debug_name = debug_name.map(|name| format!("{name}_vertex_buffer"));
        let vertex_buffer = Buffer::new(
            device.clone(),
            ConcreteBufferDescriptor::new(
                BufferUsage::from(
                    <BufferUsage as Into<ash::vk::BufferUsageFlags>>::into(usage).as_raw() |
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
            triangles_topology,
        }))
    }

    #[inline]
    pub fn buffer(&self) -> Arc<AllocatedBuffer> {
        self.buffer.clone()
    }

    #[inline]
    pub fn size(&self) -> u64 {
        self.buffer.size()
    }

    #[inline]
    pub fn buffer_device_addr(&self) -> u64 {
        self.buffer_device_addr.to_owned()
    }

    #[inline]
    pub fn triangles_topology(&self) -> &BottomLevelVerticesTopologyDecl {
        &self.triangles_topology
    }
}

pub struct BottomLevelAccelerationStructure {
    triangles_decl: smallvec::SmallVec<
        [(
            BottomLevelVerticesTopologyDecl,
            BottomLevelTrianglesGroupDecl,
        ); 1],
    >,

    handle: ash::vk::AccelerationStructureKHR,
    acceleration_structure_device_addr: u64,

    blas_buffer: Arc<AllocatedBuffer>,
    //blas_buffer_device_addr: u64,
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
        geometries_decl: &[(
            &BottomLevelVerticesTopologyDecl,
            &BottomLevelTrianglesGroupDecl,
        )],
        allowed_building_devices: AllowedBuildingDevice,
        vertex_buffer: Arc<BottomLevelAccelerationStructureVertexBuffer>,
        index_buffer: Arc<BottomLevelAccelerationStructureIndexBuffer>,
        transform_buffer: Arc<BottomLevelAccelerationStructureTransformBuffer>,
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
            transform_buffer,
        );

        let max_primitives_count = geometries_decl
            .iter()
            .map(|(_, g)| g.max_triangles())
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

        Ok(build_sizes.acceleration_structure_size)
    }

    fn ash_geometry_single<'a>(
        triangles_decl: &BottomLevelTrianglesGroupDecl,
        triangles_topology: &BottomLevelVerticesTopologyDecl,
    ) -> ash::vk::AccelerationStructureGeometryKHR<'a> {
        ash::vk::AccelerationStructureGeometryKHR::default()
            // TODO: .flags(ash::vk::GeometryFlagsKHR::NO_DUPLICATE_ANY_HIT_INVOCATION | ash::vk::GeometryFlagsKHR::OPAQUE)
            .geometry_type(ash::vk::GeometryTypeKHR::TRIANGLES)
            .geometry(ash::vk::AccelerationStructureGeometryDataKHR {
                triangles: ash::vk::AccelerationStructureGeometryTrianglesDataKHR::default()
                    .index_type(triangles_decl.vertex_indexing().ash_index_type())
                    // this is very incredibly stupid: the vulkan specification states:
                    // maxVertex is the number of vertices in vertexData minus one
                    // See https://registry.khronos.org/vulkan/specs/latest/man/html/VkAccelerationStructureGeometryTrianglesDataKHR.html
                    .max_vertex(triangles_decl.max_vertices() - 1)
                    .vertex_format(triangles_topology.vertex_format().ash_format())
                    .vertex_stride(triangles_topology.vertex_stride()),
            })
    }

    pub(crate) fn static_ash_geometry<'a>(
        geometries_decl: &[(
            &BottomLevelVerticesTopologyDecl,
            &BottomLevelTrianglesGroupDecl,
        )],
        vertex_buffer: Arc<BottomLevelAccelerationStructureVertexBuffer>,
        index_buffer: Arc<BottomLevelAccelerationStructureIndexBuffer>,
        transform_buffer: Arc<BottomLevelAccelerationStructureTransformBuffer>,
    ) -> smallvec::SmallVec<[ash::vk::AccelerationStructureGeometryKHR<'a>; 1]> {
        geometries_decl
            .iter()
            .map(|(t, g)| {
                let mut data = Self::ash_geometry_single(*g, *t);

                data.geometry.triangles.vertex_data = DeviceOrHostAddressConstKHR {
                    device_address: vertex_buffer.buffer_device_addr(),
                };

                data.geometry.triangles.index_data = DeviceOrHostAddressConstKHR {
                    device_address: index_buffer.buffer_device_addr(),
                };

                data.geometry.triangles.transform_data = DeviceOrHostAddressConstKHR {
                    device_address: transform_buffer.buffer_device_addr(),
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
            .map(|(a, b)| {
                let mut data = Self::ash_geometry_single(b, a);

                data.geometry.triangles.vertex_data = DeviceOrHostAddressConstKHR {
                    device_address: self.vertex_buffer().buffer_device_addr(),
                };

                data.geometry.triangles.index_data = DeviceOrHostAddressConstKHR {
                    device_address: self.index_buffer().buffer_device_addr(),
                };
                /*
                                data.geometry.triangles.transform_data = DeviceOrHostAddressConstKHR {
                                    device_address: self.transform_buffer().buffer_device_addr(),
                                };
                */
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
        geometries_decl: &[(
            &BottomLevelVerticesTopologyDecl,
            &BottomLevelTrianglesGroupDecl,
        )],
        allowed_building_devices: AllowedBuildingDevice,
        vertex_buffer: Arc<BottomLevelAccelerationStructureVertexBuffer>,
        index_buffer: Arc<BottomLevelAccelerationStructureIndexBuffer>,
        transform_buffer: Arc<BottomLevelAccelerationStructureTransformBuffer>,
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
            transform_buffer,
        );

        let max_primitives_count = geometries_decl
            .iter()
            .map(|(_, g)| g.max_triangles())
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
    pub fn triangles_decl(
        &self,
    ) -> &[(
        BottomLevelVerticesTopologyDecl,
        BottomLevelTrianglesGroupDecl,
    )] {
        self.triangles_decl.as_slice()
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
    pub fn transform_buffer(&self) -> Arc<BottomLevelAccelerationStructureTransformBuffer> {
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

    #[inline]
    pub fn max_primitives_count(&self) -> u32 {
        self.triangles_decl().iter().map(|(_, g)| g.max_triangles()).sum()
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
                BufferUsage::from(
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
        allowed_building_devices: AllowedBuildingDevice,
        vertex_buffer_specification: BottomLevelVertexBufferSpecifier,
        index_buffer_specification: BottomLevelIndexBufferSpecifier,
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
            return Err(VulkanError::Framework(
                FrameworkError::MemoryPoolNotAddressable,
            ));
        }

        let vertex_buffer = match vertex_buffer_specification {
            BottomLevelVertexBufferSpecifier::Preallocated(
                bottom_level_acceleration_structure_vertex_buffer,
            ) => bottom_level_acceleration_structure_vertex_buffer,
            BottomLevelVertexBufferSpecifier::Allocate(vertex_buffer_usage, triangles_topology) => {
                BottomLevelAccelerationStructureVertexBuffer::new(
                    memory_pool.clone(),
                    triangles_topology,
                    vertex_buffer_usage,
                    sharing,
                    &debug_name,
                )?
            }
        };

        let triangles_topology = vertex_buffer.triangles_topology().to_owned();

        let index_buffer = match index_buffer_specification {
            BottomLevelIndexBufferSpecifier::Preallocated(
                bottom_level_acceleration_structure_index_buffer,
            ) => bottom_level_acceleration_structure_index_buffer,
            BottomLevelIndexBufferSpecifier::Allocate(index_buffer_usage, triangles_decl) => {
                BottomLevelAccelerationStructureIndexBuffer::new(
                    memory_pool.clone(),
                    index_buffer_usage,
                    triangles_decl,
                    sharing,
                    &debug_name,
                )?
            }
        };

        let transform_buffer = BottomLevelAccelerationStructureTransformBuffer::new(
            memory_pool.clone(),
            transform_buffer_usage,
            sharing,
            &debug_name,
        )?;

        let triangles_decl = index_buffer.triangles_decl().to_owned();

        let geometries_decl = [(&triangles_topology, &triangles_decl)];

        let blas_buffer_size = Self::query_minimum_buffer_size(
            geometries_decl.as_slice(),
            allowed_building_devices,
            vertex_buffer.clone(),
            index_buffer.clone(),
            transform_buffer.clone(),
        )?;

        let build_scratch_buffer_size = Self::query_minimum_build_scratch_buffer_size(
            geometries_decl.as_slice(),
            allowed_building_devices,
            vertex_buffer.clone(),
            index_buffer.clone(),
            transform_buffer.clone(),
        )?;

        let (blas_buffer, _blas_buffer_device_addr) =
            Self::create_blas_buffer(memory_pool.clone(), blas_buffer_size, sharing, &debug_name)?;

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

        let handle = unsafe {
            as_ext.create_acceleration_structure(
                &create_info,
                device.get_parent_instance().get_alloc_callbacks(),
            )
        }?;

        let info = ash::vk::AccelerationStructureDeviceAddressInfoKHR::default()
            .acceleration_structure(handle);
        let acceleration_structure_device_addr =
            unsafe { as_ext.get_acceleration_structure_device_address(&info) };

        let triangles_decl = smallvec::smallvec![(triangles_topology, triangles_decl)];
        Ok(Arc::new(Self {
            triangles_decl,
            handle,
            acceleration_structure_device_addr,
            blas_buffer,
            //blas_buffer_device_addr,
            vertex_buffer,
            index_buffer,
            transform_buffer,
            allowed_building_devices,
            device_build_scratch_buffer,
        }))
    }

    pub(crate) fn ash_build_info(
        &self,
        primitive_offset: u32,
        primitive_count: u32,
        first_vertex: u32,
        transform_offset: u32,
    ) -> VulkanResult<(
        smallvec::SmallVec<[AccelerationStructureGeometryKHR<'_>; 1]>,
        smallvec::SmallVec<
            [smallvec::SmallVec<[ash::vk::AccelerationStructureBuildRangeInfoKHR; 1]>; 1],
        >,
    )> {
        let geometries = self.ash_geometry();

        let range_infos = smallvec::smallvec![smallvec::smallvec![
            ash::vk::AccelerationStructureBuildRangeInfoKHR::default()
                .primitive_offset(primitive_offset)
                .primitive_count(primitive_count)
                .first_vertex(first_vertex)
                .transform_offset(transform_offset),
        ]];

        // TODO: ash::vk::AccelerationStructureBuildGeometryInfoKHR

        Ok((geometries, range_infos))
    }
}

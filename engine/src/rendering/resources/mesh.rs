use std::sync::Arc;

use vulkan_framework::{
    acceleration_structure::{
        AllowedBuildingDevice,
        bottom_level::{
            BottomLevelAccelerationStructure, BottomLevelAccelerationStructureIndexBuffer,
            BottomLevelAccelerationStructureVertexBuffer, BottomLevelIndexBufferSpecifier,
            BottomLevelTrianglesGroupDecl, BottomLevelVertexBufferSpecifier,
            BottomLevelVerticesTopologyDecl,
        },
    },
    binding_tables::required_memory_type,
    buffer::BufferUsage,
    command_buffer::CommandBufferRecorder,
    descriptor_pool::{
        DescriptorPool, DescriptorPoolConcreteDescriptor,
        DescriptorPoolSizesAcceletarionStructureKHR, DescriptorPoolSizesConcreteDescriptor,
    },
    device::DeviceOwned,
    memory_allocator::{DefaultAllocator, MemoryAllocator},
    memory_heap::{ConcreteMemoryHeapDescriptor, MemoryHeap},
    memory_pool::{MemoryPool, MemoryPoolFeature, MemoryPoolFeatures},
    queue::Queue,
    queue_family::{QueueFamily, QueueFamilyOwned},
};

use crate::rendering::{
    RenderingError, RenderingResult, resources::collection::LoadableResourcesCollection,
};

type MeshType = Arc<BottomLevelAccelerationStructure>;

pub struct MeshManager {
    debug_name: String,

    queue: Arc<Queue>,

    _descriptor_pool: Arc<DescriptorPool>,

    memory_pool: Arc<MemoryPool>,

    //descriptor_sets: smallvec::SmallVec<[Arc<DescriptorSet>; MAX_FRAMES_IN_FLIGHT_NO_MALLOC]>,
    meshes: LoadableResourcesCollection<MeshType>,
}

impl MeshManager {
    #[inline]
    pub fn fetch_loaded(&self, index: usize) -> Option<&MeshType> {
        self.meshes.fetch_loaded(index)
    }

    #[inline]
    pub fn foreach_loaded<F>(&self, function: F)
    where
        F: Fn(&MeshType),
    {
        self.meshes.foreach_loaded(function)
    }

    #[inline]
    fn meshes_memory_pool_size(max_meshes: u32, frames_in_flight: u32) -> u64 {
        1024u64
            * (
                // 1KiB for good measure
                1024u64 +
            // 128KiB for each bottom level AS
            (128u64 * (frames_in_flight as u64)) +
            // 32KiB for each mesh
            (64u64 * (max_meshes as u64))
            )
    }

    #[inline]
    pub(crate) fn wait_load_nonblock(&mut self) -> RenderingResult<usize> {
        self.meshes.wait_load_nonblock()
    }

    #[inline]
    pub(crate) fn wait_load_blocking(&mut self) -> RenderingResult<usize> {
        self.meshes.wait_load_blocking()
    }

    pub fn new(
        queue_family: Arc<QueueFamily>,
        max_meshes: u32,
        frames_in_flight: u32,
        debug_name: String,
    ) -> RenderingResult<Self> {
        let device = queue_family.get_parent_device();

        let descriptor_pool = DescriptorPool::new(
            device.clone(),
            DescriptorPoolConcreteDescriptor::new(
                DescriptorPoolSizesConcreteDescriptor::new(
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    Some(DescriptorPoolSizesAcceletarionStructureKHR::new(
                        frames_in_flight,
                    )),
                ),
                frames_in_flight,
            ),
            Some("mesh_manager.descriptor_pool"),
        )?;

        let queue = Queue::new(queue_family.clone(), Some("mesh_manager.queue"))?;

        let rt_blocksize = 1024u64;
        let raytracing_allocator = Arc::new(DefaultAllocator::with_blocksize(
            rt_blocksize,
            Self::meshes_memory_pool_size(max_meshes, frames_in_flight) / rt_blocksize,
        ));

        let raytracing_memory_heap = MemoryHeap::new(
            device.clone(),
            ConcreteMemoryHeapDescriptor::new(
                required_memory_type(),
                raytracing_allocator.total_size(),
            ),
            Default::default(),
        )?;

        let memory_pool = MemoryPool::new(
            raytracing_memory_heap,
            raytracing_allocator,
            MemoryPoolFeatures::from([MemoryPoolFeature::DeviceAddressable].as_slice()),
        )?;

        let meshes = LoadableResourcesCollection::new(
            queue_family,
            max_meshes,
            String::from("mesh_manager"),
        )?;

        Ok(Self {
            debug_name,

            queue,
            _descriptor_pool: descriptor_pool,

            memory_pool,

            meshes,
        })
    }

    pub fn create_vertex_buffer(
        &self,
        triangles_topology: BottomLevelVerticesTopologyDecl,
        usage: BufferUsage,
        sharing: Option<&[std::sync::Weak<QueueFamily>]>,
    ) -> RenderingResult<Arc<BottomLevelAccelerationStructureVertexBuffer>> {
        let vertex_buffer = BottomLevelAccelerationStructureVertexBuffer::new(
            self.memory_pool.clone(),
            triangles_topology,
            usage,
            sharing,
            &Option::None,
        )?;

        Ok(vertex_buffer)
    }

    pub fn create_index_buffer(
        &self,
        usage: BufferUsage,
        triangles_decl: BottomLevelTrianglesGroupDecl,
        sharing: Option<&[std::sync::Weak<QueueFamily>]>,
    ) -> RenderingResult<Arc<BottomLevelAccelerationStructureIndexBuffer>> {
        let vertex_buffer = BottomLevelAccelerationStructureIndexBuffer::new(
            self.memory_pool.clone(),
            usage,
            triangles_decl,
            sharing,
            &Option::None,
        )?;

        Ok(vertex_buffer)
    }

    fn create_blas(
        memory_pool: Arc<MemoryPool>,
        vertex_buffer: Arc<BottomLevelAccelerationStructureVertexBuffer>,
        index_buffer: Arc<BottomLevelAccelerationStructureIndexBuffer>,
        debug_name: String,
    ) -> RenderingResult<Arc<BottomLevelAccelerationStructure>> {
        let blas = BottomLevelAccelerationStructure::new(
            memory_pool,
            AllowedBuildingDevice::DeviceOnly,
            BottomLevelVertexBufferSpecifier::Preallocated(vertex_buffer),
            BottomLevelIndexBufferSpecifier::Preallocated(index_buffer),
            BufferUsage::default(),
            None,
            Some(debug_name.as_str()),
        )?;

        Ok(blas)
    }

    fn setup_load_blas_operation(
        recorder: &mut CommandBufferRecorder,
        blas: Arc<BottomLevelAccelerationStructure>,
        queue: Arc<Queue>,
    ) -> RenderingResult<()> {
        let queue_family = queue.get_parent_queue_family();

        // create the blas
        let max_primitives_count = blas.max_primitives_count();
        recorder.build_blas(blas, 0, max_primitives_count, 0, 0);

        Ok(())
    }

    pub fn load(
        &mut self,
        vertex_buffer: Arc<BottomLevelAccelerationStructureVertexBuffer>,
        index_buffer: Arc<BottomLevelAccelerationStructureIndexBuffer>,
    ) -> RenderingResult<u32> {
        let queue = self.queue.clone();
        let queue_family = queue.get_parent_queue_family();
        let Some(blas_index) = self.meshes.load(
            queue.clone(),
            || {
                Self::create_blas(
                    self.memory_pool.clone(),
                    vertex_buffer,
                    index_buffer,
                    String::from("texture_empty_image"),
                )
            },
            |recorder, blas| Self::setup_load_blas_operation(recorder, blas, queue.clone()),
        )?
        else {
            return Err(RenderingError::ResourceError(
                super::ResourceError::NoMeshSlotAvailable,
            ));
        };

        Ok(blas_index)
    }
}

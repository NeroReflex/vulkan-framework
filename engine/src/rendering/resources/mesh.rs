use std::{
    ops::DerefMut,
    sync::{Arc, Mutex},
};

use vulkan_framework::{
    acceleration_structure::{
        bottom_level::{
            BottomLevelAccelerationStructure, BottomLevelAccelerationStructureIndexBuffer,
            BottomLevelAccelerationStructureTransformBuffer,
            BottomLevelAccelerationStructureVertexBuffer, BottomLevelTrianglesGroupDecl,
            BottomLevelVerticesTopologyDecl, IDENTITY_MATRIX,
        }, AllowedBuildingDevice
    }, ash::vk::TransformMatrixKHR, binding_tables::required_memory_type, buffer::{AllocatedBuffer, Buffer, BufferSubresourceRange, BufferTrait, BufferUsage}, command_buffer::CommandBufferRecorder, descriptor_pool::{
        DescriptorPool, DescriptorPoolConcreteDescriptor,
        DescriptorPoolSizesAcceletarionStructureKHR, DescriptorPoolSizesConcreteDescriptor,
    }, device::{Device, DeviceOwned}, memory_allocator::{DefaultAllocator, MemoryAllocator}, memory_barriers::{BufferMemoryBarrier, MemoryAccessAs}, memory_heap::{ConcreteMemoryHeapDescriptor, MemoryHeap, MemoryHostVisibility, MemoryType}, memory_management::{MemoryManagerTrait, UnallocatedResource}, memory_pool::{MemoryMap, MemoryPool, MemoryPoolBacked, MemoryPoolFeature, MemoryPoolFeatures}, pipeline_stage::PipelineStage, prelude::VulkanResult, queue::Queue, queue_family::{QueueFamily, QueueFamilyOwned}
};

use crate::rendering::{
    MAX_MESHES, RenderingError, RenderingResult, resources::collection::LoadableResourcesCollection,
};

type MeshType = Arc<BottomLevelAccelerationStructure>;

pub struct MeshManager {
    debug_name: String,

    queue: Arc<Queue>,

    memory_manager: Arc<Mutex<dyn MemoryManagerTrait>>,

    _descriptor_pool: Arc<DescriptorPool>,

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
        memory_manager: Arc<Mutex<dyn MemoryManagerTrait>>,
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
            Some(format!("{debug_name}.mesh_manager.descriptor_pool").as_str()),
        )?;

        let queue = Queue::new(
            queue_family.clone(),
            Some(format!("{debug_name}.mesh_manager.queue").as_str()),
        )?;

        let rt_blocksize = 1024u64;
        let raytracing_allocator = Arc::new(DefaultAllocator::with_blocksize(
            rt_blocksize,
            Self::meshes_memory_pool_size(MAX_MESHES, frames_in_flight) / rt_blocksize,
        ));

        let meshes = LoadableResourcesCollection::new(
            queue_family,
            MAX_MESHES,
            format!("{debug_name}.mesh_manager"),
        )?;

        Ok(Self {
            debug_name,

            queue,
            _descriptor_pool: descriptor_pool,

            memory_manager,

            meshes,
        })
    }

    pub fn create_vertex_buffer(
        &self,
        memory_manager: &mut dyn MemoryManagerTrait,
        vertices_topology: BottomLevelVerticesTopologyDecl,
        usage: BufferUsage,
        sharing: Option<&[std::sync::Weak<QueueFamily>]>,
    ) -> RenderingResult<(BottomLevelVerticesTopologyDecl, Arc<AllocatedBuffer>)> {
        let descriptor =
            BottomLevelAccelerationStructureVertexBuffer::template(&vertices_topology, usage);

        let backing_buffer = Buffer::new(
            memory_manager.get_parent_device(),
            descriptor,
            sharing,
            Some(format!("{}", self.debug_name).as_str()),
        )?;

        let buffer = memory_manager.allocate_resources(
            &MemoryType::DeviceLocal(Some(MemoryHostVisibility::visible(false))),
            &MemoryPoolFeatures::new(true),
            vec![UnallocatedResource::Buffer(backing_buffer)],
            &[],
        )?;

        Ok((vertices_topology, buffer[0].buffer()))
    }

    pub fn create_index_buffer(
        &self,
        memory_manager: &mut dyn MemoryManagerTrait,
        triangles_decl: BottomLevelTrianglesGroupDecl,
        usage: BufferUsage,
        sharing: Option<&[std::sync::Weak<QueueFamily>]>,
    ) -> RenderingResult<BottomLevelAccelerationStructureIndexBuffer> {
        let descriptor =
            BottomLevelAccelerationStructureIndexBuffer::template(&triangles_decl, usage);
        let backing_buffer = Buffer::new(
            memory_manager.get_parent_device(),
            descriptor,
            sharing,
            Some(format!("{}", self.debug_name).as_str()),
        )?;

        let buffer = memory_manager.allocate_resources(
            &MemoryType::DeviceLocal(Some(MemoryHostVisibility::visible(false))),
            &MemoryPoolFeatures::new(true),
            vec![UnallocatedResource::Buffer(backing_buffer)],
            &[],
        )?;

        Ok(BottomLevelAccelerationStructureIndexBuffer::new(
            triangles_decl,
            buffer[0].buffer(),
        )?)
    }

    pub fn create_transform_buffer(
        &self,
        memory_manager: &mut dyn MemoryManagerTrait,
        usage: BufferUsage,
        sharing: Option<&[std::sync::Weak<QueueFamily>]>,
    ) -> RenderingResult<BottomLevelAccelerationStructureTransformBuffer> {
        let descriptor = BottomLevelAccelerationStructureTransformBuffer::template(usage);
        let backing_buffer = Buffer::new(
            memory_manager.get_parent_device(),
            descriptor,
            sharing,
            Some(format!("{}", self.debug_name).as_str()),
        )?;

        let buffer = memory_manager.allocate_resources(
            &MemoryType::DeviceLocal(Some(MemoryHostVisibility::visible(false))),
            &MemoryPoolFeatures::new(true),
            vec![UnallocatedResource::Buffer(backing_buffer)],
            &[],
        )?;

        let transform_buffer = BottomLevelAccelerationStructureTransformBuffer::new(
            buffer[0].buffer(),
        )?;

        let mut mem_map = MemoryMap::new(transform_buffer.buffer().get_backing_memory_pool())?;
        let transform = mem_map.as_mut_slice_with_size::<TransformMatrixKHR>(transform_buffer.buffer().clone() as Arc<dyn MemoryPoolBacked>, transform_buffer.buffer().size())?;
        assert_eq!(transform.len(), 1);
        transform[0] = IDENTITY_MATRIX;

        Ok(transform_buffer)
    }

    fn create_blas(
        memory_manager: &mut dyn MemoryManagerTrait,
        vertex_buffer: BottomLevelAccelerationStructureVertexBuffer,
        index_buffer: BottomLevelAccelerationStructureIndexBuffer,
        transform_buffer: BottomLevelAccelerationStructureTransformBuffer,
        debug_name: String,
    ) -> RenderingResult<Arc<BottomLevelAccelerationStructure>> {
        let blas = BottomLevelAccelerationStructure::new(
            memory_manager,
            AllowedBuildingDevice::DeviceOnly,
            vertex_buffer,
            index_buffer,
            transform_buffer,
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
        // create the blas
        let max_primitives_count = blas.max_primitives_count();
        recorder.build_blas(blas, 0, max_primitives_count, 0, 0);

        Ok(())
    }

    pub fn load(
        &mut self,
        vertex_buffer: BottomLevelAccelerationStructureVertexBuffer,
        index_buffer: BottomLevelAccelerationStructureIndexBuffer,
        transform_buffer: BottomLevelAccelerationStructureTransformBuffer,
    ) -> RenderingResult<u32> {
        let queue = self.queue.clone();
        let queue_family = queue.get_parent_queue_family();

        let vertex_buffer_raw = vertex_buffer.buffer();
        let vertex_buffer_size = vertex_buffer.buffer().size();

        let index_buffer_raw = vertex_buffer.buffer();
        let index_buffer_size = vertex_buffer.buffer().size();

        let mut allocator = self.memory_manager.lock().unwrap();

        let debug_name = self.debug_name.clone();
        let Some(blas_index) = self.meshes.load(
            queue.clone(),
            move |index| {
                let debug_name = format!("{debug_name}.blas[{index}]");
                Self::create_blas(
                    allocator.deref_mut(),
                    vertex_buffer,
                    index_buffer,
                    transform_buffer,
                    debug_name,
                )
            },
            |recorder, _, blas| {
                // Wait for the host to finsh transfer to vertex buffer and index buffer
                recorder.buffer_barriers(
                    [
                        BufferMemoryBarrier::new(
                            [PipelineStage::Host].as_slice().into(),
                            [MemoryAccessAs::MemoryWrite].as_slice().into(),
                            [PipelineStage::Transfer].as_slice().into(),
                            [MemoryAccessAs::MemoryRead].as_slice().into(),
                            BufferSubresourceRange::new(
                                vertex_buffer_raw,
                                0u64,
                                vertex_buffer_size,
                            ),
                            queue_family.clone(),
                            queue_family.clone(),
                        ),
                        BufferMemoryBarrier::new(
                            [PipelineStage::Host].as_slice().into(),
                            [MemoryAccessAs::MemoryWrite].as_slice().into(),
                            [PipelineStage::Transfer].as_slice().into(),
                            [MemoryAccessAs::MemoryRead].as_slice().into(),
                            BufferSubresourceRange::new(index_buffer_raw, 0u64, index_buffer_size),
                            queue_family.clone(),
                            queue_family.clone(),
                        ),
                    ]
                    .as_slice(),
                );

                Self::setup_load_blas_operation(recorder, blas, queue.clone())
            },
        )?
        else {
            return Err(RenderingError::ResourceError(
                super::ResourceError::NoMeshSlotAvailable,
            ));
        };

        Ok(blas_index)
    }
}

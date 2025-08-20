use std::{
    ops::DerefMut,
    sync::{Arc, Mutex},
};

use vulkan_framework::{
    acceleration_structure::{
        AllowedBuildingDevice,
        bottom_level::{
            BottomLevelAccelerationStructure, BottomLevelAccelerationStructureIndexBuffer,
            BottomLevelAccelerationStructureTransformBuffer,
            BottomLevelAccelerationStructureVertexBuffer, BottomLevelTrianglesGroupDecl,
            BottomLevelVerticesTopologyDecl,
        },
    },
    ash::vk::TransformMatrixKHR,
    buffer::{AllocatedBuffer, Buffer, BufferSubresourceRange, BufferTrait, BufferUsage},
    descriptor_pool::{
        DescriptorPool, DescriptorPoolConcreteDescriptor,
        DescriptorPoolSizesAcceletarionStructureKHR, DescriptorPoolSizesConcreteDescriptor,
    },
    device::DeviceOwned,
    memory_barriers::{BufferMemoryBarrier, MemoryAccessAs},
    memory_heap::{MemoryHeapOwned, MemoryType},
    memory_management::{
        MemoryManagementTagSize, MemoryManagementTags, MemoryManagerTrait, UnallocatedResource,
    },
    memory_pool::{MemoryMap, MemoryPoolBacked, MemoryPoolFeatures},
    pipeline_stage::{PipelineStage, PipelineStageAccelerationStructureKHR},
    queue::Queue,
    queue_family::{QueueFamily, QueueFamilyOwned},
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
            Some(self.debug_name.to_string().as_str()),
        )?;

        let buffer = memory_manager.allocate_resources(
            &MemoryType::device_local_and_host_visible(),
            &MemoryPoolFeatures::new(true),
            vec![UnallocatedResource::Buffer(backing_buffer)],
            Self::allocation_tags(),
        )?;

        Ok((vertices_topology, buffer[0].buffer()))
    }

    pub fn allocation_tags() -> MemoryManagementTags {
        MemoryManagementTags::default()
            .with_name("mesh".to_string())
            .with_size(MemoryManagementTagSize::MediumLarge)
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
            Some(self.debug_name.to_string().as_str()),
        )?;

        let buffer = memory_manager.allocate_resources(
            &MemoryType::device_local_and_host_visible(),
            &MemoryPoolFeatures::new(true),
            vec![UnallocatedResource::Buffer(backing_buffer)],
            Self::allocation_tags(),
        )?;

        Ok(BottomLevelAccelerationStructureIndexBuffer::new(
            triangles_decl,
            buffer[0].buffer(),
        )?)
    }

    pub fn create_transform_buffer(
        &self,
        memory_manager: &mut dyn MemoryManagerTrait,
        transform_data: &vulkan_framework::ash::vk::TransformMatrixKHR,
        usage: BufferUsage,
        sharing: Option<&[std::sync::Weak<QueueFamily>]>,
    ) -> RenderingResult<BottomLevelAccelerationStructureTransformBuffer> {
        let descriptor = BottomLevelAccelerationStructureTransformBuffer::template(usage);
        let backing_buffer = Buffer::new(
            memory_manager.get_parent_device(),
            descriptor,
            sharing,
            Some(self.debug_name.to_string().as_str()),
        )?;

        let buffer = memory_manager.allocate_resources(
            &MemoryType::device_local_and_host_visible(),
            &MemoryPoolFeatures::new(true),
            vec![UnallocatedResource::Buffer(backing_buffer)],
            Self::allocation_tags(),
        )?;

        let transform_buffer =
            BottomLevelAccelerationStructureTransformBuffer::new(buffer[0].buffer())?;

        // if host is memory mappable load the identity matrix
        if transform_buffer
            .buffer()
            .get_backing_memory_pool()
            .get_parent_memory_heap()
            .is_host_mappable()
        {
            let mem_map = MemoryMap::new(transform_buffer.buffer().get_backing_memory_pool())?;
            let mut range = mem_map.range::<TransformMatrixKHR>(
                transform_buffer.buffer().clone() as Arc<dyn MemoryPoolBacked>,
            )?;
            let transform = range.as_mut_slice();
            assert_eq!(transform.len(), 1);
            transform[0] = *transform_data;
        }

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
            Self::allocation_tags(),
            None,
            Some(debug_name.as_str()),
        )?;

        Ok(blas)
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
        let vertex_buffer_size = vertex_buffer_raw.size();

        let index_buffer_raw = index_buffer.buffer();
        let index_buffer_size = index_buffer_raw.size();

        let transform_buffer_raw = transform_buffer.buffer();
        let transform_buffer_size = transform_buffer_raw.size();

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
                recorder.pipeline_barriers([
                    BufferMemoryBarrier::new(
                        [PipelineStage::Host].as_slice().into(),
                        [MemoryAccessAs::MemoryWrite].as_slice().into(),
                        [PipelineStage::AccelerationStructureKHR(
                            PipelineStageAccelerationStructureKHR::Build,
                        )]
                        .as_slice()
                        .into(),
                        [
                            MemoryAccessAs::AccelerationStructureRead,
                            MemoryAccessAs::MemoryRead,
                        ]
                        .as_slice()
                        .into(),
                        BufferSubresourceRange::new(
                            vertex_buffer_raw.clone(),
                            0u64,
                            vertex_buffer_size,
                        ),
                        queue_family.clone(),
                        queue_family.clone(),
                    )
                    .into(),
                    BufferMemoryBarrier::new(
                        [PipelineStage::Host].as_slice().into(),
                        [MemoryAccessAs::MemoryWrite].as_slice().into(),
                        [PipelineStage::AccelerationStructureKHR(
                            PipelineStageAccelerationStructureKHR::Build,
                        )]
                        .as_slice()
                        .into(),
                        [
                            MemoryAccessAs::AccelerationStructureRead,
                            MemoryAccessAs::MemoryRead,
                        ]
                        .as_slice()
                        .into(),
                        BufferSubresourceRange::new(
                            index_buffer_raw.clone(),
                            0u64,
                            index_buffer_size,
                        ),
                        queue_family.clone(),
                        queue_family.clone(),
                    )
                    .into(),
                    BufferMemoryBarrier::new(
                        [PipelineStage::Host].as_slice().into(),
                        [MemoryAccessAs::MemoryWrite].as_slice().into(),
                        [PipelineStage::AccelerationStructureKHR(
                            PipelineStageAccelerationStructureKHR::Build,
                        )]
                        .as_slice()
                        .into(),
                        [
                            MemoryAccessAs::AccelerationStructureRead,
                            MemoryAccessAs::MemoryRead,
                        ]
                        .as_slice()
                        .into(),
                        BufferSubresourceRange::new(
                            transform_buffer_raw.clone(),
                            0u64,
                            transform_buffer_size,
                        ),
                        queue_family.clone(),
                        queue_family.clone(),
                    )
                    .into(),
                ]);

                let primitives_count = blas.max_primitives_count();
                recorder.build_blas(blas.clone(), 0, primitives_count, 0, 0);

                recorder.pipeline_barriers([
                    BufferMemoryBarrier::new(
                        [PipelineStage::AccelerationStructureKHR(
                            PipelineStageAccelerationStructureKHR::Build,
                        )]
                        .as_slice()
                        .into(),
                        [MemoryAccessAs::AccelerationStructureWrite]
                            .as_slice()
                            .into(),
                        [PipelineStage::BottomOfPipe].as_slice().into(),
                        [MemoryAccessAs::MemoryRead].as_slice().into(),
                        BufferSubresourceRange::new(blas.buffer(), 0u64, blas.buffer_size()),
                        queue_family.clone(),
                        queue_family.clone(),
                    )
                    .into(),
                    BufferMemoryBarrier::new(
                        [PipelineStage::AccelerationStructureKHR(
                            PipelineStageAccelerationStructureKHR::Build,
                        )]
                        .as_slice()
                        .into(),
                        [MemoryAccessAs::AccelerationStructureRead]
                            .as_slice()
                            .into(),
                        [PipelineStage::BottomOfPipe].as_slice().into(),
                        [MemoryAccessAs::MemoryRead].as_slice().into(),
                        BufferSubresourceRange::new(
                            vertex_buffer_raw.clone(),
                            0u64,
                            vertex_buffer_size,
                        ),
                        queue_family.clone(),
                        queue_family.clone(),
                    )
                    .into(),
                    BufferMemoryBarrier::new(
                        [PipelineStage::AccelerationStructureKHR(
                            PipelineStageAccelerationStructureKHR::Build,
                        )]
                        .as_slice()
                        .into(),
                        [MemoryAccessAs::AccelerationStructureRead]
                            .as_slice()
                            .into(),
                        [PipelineStage::BottomOfPipe].as_slice().into(),
                        [MemoryAccessAs::MemoryRead].as_slice().into(),
                        BufferSubresourceRange::new(
                            index_buffer_raw.clone(),
                            0u64,
                            index_buffer_size,
                        ),
                        queue_family.clone(),
                        queue_family.clone(),
                    )
                    .into(),
                    BufferMemoryBarrier::new(
                        [PipelineStage::AccelerationStructureKHR(
                            PipelineStageAccelerationStructureKHR::Build,
                        )]
                        .as_slice()
                        .into(),
                        [MemoryAccessAs::AccelerationStructureRead]
                            .as_slice()
                            .into(),
                        [PipelineStage::BottomOfPipe].as_slice().into(),
                        [MemoryAccessAs::MemoryRead].as_slice().into(),
                        BufferSubresourceRange::new(
                            transform_buffer_raw.clone(),
                            0u64,
                            transform_buffer_size,
                        ),
                        queue_family.clone(),
                        queue_family.clone(),
                    )
                    .into(),
                ]);

                Ok(())
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

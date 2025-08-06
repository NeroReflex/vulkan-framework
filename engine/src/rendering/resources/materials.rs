use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

use vulkan_framework::{
    buffer::{
        AllocatedBuffer, Buffer, BufferSubresourceRange, BufferTrait, BufferUsage, BufferUseAs,
        ConcreteBufferDescriptor,
    },
    descriptor_pool::{
        DescriptorPool, DescriptorPoolConcreteDescriptor, DescriptorPoolSizesConcreteDescriptor,
    },
    descriptor_set::DescriptorSet,
    descriptor_set_layout::DescriptorSetLayout,
    device::DeviceOwned,
    memory_allocator::{DefaultAllocator, MemoryAllocator},
    memory_barriers::{BufferMemoryBarrier, MemoryAccessAs},
    memory_heap::{ConcreteMemoryHeapDescriptor, MemoryHeap, MemoryRequirements, MemoryType},
    memory_pool::{MemoryPool, MemoryPoolFeatures},
    memory_requiring::AllocationRequiring,
    pipeline_stage::PipelineStage,
    queue::Queue,
    queue_family::{QueueFamily, QueueFamilyOwned},
    shader_layout_binding::{BindingDescriptor, BindingType, NativeBindingType},
    shader_stage_access::ShaderStagesAccess,
};

use crate::rendering::{
    MAX_FRAMES_IN_FLIGHT_NO_MALLOC, MAX_MATERIALS, MAX_TEXTURES, RenderingError, RenderingResult,
    resources::{SIZEOF_MATERIAL_DEFINITION, collection::LoadableResourcesCollection},
};

type DescriptorSetsType =
    smallvec::SmallVec<[(AtomicU64, Arc<DescriptorSet>); MAX_FRAMES_IN_FLIGHT_NO_MALLOC]>;

type MeshToMaterialFramesInFlightType =
    smallvec::SmallVec<[Arc<AllocatedBuffer>; MAX_FRAMES_IN_FLIGHT_NO_MALLOC]>;

type MaterialsFrameInFlightType =
    smallvec::SmallVec<[Arc<AllocatedBuffer>; MAX_FRAMES_IN_FLIGHT_NO_MALLOC]>;

pub struct MaterialManager {
    debug_name: String,

    queue: Arc<Queue>,

    descriptor_set_layout: Arc<DescriptorSetLayout>,
    descriptor_sets: DescriptorSetsType,

    material_buffers: MaterialsFrameInFlightType,
    mesh_to_material_map: MeshToMaterialFramesInFlightType,

    // reflects the current status of GPU-loaded materials:
    // this has to be copied in uniform buffers being bound
    // to the descriptor set
    current_materials_buffer: Arc<AllocatedBuffer>,
    //mesh_to_material_map_collection: Arc<AllocatedBuffer>,
    materials: LoadableResourcesCollection<u32>,
}

impl MaterialManager {
    #[inline]
    pub fn descriptor_set_layout(&self) -> Arc<DescriptorSetLayout> {
        self.descriptor_set_layout.clone()
    }

    #[inline]
    pub fn is_loaded(&self, index: usize) -> bool {
        self.materials.fetch_loaded(index).is_some()
    }

    #[inline]
    pub(crate) fn wait_load_nonblock(&mut self) -> RenderingResult<usize> {
        self.materials.wait_load_nonblock()
    }

    #[inline]
    pub(crate) fn wait_load_blocking(&mut self) -> RenderingResult<usize> {
        self.materials.wait_load_blocking()
    }

    #[inline]
    pub fn material_descriptor_set(&self, current_frame: usize) -> Arc<DescriptorSet> {
        let (descriptor_set_status, descriptor_set) =
            self.descriptor_sets.get(current_frame).unwrap();

        let current_status = self.materials.status();

        // Check if the descriptor set needs to be updated:
        // new materials have been loaded in GPU memory since the last time it was used
        let mix_status = descriptor_set_status.fetch_min(current_status, Ordering::SeqCst);
        if mix_status != current_status {
            // Update the descriptor set with available resources
            descriptor_set
                .bind_resources(|binder| {
                    binder
                        .bind_uniform_buffer(
                            0,
                            [
                                (
                                    self.material_buffers[current_frame].clone()
                                        as Arc<dyn BufferTrait>,
                                    None,
                                    None,
                                ),
                                (
                                    self.mesh_to_material_map[current_frame].clone()
                                        as Arc<dyn BufferTrait>,
                                    None,
                                    None,
                                ),
                            ]
                            .as_slice(),
                        )
                        .unwrap();
                })
                .unwrap();

            descriptor_set_status
                .compare_exchange(
                    mix_status,
                    current_status,
                    Ordering::SeqCst,
                    Ordering::SeqCst,
                )
                .unwrap();

            println!("Updated materials descriptor set {current_frame}");
        }

        descriptor_set.clone()
    }

    pub fn new(
        queue_family: Arc<QueueFamily>,
        frames_in_flight: u32,
        debug_name: String,
    ) -> RenderingResult<Self> {
        let device = queue_family.get_parent_device();

        let queue = Queue::new(queue_family.clone(), Some("texture_manager.queue"))?;

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
                    2u32 * frames_in_flight,
                    0,
                    None,
                ),
                frames_in_flight,
            ),
            Some(format!("{debug_name}.descriptor_pool").as_str()),
        )?;

        let descriptor_set_layout = DescriptorSetLayout::new(
            device.clone(),
            [
                BindingDescriptor::new(
                    ShaderStagesAccess::graphics(),
                    BindingType::Native(NativeBindingType::UniformBuffer),
                    0,
                    2,
                ),
                //BindingDescriptor::new(
                //    ShaderStagesAccess::graphics(),
                //    BindingType::Native(NativeBindingType::UniformBuffer),
                //    0,
                //    1,
                //),
            ]
            .as_slice(),
        )?;

        let mut descriptor_sets: DescriptorSetsType = smallvec::smallvec![];
        for _ in 0..frames_in_flight as usize {
            let descriptor_set =
                DescriptorSet::new(descriptor_pool.clone(), descriptor_set_layout.clone())?;

            descriptor_sets.push((AtomicU64::new(0), descriptor_set));
        }

        let mut materials_unallocated_buffer: smallvec::SmallVec<
            [Buffer; MAX_FRAMES_IN_FLIGHT_NO_MALLOC],
        > = smallvec::smallvec![];
        for index in 0..(frames_in_flight as usize) {
            materials_unallocated_buffer.push(Buffer::new(
                device.clone(),
                ConcreteBufferDescriptor::new(
                    BufferUsage::from(
                        [BufferUseAs::TransferDst, BufferUseAs::UniformBuffer].as_slice(),
                    ),
                    (frames_in_flight as u64) * (MAX_MATERIALS as u64),
                ),
                None,
                Some(format!("{debug_name}.materials_buffer[{index}]").as_str()),
            )?);
        }

        let current_materials_buffer = Buffer::new(
            device.clone(),
            ConcreteBufferDescriptor::new(
                BufferUsage::from([BufferUseAs::TransferDst].as_slice()),
                (SIZEOF_MATERIAL_DEFINITION as u64) * (MAX_MATERIALS as u64),
            ),
            None,
            Some(format!("{debug_name}.current_materials_buffer").as_str()),
        )?;

        let materials_size = (SIZEOF_MATERIAL_DEFINITION as u64)
            * (MAX_MATERIALS as u64)
            * (frames_in_flight as u64);
        let block_size = 128u64;
        let materials_allocator = DefaultAllocator::with_blocksize(
            block_size,
            ((materials_size / block_size) + 1u64) + (frames_in_flight as u64),
        );

        let memory_heap = MemoryHeap::new(
            device.clone(),
            ConcreteMemoryHeapDescriptor::new(
                MemoryType::DeviceLocal(None),
                materials_allocator.total_size(),
            ),
            MemoryRequirements::try_from(
                materials_unallocated_buffer
                    .iter()
                    .map(|m| m as &dyn AllocationRequiring)
                    .chain(
                        [&current_materials_buffer]
                            .into_iter()
                            .map(|m| m as &dyn AllocationRequiring),
                    )
                    .collect::<smallvec::SmallVec<[_; 8]>>()
                    .as_slice(),
            )?,
        )?;

        let memory_pool = MemoryPool::new(
            memory_heap,
            Arc::new(materials_allocator),
            MemoryPoolFeatures::from([].as_slice()),
        )?;

        let current_materials_buffer =
            AllocatedBuffer::new(memory_pool.clone(), current_materials_buffer)?;

        let materials = LoadableResourcesCollection::new(
            queue_family,
            MAX_TEXTURES,
            String::from("materials_manager"),
        )?;

        let mut material_buffers: MaterialsFrameInFlightType = smallvec::smallvec![];
        for buffer in materials_unallocated_buffer.into_iter() {
            let material_buffer = AllocatedBuffer::new(memory_pool.clone(), buffer)?;
            material_buffers.push(material_buffer);
        }

        let mut mesh_to_material_map: MeshToMaterialFramesInFlightType =
            MeshToMaterialFramesInFlightType::with_capacity(frames_in_flight as usize);
        for index in 0..frames_in_flight {
            let mesh_material_buffer = Buffer::new(
                device.clone(),
                ConcreteBufferDescriptor::new(
                    BufferUsage::from(
                        [BufferUseAs::TransferDst, BufferUseAs::UniformBuffer].as_slice(),
                    ),
                    (MAX_MATERIALS as u64) * 4u64,
                ),
                None,
                Some(format!("{debug_name}.mesh_to_material_map[{index}]").as_str()),
            )?;

            mesh_to_material_map.push(AllocatedBuffer::new(
                memory_pool.clone(),
                mesh_material_buffer,
            )?);
        }

        Ok(Self {
            debug_name,

            queue,

            descriptor_set_layout,
            descriptor_sets,

            materials,
            material_buffers,

            current_materials_buffer,
            mesh_to_material_map,
        })
    }

    pub fn load(&mut self, material_data: Arc<dyn BufferTrait>) -> RenderingResult<u32> {
        let queue = self.queue.clone();
        let Some(texture_index) = self.materials.load(
            queue.clone(),
            |_| Ok(0u32),
            |recorder, index, _| {
                let queue_family = queue.get_parent_queue_family();

                // Wait for host to finish writing the material into the buffer
                let copy_size = SIZEOF_MATERIAL_DEFINITION as u64;
                let copy_offset = (index as u64) * copy_size;
                recorder.buffer_barriers(
                    [BufferMemoryBarrier::new(
                        [PipelineStage::Host].as_slice().into(),
                        [MemoryAccessAs::MemoryWrite].as_slice().into(),
                        [PipelineStage::Transfer].as_slice().into(),
                        [MemoryAccessAs::MemoryRead].as_slice().into(),
                        BufferSubresourceRange::new(material_data.clone(), 0u64, copy_size),
                        queue_family.clone(),
                        queue_family.clone(),
                    )]
                    .as_slice(),
                );

                // Copy the buffer in the correct slot
                recorder.copy_buffer(
                    material_data,
                    self.current_materials_buffer.clone(),
                    [(0u64, copy_offset, copy_size)].as_slice(),
                );

                // Place a memory barrier to wait for GPU to finish cloning the buffer
                recorder.buffer_barriers(
                    [BufferMemoryBarrier::new(
                        [PipelineStage::Transfer].as_slice().into(),
                        [MemoryAccessAs::MemoryWrite].as_slice().into(),
                        [PipelineStage::AllCommands].as_slice().into(),
                        [MemoryAccessAs::MemoryRead].as_slice().into(),
                        BufferSubresourceRange::new(
                            self.current_materials_buffer.clone(),
                            copy_offset,
                            copy_size,
                        ),
                        queue_family.clone(),
                        queue_family.clone(),
                    )]
                    .as_slice(),
                );

                Ok(())
            },
        )?
        else {
            return Err(RenderingError::ResourceError(
                super::ResourceError::NoTextureSlotAvailable,
            ));
        };

        Ok(texture_index)
    }

    #[inline]
    pub fn remove(&mut self, index: u32) -> RenderingResult<()> {
        self.materials.remove(index)
    }
}

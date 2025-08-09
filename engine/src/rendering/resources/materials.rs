use std::sync::{
    Arc, Mutex,
    atomic::{AtomicU64, Ordering},
};

use vulkan_framework::{
    buffer::{
        AllocatedBuffer, Buffer, BufferSubresourceRange, BufferTrait, BufferUsage, BufferUseAs,
        ConcreteBufferDescriptor,
    },
    command_buffer::CommandBufferRecorder,
    descriptor_pool::{
        DescriptorPool, DescriptorPoolConcreteDescriptor, DescriptorPoolSizesConcreteDescriptor,
    },
    descriptor_set::DescriptorSet,
    descriptor_set_layout::DescriptorSetLayout,
    device::DeviceOwned,
    memory_barriers::{BufferMemoryBarrier, MemoryAccessAs},
    memory_heap::{MemoryHostVisibility, MemoryType},
    memory_management::{
        MemoryManagementTagSize, MemoryManagementTags, MemoryManagerTrait, UnallocatedResource,
    },
    memory_pool::MemoryPoolFeatures,
    pipeline_stage::PipelineStage,
    queue::Queue,
    queue_family::{QueueFamily, QueueFamilyOwned},
    shader_layout_binding::{BindingDescriptor, BindingType, NativeBindingType},
    shader_stage_access::ShaderStagesAccess,
};

use crate::rendering::{
    MAX_FRAMES_IN_FLIGHT_NO_MALLOC, MAX_MATERIALS, MAX_TEXTURES, RenderingError, RenderingResult,
    resources::{
        SIZEOF_MATERIAL_DEFINITION, collection::LoadableResourcesCollection, object::MaterialGPU,
    },
};

type DescriptorSetsType =
    smallvec::SmallVec<[(AtomicU64, Arc<DescriptorSet>); MAX_FRAMES_IN_FLIGHT_NO_MALLOC]>;

type MeshToMaterialFramesInFlightType =
    smallvec::SmallVec<[Arc<AllocatedBuffer>; MAX_FRAMES_IN_FLIGHT_NO_MALLOC]>;

type MaterialsFrameInFlightType =
    smallvec::SmallVec<[Arc<AllocatedBuffer>; MAX_FRAMES_IN_FLIGHT_NO_MALLOC]>;

pub struct MaterialManager {
    memory_manager: Arc<Mutex<dyn MemoryManagerTrait>>,

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
                        .bind_storage_buffers(
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
        memory_manager: Arc<Mutex<dyn MemoryManagerTrait>>,
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
                    2u32 * frames_in_flight,
                    0,
                    0,
                    None,
                ),
                frames_in_flight,
            ),
            Some(format!("{debug_name}.descriptor_pool").as_str()),
        )?;

        let descriptor_set_layout = DescriptorSetLayout::new(
            device.clone(),
            [BindingDescriptor::new(
                ShaderStagesAccess::graphics(),
                BindingType::Native(NativeBindingType::StorageBuffer),
                0,
                2,
            )]
            .as_slice(),
        )?;

        let mut descriptor_sets: DescriptorSetsType = smallvec::smallvec![];
        for _ in 0..frames_in_flight as usize {
            let descriptor_set =
                DescriptorSet::new(descriptor_pool.clone(), descriptor_set_layout.clone())?;

            descriptor_sets.push((AtomicU64::new(0), descriptor_set));
        }

        let mut materials_unallocated_buffer = vec![];
        for index in 0..(frames_in_flight as usize) {
            materials_unallocated_buffer.push(UnallocatedResource::Buffer(Buffer::new(
                device.clone(),
                ConcreteBufferDescriptor::new(
                    BufferUsage::from(
                        [BufferUseAs::TransferDst, BufferUseAs::StorageBuffer].as_slice(),
                    ),
                    (SIZEOF_MATERIAL_DEFINITION as u64) * (MAX_MATERIALS as u64),
                ),
                None,
                Some(format!("{debug_name}.materials_buffer[{index}]").as_str()),
            )?));
        }

        let mut mesh_material_unallocated_buffers = vec![];
        for index in 0..(frames_in_flight as usize) {
            mesh_material_unallocated_buffers.push(UnallocatedResource::Buffer(Buffer::new(
                device.clone(),
                ConcreteBufferDescriptor::new(
                    BufferUsage::from(
                        [BufferUseAs::TransferDst, BufferUseAs::StorageBuffer].as_slice(),
                    ),
                    (MAX_MATERIALS as u64) * 4u64,
                ),
                None,
                Some(format!("{debug_name}.mesh_to_material_map[{index}]").as_str()),
            )?));
        }

        let current_materials_unallocated_buffer = Buffer::new(
            device.clone(),
            ConcreteBufferDescriptor::new(
                BufferUsage::from([BufferUseAs::TransferDst, BufferUseAs::TransferSrc].as_slice()),
                (SIZEOF_MATERIAL_DEFINITION as u64) * (MAX_MATERIALS as u64),
            ),
            None,
            Some(format!("{debug_name}.current_materials_buffer").as_str()),
        )?;

        let mut mem_manager = memory_manager.lock().unwrap();

        let current_materials_allocated_buffer = mem_manager.allocate_resources(
            &MemoryType::DeviceLocal(Some(MemoryHostVisibility::MemoryHostHidden)),
            &MemoryPoolFeatures::new(false),
            vec![current_materials_unallocated_buffer.into()],
            MemoryManagementTags::default()
                .with_name("material_buffers".to_string())
                .with_size(MemoryManagementTagSize::MediumSmall),
        )?;

        let materials_allocated_buffers = mem_manager.allocate_resources(
            &MemoryType::DeviceLocal(Some(MemoryHostVisibility::MemoryHostHidden)),
            &MemoryPoolFeatures::new(false),
            materials_unallocated_buffer,
            MemoryManagementTags::default()
                .with_name("material_buffers".to_string())
                .with_size(MemoryManagementTagSize::MediumSmall),
        )?;

        let mesh_material_allocated_buffers = mem_manager.allocate_resources(
            &MemoryType::DeviceLocal(Some(MemoryHostVisibility::MemoryHostHidden)),
            &MemoryPoolFeatures::new(false),
            mesh_material_unallocated_buffers,
            MemoryManagementTags::default()
                .with_name("material_buffers".to_string())
                .with_size(MemoryManagementTagSize::MediumSmall),
        )?;

        let material_buffers = materials_allocated_buffers
            .into_iter()
            .map(|allocated| allocated.buffer())
            .collect::<MaterialsFrameInFlightType>();

        let mesh_to_material_map = mesh_material_allocated_buffers
            .into_iter()
            .map(|allocated| allocated.buffer())
            .collect::<MeshToMaterialFramesInFlightType>();

        let current_materials_buffer = current_materials_allocated_buffer[0].buffer();

        drop(mem_manager);

        let materials = LoadableResourcesCollection::new(
            queue_family,
            MAX_TEXTURES,
            String::from("materials_manager"),
        )?;

        Ok(Self {
            memory_manager,

            queue,

            descriptor_set_layout,
            descriptor_sets,

            materials,
            material_buffers,

            current_materials_buffer,
            mesh_to_material_map,
        })
    }

    pub fn load(&mut self, material_data: MaterialGPU) -> RenderingResult<u32> {
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
                        [].as_slice().into(),
                        [].as_slice().into(),
                        [PipelineStage::Transfer].as_slice().into(),
                        [MemoryAccessAs::TransferRead].as_slice().into(),
                        BufferSubresourceRange::new(self.current_materials_buffer.clone(), copy_offset, copy_size),
                        queue_family.clone(),
                        queue_family.clone(),
                    )]
                    .as_slice(),
                );

                // Copy the buffer in the correct slot
                recorder.update_buffer(
                    self.current_materials_buffer.clone(),
                    copy_offset,
                    [material_data].as_slice()
                );

                // Place a memory barrier to wait for GPU to finish cloning the buffer
                recorder.buffer_barriers(
                    [BufferMemoryBarrier::new(
                        [PipelineStage::Transfer].as_slice().into(),
                        [MemoryAccessAs::TransferWrite].as_slice().into(),
                        [PipelineStage::AllCommands].as_slice().into(),
                        [MemoryAccessAs::MemoryRead, MemoryAccessAs::ShaderRead].as_slice().into(),
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

    /// Update buffers, that are already bound to the materials descriptor set for rendering
    pub fn update_buffers(
        &self,
        recorder: &mut CommandBufferRecorder,
        current_frame: usize,
        queue_family: Arc<QueueFamily>,
    ) {
        assert_eq!(
            self.current_materials_buffer.size(),
            self.material_buffers[current_frame].size()
        );

        recorder.copy_buffer(
            self.current_materials_buffer.clone(),
            self.material_buffers[current_frame].clone(),
            [(0u64, 0u64, self.current_materials_buffer.size())].as_slice(),
        );

        recorder.buffer_barriers(
            [BufferMemoryBarrier::new(
                [PipelineStage::Transfer].as_slice().into(),
                [MemoryAccessAs::TransferWrite].as_slice().into(),
                [PipelineStage::AllCommands].as_slice().into(),
                [MemoryAccessAs::MemoryRead].as_slice().into(),
                BufferSubresourceRange::new(
                    self.material_buffers[current_frame].clone(),
                    0u64,
                    self.material_buffers[current_frame].size(),
                ),
                queue_family.clone(),
                queue_family.clone(),
            )]
            .as_slice(),
        );
    }
}

use std::sync::{Arc, Mutex};

use vulkan_framework::{
    buffer::{
        AllocatedBuffer, Buffer, BufferSubresourceRange, BufferTrait, BufferUseAs,
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
    memory_heap::MemoryType,
    memory_management::{MemoryManagementTagSize, MemoryManagementTags, MemoryManagerTrait},
    memory_pool::MemoryPoolFeatures,
    pipeline_stage::{PipelineStage, PipelineStageRayTracingPipelineKHR},
    queue::Queue,
    queue_family::{QueueFamily, QueueFamilyOwned},
    shader_layout_binding::{BindingDescriptor, BindingType, NativeBindingType},
    shader_stage_access::{ShaderStageAccessIn, ShaderStageAccessInRayTracingKHR},
};

use crate::rendering::{
    MAX_FRAMES_IN_FLIGHT_NO_MALLOC, MAX_MATERIALS, MAX_MESHES, RenderingError, RenderingResult,
    resources::{ResourceError, SIZEOF_MATERIAL_DEFINITION, object::MaterialGPU},
};

type DescriptorSetsType = smallvec::SmallVec<[Arc<DescriptorSet>; MAX_FRAMES_IN_FLIGHT_NO_MALLOC]>;
type FrameBuffers = smallvec::SmallVec<[Arc<AllocatedBuffer>; MAX_FRAMES_IN_FLIGHT_NO_MALLOC]>;

pub struct MaterialManager {
    descriptor_set_layout: Arc<DescriptorSetLayout>,
    descriptor_sets: DescriptorSetsType,
    material_buffers: FrameBuffers,
    mesh_to_material_map: FrameBuffers,
    // vkCmdUpdateBuffer captures these small tables during recording. No shared
    // GPU upload buffer can then race another frame's transfer read.
    materials: Vec<Option<MaterialGPU>>,
}

impl MaterialManager {
    pub fn descriptor_set_layout(&self) -> Arc<DescriptorSetLayout> {
        self.descriptor_set_layout.clone()
    }

    pub fn is_loaded(&self, index: usize) -> bool {
        self.materials.get(index).is_some_and(Option::is_some)
    }

    pub(crate) fn wait_load_nonblock(&mut self) -> RenderingResult<usize> {
        Ok(0)
    }

    pub(crate) fn wait_load_blocking(&mut self) -> RenderingResult<usize> {
        Ok(0)
    }

    pub fn material_descriptor_set(&self, current_frame: usize) -> Arc<DescriptorSet> {
        // Buffer identities never change: bind once, never update a pending set.
        self.descriptor_sets[current_frame].clone()
    }

    pub fn new(
        queue: Arc<Queue>,
        memory_manager: Arc<Mutex<dyn MemoryManagerTrait>>,
        frames_in_flight: u32,
        debug_name: String,
    ) -> RenderingResult<Self> {
        let device = queue.get_parent_queue_family().get_parent_device();
        let pool = DescriptorPool::new(
            device.clone(),
            DescriptorPoolConcreteDescriptor::new(
                DescriptorPoolSizesConcreteDescriptor::new(
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    2 * frames_in_flight,
                    0,
                    0,
                    None,
                ),
                frames_in_flight,
            ),
            Some(format!("{debug_name}.descriptor_pool").as_str()),
        )?;
        let stages = [
            ShaderStageAccessIn::Fragment,
            ShaderStageAccessIn::RayTracing(ShaderStageAccessInRayTracingKHR::ClosestHit),
        ]
        .as_slice()
        .into();
        let descriptor_set_layout = DescriptorSetLayout::new(
            device.clone(),
            &[
                BindingDescriptor::new(
                    stages,
                    BindingType::Native(NativeBindingType::StorageBuffer),
                    0,
                    1,
                ),
                BindingDescriptor::new(
                    stages,
                    BindingType::Native(NativeBindingType::StorageBuffer),
                    1,
                    1,
                ),
            ],
        )?;
        let mut buffers = Vec::new();
        for frame in 0..frames_in_flight {
            for (name, size) in [
                (
                    "materials_buffer",
                    (SIZEOF_MATERIAL_DEFINITION as u64) * (MAX_MATERIALS as u64),
                ),
                ("mesh_to_material_map", (MAX_MESHES as u64) * 4),
            ] {
                buffers.push(
                    Buffer::new(
                        device.clone(),
                        ConcreteBufferDescriptor::new(
                            [BufferUseAs::TransferDst, BufferUseAs::StorageBuffer]
                                .as_slice()
                                .into(),
                            size,
                        ),
                        None,
                        Some(format!("{debug_name}.{name}[{frame}]").as_str()),
                    )?
                    .into(),
                );
            }
        }
        let allocated = memory_manager.lock().unwrap().allocate_resources(
            &MemoryType::device_local(),
            &MemoryPoolFeatures::new(false),
            buffers,
            MemoryManagementTags::default()
                .with_name("material_buffers".to_string())
                .with_size(MemoryManagementTagSize::MediumSmall),
        )?;
        let mut material_buffers = FrameBuffers::new();
        let mut mesh_to_material_map = FrameBuffers::new();
        let mut descriptor_sets = DescriptorSetsType::new();
        for frame in 0..frames_in_flight as usize {
            let material_buffer = allocated[frame * 2].buffer();
            let mapping_buffer = allocated[frame * 2 + 1].buffer();
            let set = DescriptorSet::new(pool.clone(), descriptor_set_layout.clone())?;
            set.bind_resources(|binder| {
                binder
                    .bind_storage_buffers(
                        0,
                        &[
                            (material_buffer.clone() as Arc<dyn BufferTrait>, None, None),
                            (mapping_buffer.clone() as Arc<dyn BufferTrait>, None, None),
                        ],
                    )
                    .unwrap();
            })?;
            material_buffers.push(material_buffer);
            mesh_to_material_map.push(mapping_buffer);
            descriptor_sets.push(set);
        }
        Ok(Self {
            descriptor_set_layout,
            descriptor_sets,
            material_buffers,
            mesh_to_material_map,
            materials: vec![None; MAX_MATERIALS as usize],
        })
    }

    pub fn load(&mut self, material: MaterialGPU) -> RenderingResult<u32> {
        let index = self
            .materials
            .iter()
            .position(Option::is_none)
            .ok_or_else(|| RenderingError::ResourceError(ResourceError::NoMaterialSlotAvailable))?;
        self.materials[index] = Some(material);
        Ok(index as u32)
    }

    pub fn remove(&mut self, index: u32) -> RenderingResult<()> {
        let slot = self.materials.get_mut(index as usize).ok_or_else(|| {
            RenderingError::ResourceError(ResourceError::ResourceIndexOutOfRange(index as usize))
        })?;
        *slot = None;
        Ok(())
    }

    fn snapshot(materials: &[Option<MaterialGPU>]) -> Vec<MaterialGPU> {
        materials
            .iter()
            .map(|material| material.unwrap_or_default())
            .collect()
    }

    /// The caller must have waited for the previous submission using this frame slot.
    pub fn update_buffers(
        &self,
        recorder: &mut CommandBufferRecorder,
        current_frame: usize,
        mesh_to_material: &[u32],
        queue_family: Arc<QueueFamily>,
    ) {
        let materials = Self::snapshot(&self.materials);
        assert_eq!(mesh_to_material.len(), MAX_MESHES as usize);
        assert_eq!(
            std::mem::size_of_val(materials.as_slice()) as u64,
            self.material_buffers[current_frame].size()
        );
        recorder.update_buffer(self.material_buffers[current_frame].clone(), 0, &materials);
        recorder.update_buffer(
            self.mesh_to_material_map[current_frame].clone(),
            0,
            mesh_to_material,
        );
        for buffer in [
            &self.material_buffers[current_frame],
            &self.mesh_to_material_map[current_frame],
        ] {
            recorder.pipeline_barriers([BufferMemoryBarrier::new(
                [PipelineStage::Transfer].as_slice().into(),
                [MemoryAccessAs::TransferWrite].as_slice().into(),
                [
                    PipelineStage::FragmentShader,
                    PipelineStage::RayTracingPipelineKHR(
                        PipelineStageRayTracingPipelineKHR::RayTracingShader,
                    ),
                ]
                .as_slice()
                .into(),
                [MemoryAccessAs::ShaderRead].as_slice().into(),
                BufferSubresourceRange::new(buffer.clone(), 0, buffer.size()),
                queue_family.clone(),
                queue_family.clone(),
            )
            .into()]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshots_preserve_slots_and_zero_missing_materials() {
        let material = MaterialGPU {
            diffuse_texture_index: 9,
            normal_texture_index: 8,
            reflection_texture_index: 7,
            displacement_texture_index: 6,
        };
        let snapshot = MaterialManager::snapshot(&[None, Some(material), None]);
        let diffuse: Vec<_> = snapshot
            .iter()
            .map(|entry| entry.diffuse_texture_index)
            .collect();
        assert_eq!(diffuse, [0, 9, 0]);
        assert_eq!(
            std::mem::size_of_val(snapshot.as_slice()),
            3 * SIZEOF_MATERIAL_DEFINITION
        );
    }

    #[test]
    fn recorded_snapshot_is_independent_of_later_edits() {
        let mut materials = vec![Some(MaterialGPU {
            diffuse_texture_index: 9,
            ..MaterialGPU::default()
        })];
        let recorded = MaterialManager::snapshot(&materials);
        materials[0] = None;
        let before = recorded[0].diffuse_texture_index;
        let after = MaterialManager::snapshot(&materials)[0].diffuse_texture_index;
        assert_eq!(before, 9);
        assert_eq!(after, 0);
    }
}

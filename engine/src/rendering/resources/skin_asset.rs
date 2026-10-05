use std::collections::HashMap;
use std::io::Read;
use std::sync::Arc;

use serde::Deserialize;
use vulkan_framework::{
    buffer::{
        AllocatedBuffer, Buffer, BufferTrait, BufferUsage, BufferUseAs, ConcreteBufferDescriptor,
    },
    descriptor_pool::{
        DescriptorPool, DescriptorPoolConcreteDescriptor, DescriptorPoolSizesConcreteDescriptor,
    },
    descriptor_set::DescriptorSet,
    descriptor_set_layout::DescriptorSetLayout,
    device::DeviceOwned,
    memory_management::{MemoryManagementTags, MemoryManagerTrait},
    memory_pool::{MemoryMap, MemoryPoolBacked, MemoryPoolFeatures},
    shader_layout_binding::{BindingDescriptor, BindingType, NativeBindingType},
    shader_stage_access::ShaderStageAccessIn,
};

use crate::rendering::{RenderingError, RenderingResult};
use crate::rendering::resources::ResourceError;
use crate::scene::{
    AnimationClipInfo, AnimationGpuChannel, AnimationPlayState, ArmatureGpuElement,
    SkeletonGpuElement, MAX_ANIMATION_CHANNELS,
};

#[derive(Debug, Deserialize)]
struct ObjectMeta {
    bone_count: u32,
    armature_node_count: u32,
    animations: Vec<MetaAnimation>,
}

#[derive(Debug, Deserialize)]
struct MetaAnimation {
    name: String,
    duration_ticks: f64,
    ticks_per_second: f64,
    channel_count: u32,
}

#[derive(Default)]
pub struct SkinnedTarLoader {
    skeleton_original: Option<Vec<u8>>,
    skeleton_armature: Option<Vec<u8>>,
    skinned_vertices: Option<Vec<u8>>,
    channel_blobs: HashMap<String, Vec<u8>>,
    meta: Option<ObjectMeta>,
}

impl SkinnedTarLoader {
    pub fn ingest(&mut self, path: &str, reader: &mut impl Read, size: u64) -> RenderingResult<()> {
        let mut data = vec![0u8; size as usize];
        reader.read_exact(&mut data)?;
        match path.strip_prefix("./").unwrap_or(path) {
            "skeleton/original" => self.skeleton_original = Some(data),
            "skeleton/armature" => self.skeleton_armature = Some(data),
            "skinned_vertex_buffer" => self.skinned_vertices = Some(data),
            "meta" => {
                self.meta = Some(serde_json::from_slice(&data).map_err(|_| {
                    RenderingError::ResourceError(ResourceError::InvalidObjectFormat)
                })?);
            }
            path if path.starts_with("animations/") && path.ends_with("/channels") => {
                let name = path
                    .strip_prefix("animations/")
                    .and_then(|rest| rest.strip_suffix("/channels"))
                    .unwrap_or("clip")
                    .to_string();
                self.channel_blobs.insert(name, data);
            }
            _ => {}
        }
        Ok(())
    }

    pub fn is_skinned_archive(&self) -> bool {
        self.skeleton_original.is_some() && self.skeleton_armature.is_some()
    }

    pub fn finish(
        self,
        device: Arc<vulkan_framework::device::Device>,
        memory_manager: &mut dyn MemoryManagerTrait,
    ) -> RenderingResult<SkinnedAsset> {
        let original_bytes = self
            .skeleton_original
            .ok_or(RenderingError::ResourceError(ResourceError::InvalidObjectFormat))?;
        let armature_bytes = self
            .skeleton_armature
            .ok_or(RenderingError::ResourceError(ResourceError::InvalidObjectFormat))?;
        let meta = self
            .meta
            .ok_or(RenderingError::ResourceError(ResourceError::InvalidObjectFormat))?;

        let bone_count = meta.bone_count;
        let per_frame_bytes = (bone_count as u64) * 64;

        let original = upload_storage(device.clone(), memory_manager, &original_bytes, "skin_original")?;
        let armature = upload_storage(device.clone(), memory_manager, &armature_bytes, "skin_armature")?;
        let per_frame = upload_storage(
            device.clone(),
            memory_manager,
            &vec![0u8; per_frame_bytes as usize],
            "skin_per_frame",
        )?;

        let skinned_vertices = if let Some(bytes) = self.skinned_vertices {
            upload_storage(device.clone(), memory_manager, &bytes, "skin_vertices")?
        } else {
            return Err(RenderingError::ResourceError(ResourceError::MissingVertexBuffer));
        };

        let vertex_count = (skinned_vertices.size() / 64) as u32;
        let deformed_vertices = upload_storage(
            device.clone(),
            memory_manager,
            &vec![0u8; skinned_vertices.size() as usize],
            "skin_deformed",
        )?;

        let mut clips = HashMap::new();
        let mut channel_buffers = HashMap::new();
        for anim in meta.animations {
            let Some(blob) = self.channel_blobs.get(&anim.name) else {
                continue;
            };
            let buffer = upload_storage(
                device.clone(),
                memory_manager,
                blob,
                &format!("anim_{}", anim.name),
            )?;
            clips.insert(
                anim.name.clone(),
                AnimationClipInfo {
                    name: anim.name.clone(),
                    duration_ticks: anim.duration_ticks,
                    ticks_per_second: anim.ticks_per_second,
                    channel_count: anim.channel_count,
                },
            );
            channel_buffers.insert(anim.name, buffer);
        }

        let channels_layout = skin_channels_layout(device.clone())?;
        let bind_pose_layout = skin_bind_pose_layout(device.clone())?;
        let palette_layout = skin_palette_layout(device.clone())?;
        let deform_layout = skin_deform_layout(device.clone())?;

        let pool = DescriptorPool::new(
            device.clone(),
            DescriptorPoolConcreteDescriptor::new(
                DescriptorPoolSizesConcreteDescriptor::new(0, 0, 0, 0, 0, 0, 8, 0, 0, None),
                8,
            ),
            Some("skin_descriptor_pool"),
        )?;

        let channels_set = DescriptorSet::new(pool.clone(), channels_layout.clone())?;
        let bind_pose_set = DescriptorSet::new(pool.clone(), bind_pose_layout.clone())?;
        let palette_set = DescriptorSet::new(pool.clone(), palette_layout.clone())?;
        let deform_set = DescriptorSet::new(pool.clone(), deform_layout.clone())?;

        bind_pose_set.bind_resources(|writer| {
            writer.bind_storage_buffers(
                0,
                [(
                    original.clone() as Arc<dyn BufferTrait>,
                    None,
                    None,
                )]
                .as_slice(),
            )?;
            writer.bind_storage_buffers(
                1,
                [(
                    per_frame.clone() as Arc<dyn BufferTrait>,
                    None,
                    None,
                )]
                .as_slice(),
            )?;
            writer.bind_storage_buffers(
                2,
                [(
                    armature.clone() as Arc<dyn BufferTrait>,
                    None,
                    None,
                )]
                .as_slice(),
            )?;
            Ok::<(), vulkan_framework::prelude::VulkanError>(())
        })
        .map_err(|err| RenderingError::Unknown(err.to_string()))?;

        palette_set.bind_resources(|writer| {
            writer.bind_storage_buffers(
                0,
                [(
                    per_frame.clone() as Arc<dyn BufferTrait>,
                    None,
                    None,
                )]
                .as_slice(),
            )?;
            Ok::<(), vulkan_framework::prelude::VulkanError>(())
        })
        .map_err(|err| RenderingError::Unknown(err.to_string()))?;

        deform_set.bind_resources(|writer| {
            writer.bind_storage_buffers(
                1,
                [(
                    skinned_vertices.clone() as Arc<dyn BufferTrait>,
                    None,
                    None,
                )]
                .as_slice(),
            )?;
            writer.bind_storage_buffers(
                2,
                [(
                    deformed_vertices.clone() as Arc<dyn BufferTrait>,
                    None,
                    None,
                )]
                .as_slice(),
            )?;
            Ok::<(), vulkan_framework::prelude::VulkanError>(())
        })
        .map_err(|err| RenderingError::Unknown(err.to_string()))?;

        Ok(SkinnedAsset {
            bone_count,
            vertex_count,
            clips,
            channel_buffers,
            original_skeleton: original,
            per_frame_skeleton: per_frame,
            armature,
            skinned_vertices,
            deformed_vertices,
            play_state: AnimationPlayState::default(),
            descriptor_pool: pool,
            channels_layout,
            bind_pose_layout,
            palette_layout,
            deform_layout,
            bind_pose_set,
            palette_set,
            deform_set,
            channels_sets: HashMap::new(),
        })
    }
}

pub struct SkinnedAsset {
    pub bone_count: u32,
    pub vertex_count: u32,
    pub clips: HashMap<String, AnimationClipInfo>,
    channel_buffers: HashMap<String, Arc<AllocatedBuffer>>,
    original_skeleton: Arc<AllocatedBuffer>,
    per_frame_skeleton: Arc<AllocatedBuffer>,
    armature: Arc<AllocatedBuffer>,
    skinned_vertices: Arc<AllocatedBuffer>,
    deformed_vertices: Arc<AllocatedBuffer>,
    pub play_state: AnimationPlayState,
    descriptor_pool: Arc<DescriptorPool>,
    channels_layout: Arc<DescriptorSetLayout>,
    bind_pose_layout: Arc<DescriptorSetLayout>,
    palette_layout: Arc<DescriptorSetLayout>,
    deform_layout: Arc<DescriptorSetLayout>,
    bind_pose_set: Arc<DescriptorSet>,
    palette_set: Arc<DescriptorSet>,
    deform_set: Arc<DescriptorSet>,
    channels_sets: HashMap<String, Arc<DescriptorSet>>,
}

impl SkinnedAsset {
    pub fn clip_names(&self) -> Vec<String> {
        self.clips.keys().cloned().collect()
    }

    pub fn channels_set(&mut self, clip: &str) -> RenderingResult<Arc<DescriptorSet>> {
        if let Some(set) = self.channels_sets.get(clip) {
            return Ok(set.clone());
        }
        let Some(channels) = self.channel_buffers.get(clip) else {
            return Err(RenderingError::ResourceError(ResourceError::InvalidObjectFormat));
        };
        let set = DescriptorSet::new(self.descriptor_pool.clone(), self.channels_layout.clone())?;
        set.bind_resources(|writer| {
            writer.bind_storage_buffers(
                0,
                [(
                    self.original_skeleton.clone() as Arc<dyn BufferTrait>,
                    None,
                    None,
                )]
                .as_slice(),
            )?;
            writer.bind_storage_buffers(
                1,
                [(
                    self.per_frame_skeleton.clone() as Arc<dyn BufferTrait>,
                    None,
                    None,
                )]
                .as_slice(),
            )?;
            writer.bind_storage_buffers(
                2,
                [(self.armature.clone() as Arc<dyn BufferTrait>, None, None)].as_slice(),
            )?;
            writer.bind_storage_buffers(
                3,
                [(channels.clone() as Arc<dyn BufferTrait>, None, None)].as_slice(),
            )?;
            Ok::<(), vulkan_framework::prelude::VulkanError>(())
        })
        .map_err(|err| RenderingError::Unknown(err.to_string()))?;
        self.channels_sets.insert(clip.to_string(), set.clone());
        Ok(set)
    }

    pub fn bind_pose_set(&self) -> Arc<DescriptorSet> {
        self.bind_pose_set.clone()
    }

    pub fn palette_set(&self) -> Arc<DescriptorSet> {
        self.palette_set.clone()
    }

    pub fn deform_set(&self) -> Arc<DescriptorSet> {
        self.deform_set.clone()
    }

    pub fn per_frame(&self) -> Arc<AllocatedBuffer> {
        self.per_frame_skeleton.clone()
    }

    pub fn bind_palette(&self, bone_palette: Arc<AllocatedBuffer>) -> RenderingResult<()> {
        let palette = bone_palette.clone();
        let deform_palette = bone_palette;
        self.palette_set.bind_resources(|writer| {
            writer.bind_storage_buffers(
                0,
                [(
                    self.per_frame_skeleton.clone() as Arc<dyn BufferTrait>,
                    None,
                    None,
                )]
                .as_slice(),
            )?;
            writer.bind_storage_buffers(
                1,
                [(palette as Arc<dyn BufferTrait>, None, None)].as_slice(),
            )?;
            Ok::<(), vulkan_framework::prelude::VulkanError>(())
        })
        .map_err(|err| RenderingError::Unknown(err.to_string()))?;
        self.deform_set.bind_resources(|writer| {
            writer.bind_storage_buffers(
                0,
                [(deform_palette as Arc<dyn BufferTrait>, None, None)].as_slice(),
            )?;
            writer.bind_storage_buffers(
                1,
                [(
                    self.skinned_vertices.clone() as Arc<dyn BufferTrait>,
                    None,
                    None,
                )]
                .as_slice(),
            )?;
            writer.bind_storage_buffers(
                2,
                [(
                    self.deformed_vertices.clone() as Arc<dyn BufferTrait>,
                    None,
                    None,
                )]
                .as_slice(),
            )?;
            Ok::<(), vulkan_framework::prelude::VulkanError>(())
        })
        .map_err(|err| RenderingError::Unknown(err.to_string()))?;
        Ok(())
    }

    pub fn deformed_vertices(&self) -> Arc<AllocatedBuffer> {
        self.deformed_vertices.clone()
    }
}

pub struct SkinDispatchView<'a> {
    pub bone_count: u32,
    pub vertex_count: u32,
    pub animated: bool,
    pub time_ticks: f32,
    pub channel_count: u32,
    pub channels_set: Option<Arc<DescriptorSet>>,
    pub bind_pose_set: Arc<DescriptorSet>,
    pub palette_set: Arc<DescriptorSet>,
    pub deform_set: Arc<DescriptorSet>,
    pub per_frame: Arc<AllocatedBuffer>,
    pub _marker: std::marker::PhantomData<&'a ()>,
}

impl SkinnedAsset {
    pub fn dispatch_view(&mut self) -> SkinDispatchView<'_> {
        let animated = self.play_state.clip.is_some();
        let time_ticks = self
            .play_state
            .time_in_ticks(&self.clips)
            .unwrap_or(0.0);
        let channel_count = self.play_state.active_channel_count(&self.clips);
        let clip_name = self.play_state.clip.clone();
        let channels_set = clip_name
            .as_deref()
            .and_then(|name| self.channels_set(name).ok());
        SkinDispatchView {
            bone_count: self.bone_count,
            vertex_count: self.vertex_count,
            animated,
            time_ticks,
            channel_count,
            channels_set,
            bind_pose_set: self.bind_pose_set.clone(),
            palette_set: self.palette_set.clone(),
            deform_set: self.deform_set.clone(),
            per_frame: self.per_frame_skeleton.clone(),
            _marker: std::marker::PhantomData,
        }
    }
}

fn upload_storage(
    device: Arc<vulkan_framework::device::Device>,
    memory_manager: &mut dyn MemoryManagerTrait,
    bytes: &[u8],
    name: &str,
) -> RenderingResult<Arc<AllocatedBuffer>> {
    let buffer = Buffer::new(
        device,
        ConcreteBufferDescriptor::new(
            BufferUsage::from([BufferUseAs::StorageBuffer].as_slice()),
            bytes.len() as u64,
        ),
        None,
        Some(name),
    )?;
    let allocated = memory_manager.allocate_resources(
        &vulkan_framework::memory_heap::MemoryType::host_visible_and_coherent(),
        &MemoryPoolFeatures::default(),
        vec![buffer.into()],
        MemoryManagementTags::default().with_name(name.to_string()),
    )?;
    let buffer = allocated[0].buffer();
    let map = MemoryMap::new(buffer.get_backing_memory_pool())?;
    let mut range = map.range::<u8>(buffer.clone() as Arc<dyn MemoryPoolBacked>)?;
    range.as_mut_slice().copy_from_slice(bytes);
    Ok(buffer)
}

fn skin_channels_layout(
    device: Arc<vulkan_framework::device::Device>,
) -> RenderingResult<Arc<DescriptorSetLayout>> {
    let compute = [ShaderStageAccessIn::Compute].as_slice().into();
    DescriptorSetLayout::new(
        device,
        [
            BindingDescriptor::new(compute, BindingType::Native(NativeBindingType::StorageBuffer), 0, 1),
            BindingDescriptor::new(compute, BindingType::Native(NativeBindingType::StorageBuffer), 1, 1),
            BindingDescriptor::new(compute, BindingType::Native(NativeBindingType::StorageBuffer), 2, 1),
            BindingDescriptor::new(compute, BindingType::Native(NativeBindingType::StorageBuffer), 3, 1),
        ]
        .as_slice(),
    )
    .map_err(|err| RenderingError::Unknown(err.to_string()))
}

fn skin_bind_pose_layout(
    device: Arc<vulkan_framework::device::Device>,
) -> RenderingResult<Arc<DescriptorSetLayout>> {
    let compute = [ShaderStageAccessIn::Compute].as_slice().into();
    DescriptorSetLayout::new(
        device,
        [
            BindingDescriptor::new(compute, BindingType::Native(NativeBindingType::StorageBuffer), 0, 1),
            BindingDescriptor::new(compute, BindingType::Native(NativeBindingType::StorageBuffer), 1, 1),
            BindingDescriptor::new(compute, BindingType::Native(NativeBindingType::StorageBuffer), 2, 1),
        ]
        .as_slice(),
    )
    .map_err(|err| RenderingError::Unknown(err.to_string()))
}

fn skin_palette_layout(
    device: Arc<vulkan_framework::device::Device>,
) -> RenderingResult<Arc<DescriptorSetLayout>> {
    let compute = [ShaderStageAccessIn::Compute].as_slice().into();
    DescriptorSetLayout::new(
        device,
        [
            BindingDescriptor::new(compute, BindingType::Native(NativeBindingType::StorageBuffer), 0, 1),
            BindingDescriptor::new(compute, BindingType::Native(NativeBindingType::StorageBuffer), 1, 1),
        ]
        .as_slice(),
    )
    .map_err(|err| RenderingError::Unknown(err.to_string()))
}

fn skin_deform_layout(
    device: Arc<vulkan_framework::device::Device>,
) -> RenderingResult<Arc<DescriptorSetLayout>> {
    let compute = [ShaderStageAccessIn::Compute].as_slice().into();
    DescriptorSetLayout::new(
        device,
        [
            BindingDescriptor::new(compute, BindingType::Native(NativeBindingType::StorageBuffer), 0, 1),
            BindingDescriptor::new(compute, BindingType::Native(NativeBindingType::StorageBuffer), 1, 1),
            BindingDescriptor::new(compute, BindingType::Native(NativeBindingType::StorageBuffer), 2, 1),
        ]
        .as_slice(),
    )
    .map_err(|err| RenderingError::Unknown(err.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn channel_struct_size_is_stable() {
        assert_eq!(std::mem::size_of::<SkeletonGpuElement>(), 80);
        assert_eq!(std::mem::size_of::<ArmatureGpuElement>(), 80);
        assert!(std::mem::size_of::<AnimationGpuChannel>() > 3000);
    }
}

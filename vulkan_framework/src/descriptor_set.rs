use std::sync::Arc;

use ash::vk::Handle;

#[cfg(feature = "better_mutex")]
use parking_lot::{const_mutex, Mutex};

#[cfg(not(feature = "better_mutex"))]
use std::sync::Mutex;

use crate::{
    acceleration_structure::TopLevelAccelerationStructure,
    buffer::BufferTrait,
    descriptor_pool::{DescriptorPool, DescriptorPoolOwned},
    descriptor_set_layout::{DescriptorSetLayout, DescriptorSetLayoutDependant},
    device::DeviceOwned,
    image::ImageLayout,
    image_view::ImageView,
    prelude::{FrameworkError, VulkanError, VulkanResult},
    sampler::Sampler,
};

pub struct DescriptorSetWriter<'a> {
    //device: Arc<Device>,
    descriptor_set: &'a DescriptorSet,
    pub(crate) acceleration_structures: smallvec::SmallVec<
        [(
            u32,
            smallvec::SmallVec<[Arc<TopLevelAccelerationStructure>; 4]>,
        ); 8],
    >,
    pub(crate) combined_image_sampler: smallvec::SmallVec<
        [(
            u32,
            smallvec::SmallVec<[(ImageLayout, Arc<ImageView>, Arc<Sampler>); 4]>,
        ); 8],
    >,
    pub(crate) sampled_images:
        smallvec::SmallVec<[(u32, smallvec::SmallVec<[(ImageLayout, Arc<ImageView>); 4]>); 8]>,
    pub(crate) storage_images:
        smallvec::SmallVec<[(u32, smallvec::SmallVec<[(ImageLayout, Arc<ImageView>); 4]>); 8]>,
    pub(crate) uniform_buffers: smallvec::SmallVec<
        [(
            u32,
            smallvec::SmallVec<[(Arc<dyn BufferTrait>, u64, u64); 4]>,
        ); 8],
    >,
    pub(crate) storage_buffers: smallvec::SmallVec<
        [(
            u32,
            smallvec::SmallVec<[(Arc<dyn BufferTrait>, u64, u64); 4]>,
        ); 8],
    >,

    used_resources: smallvec::SmallVec<[Option<DescriptorSetBoundResource>; 32]>,
}

impl<'a> DescriptorSetWriter<'a> {
    pub(crate) fn new(descriptor_set: &'a DescriptorSet, size: u32) -> Self {
        Self {
            descriptor_set,
            acceleration_structures: smallvec::smallvec![],
            combined_image_sampler: smallvec::smallvec![],
            sampled_images: smallvec::smallvec![],
            storage_images: smallvec::smallvec![],
            uniform_buffers: smallvec::smallvec![],
            storage_buffers: smallvec::smallvec![],
            used_resources: (0..size).map(|_idx| Option::default()).collect(),
        }
    }

    pub(crate) fn used_resources(
        &self,
    ) -> &smallvec::SmallVec<[Option<DescriptorSetBoundResource>; 32]> {
        &self.used_resources
    }

    pub fn bind_tlas(
        &mut self,
        first_layout_id: u32,
        tlas_collection: &[Arc<TopLevelAccelerationStructure>],
    ) -> VulkanResult<()> {
        if first_layout_id as usize + tlas_collection.len() > self.used_resources.len() {
            return Err(VulkanError::Framework(
                FrameworkError::DescriptorSetBindingOutOfRange,
            ));
        }

        for (idx, el) in tlas_collection.iter().enumerate() {
            if self.used_resources[first_layout_id as usize + idx]
                .replace(DescriptorSetBoundResource::TLAS(el.clone()))
                .is_some()
            {
                return Err(VulkanError::Framework(
                    FrameworkError::DescriptorSetBindingDuplicated,
                ));
            }
        }

        self.acceleration_structures
            .push((first_layout_id, tlas_collection.iter().cloned().collect()));
        Ok(())
    }

    pub fn bind_combined_images_samplers(
        &mut self,
        first_layout_id: u32,
        images: &[(ImageLayout, Arc<ImageView>, Arc<Sampler>)],
    ) -> VulkanResult<()> {
        if first_layout_id as usize + images.len() > self.used_resources.len() {
            return Err(VulkanError::Framework(
                FrameworkError::DescriptorSetBindingOutOfRange,
            ));
        }

        for (idx, (_, image_view, sampler)) in images.iter().enumerate() {
            if self.used_resources[first_layout_id as usize + idx]
                .replace(DescriptorSetBoundResource::CombinedImageViewSampler((
                    image_view.clone(),
                    sampler.clone(),
                )))
                .is_some()
            {
                return Err(VulkanError::Framework(
                    FrameworkError::DescriptorSetBindingDuplicated,
                ));
            }
        }

        self.combined_image_sampler
            .push((first_layout_id, images.iter().cloned().collect()));
        Ok(())
    }

    pub fn bind_sampled_images(
        &mut self,
        first_layout_id: u32,
        images: &[(ImageLayout, Arc<ImageView>)],
    ) -> VulkanResult<()> {
        if first_layout_id as usize + images.len() > self.used_resources.len() {
            return Err(VulkanError::Framework(
                FrameworkError::DescriptorSetBindingOutOfRange,
            ));
        }

        for (idx, (_, image_view)) in images.iter().enumerate() {
            if self.used_resources[first_layout_id as usize + idx]
                .replace(DescriptorSetBoundResource::ImageView(image_view.clone()))
                .is_some()
            {
                return Err(VulkanError::Framework(
                    FrameworkError::DescriptorSetBindingDuplicated,
                ));
            }
        }

        self.sampled_images
            .push((first_layout_id, images.iter().cloned().collect()));
        Ok(())
    }

    pub fn bind_storage_images(
        &mut self,
        first_layout_id: u32,
        images: &[(ImageLayout, Arc<ImageView>)],
    ) -> VulkanResult<()> {
        if first_layout_id as usize + images.len() > self.used_resources.len() {
            return Err(VulkanError::Framework(
                FrameworkError::DescriptorSetBindingOutOfRange,
            ));
        }

        for (idx, (_, image_view)) in images.iter().enumerate() {
            if self.used_resources[first_layout_id as usize + idx]
                .replace(DescriptorSetBoundResource::ImageView(image_view.clone()))
                .is_some()
            {
                return Err(VulkanError::Framework(
                    FrameworkError::DescriptorSetBindingDuplicated,
                ));
            }
        }

        self.storage_images
            .push((first_layout_id, images.iter().cloned().collect()));
        Ok(())
    }

    pub fn bind_uniform_buffer(
        &mut self,
        first_layout_id: u32,
        buffers: &[(Arc<dyn BufferTrait>, Option<u64>, Option<u64>)],
    ) -> VulkanResult<()> {
        if first_layout_id as usize + buffers.len() > self.used_resources.len() {
            return Err(VulkanError::Framework(
                FrameworkError::DescriptorSetBindingOutOfRange,
            ));
        }

        for (idx, (buffer, _, _)) in buffers.iter().enumerate() {
            if self.used_resources[first_layout_id as usize + idx]
                .replace(DescriptorSetBoundResource::Buffer(buffer.clone()))
                .is_some()
            {
                return Err(VulkanError::Framework(
                    FrameworkError::DescriptorSetBindingDuplicated,
                ));
            }
        }

        self.uniform_buffers.push((
            first_layout_id,
            buffers
                .iter()
                .map(|(buffer, maybe_offset, maybe_size)| {
                    let offset = match maybe_offset {
                        Some(offset) => offset.to_owned(),
                        None => 0u64,
                    };

                    let size = match maybe_size {
                        Some(sz) => u64::min(sz.to_owned(), buffer.size() - offset),
                        None => buffer.size() - offset,
                    };

                    (buffer.to_owned(), offset, size)
                })
                .collect(),
        ));
        Ok(())
    }

    pub fn bind_storage_buffers(
        &mut self,
        first_layout_id: u32,
        buffers: &[(Arc<dyn BufferTrait>, Option<u64>, Option<u64>)],
    ) -> VulkanResult<()> {
        if first_layout_id as usize + buffers.len() > self.used_resources.len() {
            return Err(VulkanError::Framework(
                FrameworkError::DescriptorSetBindingOutOfRange,
            ));
        }

        for (idx, (buffer, _, _)) in buffers.iter().enumerate() {
            if self.used_resources[first_layout_id as usize + idx]
                .replace(DescriptorSetBoundResource::Buffer(buffer.clone()))
                .is_some()
            {
                return Err(VulkanError::Framework(
                    FrameworkError::DescriptorSetBindingDuplicated,
                ));
            }
        }

        self.uniform_buffers.push((
            first_layout_id,
            buffers
                .iter()
                .map(|(buffer, maybe_offset, maybe_size)| {
                    let offset = match maybe_offset {
                        Some(offset) => offset.to_owned(),
                        None => 0u64,
                    };

                    let size = match maybe_size {
                        Some(sz) => u64::min(sz.to_owned(), buffer.size() - offset),
                        None => buffer.size() - offset,
                    };

                    (buffer.to_owned(), offset, size)
                })
                .collect(),
        ));
        Ok(())
    }
}

#[derive(Clone)]
pub enum DescriptorSetBoundResource {
    TLAS(Arc<TopLevelAccelerationStructure>),
    Buffer(Arc<dyn BufferTrait>),
    CombinedImageViewSampler((Arc<ImageView>, Arc<Sampler>)),
    ImageView(Arc<ImageView>),
}

pub struct DescriptorSet {
    pool: Arc<DescriptorPool>,
    layout: Arc<DescriptorSetLayout>,
    descriptor_set: ash::vk::DescriptorSet,
    bound_resources: Mutex<smallvec::SmallVec<[Option<DescriptorSetBoundResource>; 32]>>,
}

impl DescriptorSetLayoutDependant for DescriptorSet {
    fn get_parent_descriptor_set_layout(&self) -> Arc<DescriptorSetLayout> {
        self.layout.clone()
    }
}

impl DescriptorPoolOwned for DescriptorSet {
    fn get_parent_descriptor_pool(&self) -> Arc<DescriptorPool> {
        self.pool.clone()
    }
}

impl Drop for DescriptorSet {
    fn drop(&mut self) {
        let device = self.get_parent_descriptor_pool().get_parent_device();

        let _ = unsafe {
            device
                .ash_handle()
                .free_descriptor_sets(self.pool.ash_handle(), [self.descriptor_set].as_slice())
        };
    }
}

impl DescriptorSet {
    pub fn native_handle(&self) -> u64 {
        ash::vk::Handle::as_raw(self.descriptor_set)
    }

    pub(crate) fn ash_handle(&self) -> ash::vk::DescriptorSet {
        self.descriptor_set
    }

    pub fn perform_binding(&self, writer: &mut DescriptorSetWriter) -> VulkanResult<()> {
        assert_eq!(self as *const _, writer.descriptor_set as *const _);

        let acceleration_structures: smallvec::SmallVec<
            [(
                u32,
                smallvec::SmallVec<[ash::vk::AccelerationStructureKHR; 8]>,
            ); 4],
        > = writer
            .acceleration_structures
            .iter()
            .map(|(layout_idenx, accel_structures)| {
                (
                    layout_idenx.to_owned(),
                    accel_structures
                        .iter()
                        .map(|accel_s| accel_s.ash_handle())
                        .collect(),
                )
            })
            .collect();

        let mut vk_acceleration_structure_writers: smallvec::SmallVec<
            [(u32, ash::vk::WriteDescriptorSetAccelerationStructureKHR); 4],
        > = acceleration_structures
            .iter()
            .map(|(first_index, accel_structures)| {
                (
                    first_index.to_owned(),
                    ash::vk::WriteDescriptorSetAccelerationStructureKHR::default()
                        .acceleration_structures(accel_structures.as_slice()),
                )
            })
            .collect();

        let vk_combined_image_sampler_writers: smallvec::SmallVec<
            [(u32, smallvec::SmallVec<[ash::vk::DescriptorImageInfo; 8]>); 4],
        > = writer
            .combined_image_sampler
            .iter()
            .map(|(first_layout_id, images)| {
                (
                    first_layout_id.to_owned(),
                    images
                        .iter()
                        .map(|(image_layout, image_view, image_sampler)| {
                            ash::vk::DescriptorImageInfo::default()
                                .image_view(image_view.ash_handle())
                                .sampler(image_sampler.ash_handle())
                                .image_layout(image_layout.ash_layout())
                        })
                        .collect(),
                )
            })
            .collect();

        let vk_sampled_images_writers: smallvec::SmallVec<
            [(u32, smallvec::SmallVec<[ash::vk::DescriptorImageInfo; 8]>); 4],
        > = writer
            .sampled_images
            .iter()
            .map(|(first_layout_id, images)| {
                (
                    first_layout_id.to_owned(),
                    images
                        .iter()
                        .map(|(image_layout, image_view)| {
                            ash::vk::DescriptorImageInfo::default()
                                .image_view(image_view.ash_handle())
                                .image_layout(image_layout.ash_layout())
                        })
                        .collect(),
                )
            })
            .collect();

        let vk_storage_images_writers: smallvec::SmallVec<
            [(u32, smallvec::SmallVec<[ash::vk::DescriptorImageInfo; 8]>); 4],
        > = writer
            .storage_images
            .iter()
            .map(|(first_layout_id, images)| {
                (
                    first_layout_id.to_owned(),
                    images
                        .iter()
                        .map(|(image_layout, image_view)| {
                            ash::vk::DescriptorImageInfo::default()
                                .image_view(image_view.ash_handle())
                                .image_layout(image_layout.ash_layout())
                        })
                        .collect(),
                )
            })
            .collect();

        let vk_storage_buffer_writers: smallvec::SmallVec<
            [(u32, smallvec::SmallVec<[ash::vk::DescriptorBufferInfo; 8]>); 4],
        > = writer
            .storage_buffers
            .iter()
            .map(|(first_layout_id, images)| {
                (
                    first_layout_id.to_owned(),
                    images
                        .iter()
                        .map(|(buffer, offset, range)| {
                            ash::vk::DescriptorBufferInfo::default()
                                .range(range.to_owned())
                                .buffer(ash::vk::Buffer::from_raw(buffer.native_handle()))
                                .offset(offset.to_owned())
                        })
                        .collect(),
                )
            })
            .collect();

        let vk_uniform_buffer_writers: smallvec::SmallVec<
            [(u32, smallvec::SmallVec<[ash::vk::DescriptorBufferInfo; 8]>); 4],
        > = writer
            .uniform_buffers
            .iter()
            .map(|(first_layout_id, images)| {
                (
                    first_layout_id.to_owned(),
                    images
                        .iter()
                        .map(|(buffer, offset, range)| {
                            ash::vk::DescriptorBufferInfo::default()
                                .range(range.to_owned())
                                .buffer(ash::vk::Buffer::from_raw(buffer.native_handle()))
                                .offset(offset.to_owned())
                        })
                        .collect(),
                )
            })
            .collect();

        let vk_writers: smallvec::SmallVec<[ash::vk::WriteDescriptorSet; 24]> =
            vk_acceleration_structure_writers
                .iter_mut()
                .map(|(first_layout_id, structure)| {
                    ash::vk::WriteDescriptorSet::default()
                        .dst_set(self.ash_handle())
                        .descriptor_type(ash::vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
                        .dst_binding(first_layout_id.to_owned())
                        .descriptor_count(1)
                        .push_next(structure)
                })
                .chain(vk_combined_image_sampler_writers.iter().map(
                    |(first_layout_id, image_info)| {
                        ash::vk::WriteDescriptorSet::default()
                            .dst_set(self.ash_handle())
                            .dst_binding(first_layout_id.to_owned())
                            .descriptor_type(ash::vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                            .image_info(image_info.as_slice())
                    },
                ))
                .chain(
                    vk_sampled_images_writers
                        .iter()
                        .map(|(first_layout_id, image_info)| {
                            ash::vk::WriteDescriptorSet::default()
                                .dst_set(self.ash_handle())
                                .dst_binding(first_layout_id.to_owned())
                                .descriptor_type(ash::vk::DescriptorType::SAMPLED_IMAGE)
                                .image_info(image_info.as_slice())
                        }),
                )
                .chain(
                    vk_storage_images_writers
                        .iter()
                        .map(|(first_layout_id, image_info)| {
                            ash::vk::WriteDescriptorSet::default()
                                .dst_set(self.ash_handle())
                                .dst_binding(first_layout_id.to_owned())
                                .descriptor_type(ash::vk::DescriptorType::STORAGE_IMAGE)
                                .image_info(image_info.as_slice())
                        }),
                )
                .chain(
                    vk_storage_buffer_writers
                        .iter()
                        .map(|(first_layout_id, buffer_info)| {
                            ash::vk::WriteDescriptorSet::default()
                                .dst_set(self.ash_handle())
                                .dst_binding(first_layout_id.to_owned())
                                .descriptor_type(ash::vk::DescriptorType::STORAGE_BUFFER)
                                .buffer_info(buffer_info.as_slice())
                        }),
                )
                .chain(
                    vk_uniform_buffer_writers
                        .iter()
                        .map(|(first_layout_id, buffer_info)| {
                            ash::vk::WriteDescriptorSet::default()
                                .dst_set(self.ash_handle())
                                .dst_binding(first_layout_id.to_owned())
                                .descriptor_type(ash::vk::DescriptorType::UNIFORM_BUFFER)
                                .buffer_info(buffer_info.as_slice())
                        }),
                )
                .collect();

        unsafe {
            self.get_parent_descriptor_pool()
                .get_parent_device()
                .ash_handle()
                .update_descriptor_sets(vk_writers.as_slice(), &[])
        };

        Ok(())
    }

    pub fn bind_resources<F>(&self, f: F) -> VulkanResult<()>
    where
        F: FnOnce(&mut DescriptorSetWriter),
    {
        #[cfg(feature = "better_mutex")]
        let mut lck = self.bound_resources.lock();

        #[cfg(not(feature = "better_mutex"))]
        let mut lck = match self.bound_resources.lock() {
            Ok(lock) => lock,
            Err(err) => {
                return Err(VulkanError::Framework(FrameworkError::Unknown(Some(
                    format!("Error acquiring the descriptor set mutex: {}", err),
                ))))
            }
        };

        let mut writer = DescriptorSetWriter::new(self, lck.len() as u32);

        f(&mut writer);

        self.perform_binding(&mut writer)?;

        for (idx, res) in writer.used_resources().iter().enumerate() {
            match res {
                None => {}
                Some(updated_resource) => {
                    (*lck)[idx] = Some(updated_resource.clone());
                }
            }
        }

        Ok(())
    }

    pub fn new(
        pool: Arc<DescriptorPool>,
        layout: Arc<DescriptorSetLayout>,
    ) -> VulkanResult<Arc<Self>> {
        #[cfg(debug_assertions)]
        {
            assert_eq!(
                layout.get_parent_device().native_handle(),
                pool.get_parent_device().native_handle()
            );
        }

        let (min_idx, max_idx) = layout.binding_range();

        if min_idx != 0 {
            return Err(VulkanError::Framework(FrameworkError::UserInput(Some(
                "Error creating the descriptor set: bindings are not starting from zero"
                    .to_string(),
            ))));
        }

        let layouts = [layout.ash_handle()];
        let create_info = ash::vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(pool.ash_handle())
            .set_layouts(&layouts);

        match unsafe {
            pool.get_parent_device()
                .ash_handle()
                .allocate_descriptor_sets(&create_info)
        } {
            Ok(descriptor_set) => {
                let guarded_resource = (0..(max_idx + 1)).map(|_idx| Option::default()).collect();

                #[cfg(feature = "better_mutex")]
                let bound_resources = const_mutex(guarded_resource);

                #[cfg(not(feature = "better_mutex"))]
                let bound_resources = Mutex::new(guarded_resource);

                Ok(Arc::new(Self {
                    pool,
                    descriptor_set: descriptor_set[0],
                    layout,
                    bound_resources,
                }))
            }
            Err(err) => Err(VulkanError::Vulkan(
                err.as_raw(),
                Some(format!("Error creating the descriptor set: {}", err)),
            )),
        }
    }
}

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
    acceleration_structures: smallvec::SmallVec<[Vec<ash::vk::AccelerationStructureKHR>; 4]>,
    acceleration_structure_writers:
        smallvec::SmallVec<[ash::vk::WriteDescriptorSetAccelerationStructureKHR; 4]>,
    images_writers: smallvec::SmallVec<[Vec<ash::vk::DescriptorImageInfo>; 32]>,
    buffers_writers: smallvec::SmallVec<[Vec<ash::vk::DescriptorBufferInfo>; 32]>,
    writer: smallvec::SmallVec<[ash::vk::WriteDescriptorSet; 32]>,
    used_resources: smallvec::SmallVec<[DescriptorSetBoundResource; 32]>,
}

impl<'a> DescriptorSetWriter<'a> {
    pub(crate) fn new(descriptor_set: &'a DescriptorSet, size: u32) -> Self {
        Self {
            /*device: descriptor_set
            .get_parent_descriptor_pool()
            .get_parent_device()
            .clone(),*/
            descriptor_set,
            acceleration_structures: smallvec::smallvec![],
            acceleration_structure_writers: smallvec::smallvec![],
            images_writers: smallvec::smallvec![],
            buffers_writers: smallvec::smallvec![],
            writer: smallvec::smallvec![],
            //binder: ash::vk::WriteDescriptorSet::builder(),
            used_resources: (0..size)
                .map(|_idx| DescriptorSetBoundResource::None)
                .collect(),
        }
    }

    pub(crate) fn ref_used_resources<'b>(
        &'a self,
    ) -> impl Iterator<Item = &'a DescriptorSetBoundResource>
    where
        'b: 'a,
    {
        self.used_resources.iter()
    }

    pub fn bind_tlas(
        &mut self,
        first_layout_id: u32,
        tlas_collection: &[Arc<TopLevelAccelerationStructure>],
    ) {
        let as_handles = tlas_collection
            .iter()
            .map(|accel_structure| accel_structure.ash_handle())
            .collect();
        self.acceleration_structures.push(as_handles);

        self.acceleration_structure_writers.push(
            ash::vk::WriteDescriptorSetAccelerationStructureKHR::builder()
                .acceleration_structures(
                    self.acceleration_structures[self.acceleration_structures.len() - 1].as_slice(),
                )
                .build(),
        );

        let mut descriptor_writes = ash::vk::WriteDescriptorSet::builder()
            .dst_set(self.descriptor_set.ash_handle())
            .dst_binding(first_layout_id)
            .descriptor_type(ash::vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
            .build();

        descriptor_writes.p_next = (&self.acceleration_structure_writers
            [self.acceleration_structure_writers.len() - 1]
            as *const _) as *const std::ffi::c_void;
        descriptor_writes.descriptor_count = 1;

        self.writer.push(descriptor_writes);
    }

    pub fn bind_combined_images_samplers(
        &mut self,
        first_layout_id: u32,
        images: &[(ImageLayout, Arc<ImageView>, Arc<Sampler>)],
    ) {
        let descriptors: Vec<ash::vk::DescriptorImageInfo> = images
            .iter()
            .enumerate()
            .map(|(index, image)| {
                // TODO: assert usage has VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT bit set
                let (image_layout, image_view, image_sampler) = image;

                self.used_resources[(first_layout_id as usize) + index] =
                    DescriptorSetBoundResource::CombinedImageViewSampler((
                        image_view.clone(),
                        image_sampler.clone(),
                    ));

                ash::vk::DescriptorImageInfo::builder()
                    .image_view(image_view.ash_handle())
                    .sampler(image_sampler.ash_handle())
                    .image_layout(image_layout.ash_layout())
                    .build()
            })
            .collect();

        self.images_writers.push(descriptors);

        let descriptor_writes = ash::vk::WriteDescriptorSet::builder()
            .dst_set(self.descriptor_set.ash_handle())
            .dst_binding(first_layout_id)
            .descriptor_type(ash::vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(self.images_writers[self.images_writers.len() - 1].as_slice())
            .build();

        self.writer.push(descriptor_writes);
    }

    pub fn bind_sampled_images(
        &mut self,
        first_layout_id: u32,
        images: &[(ImageLayout, Arc<ImageView>)],
    ) {
        let descriptors: Vec<ash::vk::DescriptorImageInfo> = images
            .iter()
            .enumerate()
            .map(|(index, image)| {
                // TODO: assert usage has VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT bit set
                let (image_layout, image_view) = image;

                self.used_resources[(first_layout_id as usize) + index] =
                    DescriptorSetBoundResource::ImageView(image_view.clone());

                ash::vk::DescriptorImageInfo::builder()
                    .image_view(image_view.ash_handle())
                    .image_layout(image_layout.ash_layout())
                    .build()
            })
            .collect();

        self.images_writers.push(descriptors);

        let descriptor_writes = ash::vk::WriteDescriptorSet::builder()
            .dst_set(self.descriptor_set.ash_handle())
            .dst_binding(first_layout_id)
            .descriptor_type(ash::vk::DescriptorType::SAMPLED_IMAGE)
            .image_info(self.images_writers[self.images_writers.len() - 1].as_slice())
            .build();

        self.writer.push(descriptor_writes);
    }

    pub fn bind_uniform_buffer(
        &mut self,
        first_layout_id: u32,
        buffers: &[Arc<dyn BufferTrait>],
        offset: Option<u64>,
        size: Option<u64>,
    ) {
        let descriptors: Vec<ash::vk::DescriptorBufferInfo> = buffers
            .iter()
            .enumerate()
            .map(|(index, buffer)| {
                // TODO: assert usage has VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT bit set

                self.used_resources[(first_layout_id as usize) + index] =
                    DescriptorSetBoundResource::Buffer(buffer.clone());

                ash::vk::DescriptorBufferInfo::builder()
                    .range(match size {
                        Option::Some(sz) => sz,
                        Option::None => buffer.size(),
                    })
                    .buffer(ash::vk::Buffer::from_raw(buffer.native_handle()))
                    .offset(offset.unwrap_or(0))
                    .build()
            })
            .collect();

        self.buffers_writers.push(descriptors);

        let descriptor_writes = ash::vk::WriteDescriptorSet::builder()
            .dst_set(self.descriptor_set.ash_handle())
            .dst_binding(first_layout_id)
            .descriptor_type(ash::vk::DescriptorType::UNIFORM_BUFFER)
            .buffer_info(self.buffers_writers[self.buffers_writers.len() - 1].as_slice())
            .build();

        self.writer.push(descriptor_writes);
    }

    pub fn bind_storage_buffers(
        &mut self,
        first_layout_id: u32,
        buffers: &[Arc<dyn BufferTrait>],
        offset: Option<u64>,
        size: Option<u64>,
    ) {
        let descriptors: Vec<ash::vk::DescriptorBufferInfo> = buffers
            .iter()
            .enumerate()
            .map(|(index, buffer)| {
                // TODO: assert usage has VK_BUFFER_USAGE_STORAGE_BUFFER_BIT bit set

                self.used_resources[(first_layout_id as usize) + index] =
                    DescriptorSetBoundResource::Buffer(buffer.clone());

                ash::vk::DescriptorBufferInfo::builder()
                    .range(match size {
                        Option::Some(sz) => sz,
                        Option::None => buffer.size(),
                    })
                    .buffer(ash::vk::Buffer::from_raw(buffer.native_handle()))
                    .offset(offset.unwrap_or(0))
                    .build()
            })
            .collect();

        self.buffers_writers.push(descriptors);

        let descriptor_writes = ash::vk::WriteDescriptorSet::builder()
            .dst_set(self.descriptor_set.ash_handle())
            .dst_binding(first_layout_id)
            .descriptor_type(ash::vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(self.buffers_writers[self.buffers_writers.len() - 1].as_slice())
            .build();

        self.writer.push(descriptor_writes);
    }

    pub fn bind_storage_images(
        &mut self,
        first_layout_id: u32,
        images: &[(ImageLayout, Arc<ImageView>)],
    ) {
        let descriptors: Vec<ash::vk::DescriptorImageInfo> = images
            .iter()
            .enumerate()
            .map(|(index, (layout, image))| {
                // TODO: assert usage has the right bit set and layout is not ImageLayout::Unspecified

                self.used_resources[(first_layout_id as usize) + index] =
                    DescriptorSetBoundResource::ImageView(image.clone());

                ash::vk::DescriptorImageInfo::builder()
                    .image_layout(layout.ash_layout())
                    .image_view(image.ash_handle())
                    .build()
            })
            .collect();

        self.images_writers.push(descriptors);

        let descriptor_writes = ash::vk::WriteDescriptorSet::builder()
            .dst_set(self.descriptor_set.ash_handle())
            .dst_binding(first_layout_id)
            .descriptor_type(ash::vk::DescriptorType::STORAGE_IMAGE)
            .image_info(self.images_writers[self.images_writers.len() - 1].as_slice())
            .build();

        self.writer.push(descriptor_writes);
    }
}

#[derive(Clone)]
pub enum DescriptorSetBoundResource {
    None,
    Buffer(Arc<dyn BufferTrait>),
    CombinedImageViewSampler((Arc<ImageView>, Arc<Sampler>)),
    ImageView(Arc<ImageView>),
}

pub struct DescriptorSet {
    pool: Arc<DescriptorPool>,
    layout: Arc<DescriptorSetLayout>,
    descriptor_set: ash::vk::DescriptorSet,
    bound_resources: Mutex<Vec<DescriptorSetBoundResource>>,
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

    pub fn bind_resources<F>(&self, f: F) -> VulkanResult<()>
    where
        F: Fn(&mut DescriptorSetWriter),
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

        unsafe {
            self.get_parent_descriptor_pool()
                .get_parent_device()
                .ash_handle()
                .update_descriptor_sets(writer.writer.as_slice(), &[])
        };

        for (idx, res) in writer.ref_used_resources().enumerate() {
            match res {
                DescriptorSetBoundResource::None => {}
                updated_resource => {
                    (*lck)[idx] = updated_resource.clone();
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

        let create_info = ash::vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(pool.ash_handle())
            .set_layouts([layout.ash_handle()].as_slice())
            .build();

        match unsafe {
            pool.get_parent_device()
                .ash_handle()
                .allocate_descriptor_sets(&create_info)
        } {
            Ok(descriptor_set) => {
                let guarded_resource = (0..(max_idx + 1))
                    .map(|_idx| DescriptorSetBoundResource::None)
                    .collect();

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

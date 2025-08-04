use std::{collections::VecDeque, sync::Arc};

use vulkan_framework::{
    buffer::{AllocatedBuffer, Buffer},
    device::Device,
    image::{AllocatedImage, Image},
    memory_allocator::{DefaultAllocator, MemoryAllocator},
    memory_heap::{
        ConcreteMemoryHeapDescriptor, MemoryHeap, MemoryHeapOwned, MemoryRequirements, MemoryType,
    },
    memory_pool::{MemoryPool, MemoryPoolFeatures},
    memory_requiring::{MemoryRequiring, UnallocatedResource},
    prelude::{FrameworkError, VulkanError, VulkanResult},
};

pub struct MemoryManager {
    device: Arc<Device>,
    memory_pools: Vec<Arc<MemoryPool>>,
}

const BLOCK_SIZE: u64 = 512u64;

enum ResourceAllocationResult {
    AllocatedBuffer(Arc<AllocatedBuffer>),
    AllocatedImage(Arc<AllocatedImage>),
    Unallocated(UnallocatedResource),
}

impl MemoryManager {
    pub fn new(device: Arc<Device>) -> Self {
        let memory_pools = vec![];
        Self {
            device,
            memory_pools,
        }
    }

    pub fn allocate_buffers<I>(
        &mut self,
        memory_type: impl AsRef<MemoryType>,
        features: impl AsRef<MemoryPoolFeatures>,
        buffers: I,
    ) -> VulkanResult<smallvec::SmallVec<[Arc<AllocatedBuffer>; 4]>>
    where
        I: IntoIterator<Item = Buffer>,
    {
        let mut unallocated: VecDeque<UnallocatedResource> = buffers
            .into_iter()
            .map(|resource| UnallocatedResource::Buffer(resource))
            .collect();

        let requested_allocations = unallocated.len();

        let (allocated_buffers, allocated_images) =
            self.allocate_resources(memory_type, features, &mut unallocated)?;
        assert_eq!(allocated_buffers.len(), requested_allocations);
        assert!(allocated_images.is_empty());

        Ok(allocated_buffers)
    }

    pub fn allocate_images<I>(
        &mut self,
        memory_type: impl AsRef<MemoryType>,
        features: impl AsRef<MemoryPoolFeatures>,
        images: I,
    ) -> VulkanResult<smallvec::SmallVec<[Arc<AllocatedImage>; 4]>>
    where
        I: IntoIterator<Item = Image>,
    {
        let mut unallocated: VecDeque<UnallocatedResource> = images
            .into_iter()
            .map(|resource| UnallocatedResource::Image(resource))
            .collect();

        let requested_allocations = unallocated.len();

        let (allocated_buffers, allocated_images) =
            self.allocate_resources(memory_type, features, &mut unallocated)?;
        assert_eq!(allocated_images.len(), requested_allocations);
        assert!(allocated_buffers.is_empty());

        Ok(allocated_images)
    }

    /// Allocate resources on the specified pool, ignoring resources that don't fit.
    fn allocate_resources_on_pool(
        memory_pool: Arc<MemoryPool>,
        unallocated: &mut VecDeque<UnallocatedResource>,
    ) -> VulkanResult<(
        smallvec::SmallVec<[Arc<AllocatedBuffer>; 4]>,
        smallvec::SmallVec<[Arc<AllocatedImage>; 4]>,
    )> {
        let mut allocated_buffers =
            smallvec::SmallVec::<[Arc<AllocatedBuffer>; 4]>::with_capacity(unallocated.len());

        let mut allocated_images =
            smallvec::SmallVec::<[Arc<AllocatedImage>; 4]>::with_capacity(unallocated.len());

        let mut index = (unallocated.len() as i128) - 1;
        while index >= 0 {
            let taken = unallocated.remove(index as usize).unwrap();

            match Self::allocate_resource_on_pool(memory_pool.clone(), taken)? {
                ResourceAllocationResult::AllocatedBuffer(allocated_buffer) => {
                    allocated_buffers.push(allocated_buffer)
                }
                ResourceAllocationResult::AllocatedImage(allocated_image) => {
                    allocated_images.push(allocated_image)
                }
                ResourceAllocationResult::Unallocated(failed_resource) => {
                    unallocated.push_back(failed_resource);
                }
            };

            // go to the preceding element
            index -= 1;
        }

        Ok((allocated_buffers, allocated_images))
    }

    pub fn allocate_resources(
        &mut self,
        memory_type: impl AsRef<MemoryType>,
        features: impl AsRef<MemoryPoolFeatures>,
        unallocated: &mut VecDeque<UnallocatedResource>,
    ) -> VulkanResult<(
        smallvec::SmallVec<[Arc<AllocatedBuffer>; 4]>,
        smallvec::SmallVec<[Arc<AllocatedImage>; 4]>,
    )> {
        let buffer_mem_requiring =
            smallvec::SmallVec::<[&dyn MemoryRequiring; 4]>::with_capacity(unallocated.len());
        let requirements = MemoryRequirements::try_from(buffer_mem_requiring.as_slice())?;
        drop(buffer_mem_requiring);

        let mut allocated_buffers =
            smallvec::SmallVec::<[Arc<AllocatedBuffer>; 4]>::with_capacity(unallocated.len());

        let mut allocated_images =
            smallvec::SmallVec::<[Arc<AllocatedImage>; 4]>::with_capacity(unallocated.len());

        // try allocating resources on available memory pools first
        for memory_pool in self.memory_pools.iter() {
            if !memory_pool.support_features(features.as_ref()) {
                continue;
            } else if !memory_pool
                .get_parent_memory_heap()
                .suitable_memory_type(&requirements)
            {
                continue;
            } else if memory_pool.get_parent_memory_heap().memory_type() != *(memory_type.as_ref())
            {
                continue;
            }

            let (alloc_bufs, alloc_imgs) =
                Self::allocate_resources_on_pool(memory_pool.clone(), unallocated)?;

            allocated_buffers = allocated_buffers
                .into_iter()
                .chain(alloc_bufs.into_iter())
                .collect();
            allocated_images = allocated_images
                .into_iter()
                .chain(alloc_imgs.into_iter())
                .collect();
        }

        // try to allocate resources until either all have been allocated or an error is found
        if !unallocated.is_empty() {
            let memory_amount: u64 = unallocated
                .iter()
                .map(|resource| {
                    let requirements = resource.memory_requirements();

                    let required_size = (requirements.size() / BLOCK_SIZE) + 1u64;

                    let required_alignment = (requirements.alignment() / BLOCK_SIZE) + 1u64;

                    required_size + required_alignment
                })
                .sum();

            let memory_pool = self.take_new_pool(
                memory_type.as_ref(),
                memory_amount,
                features.as_ref(),
                &requirements,
            )?;

            // register this memory pool as allocated
            self.memory_pools.push(memory_pool.clone());

            let (alloc_bufs, alloc_imgs) =
                Self::allocate_resources_on_pool(memory_pool.clone(), unallocated)?;

            allocated_buffers = allocated_buffers
                .into_iter()
                .chain(alloc_bufs.into_iter())
                .collect();
            allocated_images = allocated_images
                .into_iter()
                .chain(alloc_imgs.into_iter())
                .collect();
        }

        assert!(unallocated.is_empty());

        Ok((allocated_buffers, allocated_images))
    }

    fn allocate_resource_on_pool(
        memory_pool: Arc<MemoryPool>,
        resource: UnallocatedResource,
    ) -> VulkanResult<ResourceAllocationResult> {
        match resource {
            UnallocatedResource::Buffer(buffer) => {
                match AllocatedBuffer::new(memory_pool.clone(), buffer) {
                    Ok(allocated_buffer) => {
                        Ok(ResourceAllocationResult::AllocatedBuffer(allocated_buffer))
                    }
                    Err(err) => match err {
                        VulkanError::Framework(fw_err) => match fw_err {
                            FrameworkError::MallocFail(failed_resource) => {
                                Ok(ResourceAllocationResult::Unallocated(failed_resource))
                            }
                            fw_err => return Err(VulkanError::Framework(fw_err)),
                        },
                        _ => return Err(err),
                    },
                }
            }
            UnallocatedResource::Image(image) => {
                match AllocatedImage::new(memory_pool.clone(), image) {
                    Ok(allocated_image) => {
                        Ok(ResourceAllocationResult::AllocatedImage(allocated_image))
                    }
                    Err(err) => match err {
                        VulkanError::Framework(fw_err) => match fw_err {
                            FrameworkError::MallocFail(failed_resource) => {
                                Ok(ResourceAllocationResult::Unallocated(failed_resource))
                            }
                            fw_err => return Err(VulkanError::Framework(fw_err)),
                        },
                        _ => return Err(err),
                    },
                }
            }
        }
    }

    fn take_new_pool(
        &self,
        memory_type: &MemoryType,
        number_of_blocks: u64,
        features: &MemoryPoolFeatures,
        requirements: &MemoryRequirements,
    ) -> VulkanResult<Arc<MemoryPool>> {
        let allocator = DefaultAllocator::with_blocksize(BLOCK_SIZE, number_of_blocks + 1u64);

        let memory_heap = match self.find_suitable_heap(memory_type, &requirements) {
            Some(memory_heap) => memory_heap,
            None => MemoryHeap::new(
                self.device.clone(),
                ConcreteMemoryHeapDescriptor::new(*memory_type, allocator.total_size()),
                *requirements,
            )?,
        };

        Ok(MemoryPool::new(
            memory_heap,
            Arc::new(allocator),
            *features,
        )?)
    }

    fn find_suitable_heap(
        &self,
        memory_type: &MemoryType,
        requirements: &MemoryRequirements,
    ) -> Option<Arc<MemoryHeap>> {
        for memory_pool in self.memory_pools.iter() {
            let memory_heap = memory_pool.get_parent_memory_heap();
            if !memory_heap.suitable_memory_type(&requirements) {
                continue;
            } else if memory_heap.memory_type() != *memory_type {
                continue;
            }

            return Some(memory_heap);
        }

        None
    }
}

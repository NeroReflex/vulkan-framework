use std::{sync::Arc, vec::Vec};

use crate::{
    buffer::{AllocatedBuffer, Buffer},
    device::{Device, DeviceOwned},
    image::{AllocatedImage, Image},
    memory_allocator::{DefaultAllocator, MemoryAllocator},
    memory_heap::{
        ConcreteMemoryHeapDescriptor, MemoryHeap, MemoryHeapOwned, MemoryRequirements, MemoryType,
    },
    memory_pool::{MemoryPool, MemoryPoolFeatures},
    memory_requiring::{AllocationRequirements, AllocationRequiring},
    prelude::{FrameworkError, VulkanError, VulkanResult},
};

#[derive(Debug)]
pub enum UnallocatedResource {
    Buffer(Buffer),
    Image(Image),
}

impl From<Buffer> for UnallocatedResource {
    fn from(value: Buffer) -> Self {
        UnallocatedResource::Buffer(value)
    }
}

impl From<Image> for UnallocatedResource {
    fn from(value: Image) -> Self {
        UnallocatedResource::Image(value)
    }
}

pub enum AllocatedResource {
    Buffer(Arc<AllocatedBuffer>),
    Image(Arc<AllocatedImage>),
}

impl AllocatedResource {
    pub fn buffer(&self) -> Arc<AllocatedBuffer> {
        match self {
            AllocatedResource::Buffer(buffer) => buffer.clone(),
            AllocatedResource::Image(_) => panic!("Resource is an image."),
        }
    }

    pub fn image(&self) -> Arc<AllocatedImage> {
        match self {
            AllocatedResource::Buffer(_) => panic!("Resource is a buffer."),
            AllocatedResource::Image(image) => image.clone(),
        }
    }
}

impl UnallocatedResource {
    pub fn memory_requirements(&self) -> AllocationRequirements {
        let memory_requiring = match self {
            UnallocatedResource::Buffer(buffer) => buffer as &dyn AllocationRequiring,
            UnallocatedResource::Image(image) => image as &dyn AllocationRequiring,
        };

        memory_requiring.allocation_requirements()
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Default)]
pub enum MemoryManagementTagSize {
    #[default]
    Small,
    MediumSmall,
    Medium,
    MediumLarge,
    Large,
    VeryLarge,
}

/// Represents hints to the memory manager:
///
/// Name is used to group resources together
/// Exclusive is used to create a pool large enough to hold resources and reserve it
/// Size: give the preferred free size of the new pool in case a new one gets allocated
#[derive(Default, Clone, PartialEq, Eq)]
pub struct MemoryManagementTags {
    name: Option<String>,
    exlusive: bool,
    size: MemoryManagementTagSize,
}

impl MemoryManagementTags {
    pub fn with_name(&self, name: String) -> MemoryManagementTags {
        let mut muted_tags = self.clone();
        muted_tags.name = Some(name);
        muted_tags
    }

    pub fn with_exclusivity(&self, exclusive: bool) -> MemoryManagementTags {
        let mut muted_tags = self.clone();
        muted_tags.exlusive = exclusive;
        muted_tags
    }

    pub fn with_size(&self, size: MemoryManagementTagSize) -> MemoryManagementTags {
        let mut muted_tags = self.clone();
        muted_tags.size = size;
        muted_tags
    }

    pub fn exlusive(&self) -> bool {
        self.exlusive
    }

    pub fn name(&self) -> &Option<String> {
        &self.name
    }

    pub fn size(&self) -> &MemoryManagementTagSize {
        &self.size
    }

    pub fn same_name(&self, other: &Self) -> bool {
        let Some(my_name) = &self.name else {
            return false;
        };

        let Some(other_name) = &other.name else {
            return false;
        };

        my_name.eq(other_name)
    }

    pub fn has_name(&self) -> bool {
        self.name.is_some()
    }
}

/// Define the interface for a memory pool manager, to be used
/// for safely allocating memory for vulkan `AllocationRequiring` resources.
pub trait MemoryManagerTrait: DeviceOwned {
    /// Implement this function to allocate resoureces:
    ///
    /// The allocator is free to ignore tags or implement cusom behaviour
    /// for each tag.
    ///
    /// It is VERY important that these two conditions are ALWAYS met:
    ///     - every resource is allocated and MallocFail is NEVER returned
    ///     - resources are returned in the very same order they come in
    fn allocate_resources(
        &mut self,
        memory_type: &MemoryType,
        features: &MemoryPoolFeatures,
        resources: Vec<UnallocatedResource>,
        tags: MemoryManagementTags,
    ) -> VulkanResult<Vec<AllocatedResource>>;
}

const BLOCK_SIZE: u64 = 512u64;

enum ResourceAllocationResult {
    AllocatedBuffer((usize, Arc<AllocatedBuffer>)),
    AllocatedImage((usize, Arc<AllocatedImage>)),
    Unallocated((usize, UnallocatedResource)),
}

pub struct DefaultMemoryManager {
    device: Arc<Device>,
    memory_pools: Vec<(MemoryManagementTags, Arc<MemoryPool>)>,
}

impl DeviceOwned for DefaultMemoryManager {
    fn get_parent_device(&self) -> Arc<Device> {
        self.device.clone()
    }
}

impl MemoryManagerTrait for DefaultMemoryManager {
    fn allocate_resources(
        &mut self,
        memory_type: &MemoryType,
        features: &MemoryPoolFeatures,
        unallocated: Vec<UnallocatedResource>,
        tags: MemoryManagementTags,
    ) -> VulkanResult<Vec<AllocatedResource>> {
        let requirements = {
            let mem_requiring = unallocated
                .iter()
                .map(|r| match r {
                    UnallocatedResource::Buffer(buffer) => buffer as &dyn AllocationRequiring,
                    UnallocatedResource::Image(image) => image as &dyn AllocationRequiring,
                })
                .collect::<smallvec::SmallVec<[&dyn AllocationRequiring; 8]>>();
            MemoryRequirements::try_from(mem_requiring.as_slice())?
        };

        let mut allocations = Vec::with_capacity(unallocated.len());
        let mut unallocated = unallocated.into_iter().enumerate().collect::<Vec<_>>();

        // try allocating resources on available memory pools first
        for attempt in 0..2 {
            // if resources have been marked as pool-exlusive do not try to allocate them in other pools:
            // keep them together by skipping this for loop: the following if will create a new pool reserved for them
            if tags.exlusive() {
                break;
            }

            for (pool_tags, memory_pool) in self.memory_pools.iter() {
                if pool_tags.exlusive() {
                    continue;
                } else if (attempt == 0) && (tags.has_name()) && (!tags.same_name(pool_tags)) {
                    continue;
                } else if !memory_pool.support_features(features) {
                    continue;
                } else if !memory_pool
                    .get_parent_memory_heap()
                    .suitable_memory_type(&requirements)
                {
                    continue;
                } else if memory_pool.get_parent_memory_heap().memory_type() != *memory_type {
                    continue;
                }

                for allocation in
                    Self::allocate_resources_on_pool(memory_pool.clone(), &mut unallocated)?
                        .into_iter()
                {
                    allocations.push(allocation);
                }
            }

            // do not retry if a preferred name was not specified
            if !tags.has_name() {
                break;
            }
        }

        // resouces could not have been fit into existing pools: allocate a new one
        let leftover_memory = match tags.exlusive() {
            true => 0,
            false => match tags.size() {
                // small is 4MiB
                MemoryManagementTagSize::Small => 4 * 1024 * 1024,
                // medium-small is 32MiB
                MemoryManagementTagSize::MediumSmall => 32 * 1024 * 1024,
                // medium is 64MiB
                MemoryManagementTagSize::Medium => 64 * 1024 * 1024,
                // medium-large is 128MiB
                MemoryManagementTagSize::MediumLarge => 128 * 1024 * 1024,
                // large is 256MiB
                MemoryManagementTagSize::Large => 256 * 1024 * 1024,
                // very large is 512MiB
                MemoryManagementTagSize::VeryLarge => 512 * 1024 * 1024,
            },
        };

        if !unallocated.is_empty() {
            let memory_amount: u64 = (leftover_memory / BLOCK_SIZE)
                + unallocated
                    .iter()
                    .map(|(_, resource)| {
                        let requirements = resource.memory_requirements();

                        let required_size = (requirements.size() / BLOCK_SIZE) + 1u64;

                        let required_alignment = (requirements.alignment() / BLOCK_SIZE) + 1u64;

                        required_size + required_alignment
                    })
                    .sum::<u64>();

            let memory_pool =
                self.take_new_pool(memory_type, memory_amount, features, &requirements)?;

            // register this memory pool as allocated
            self.memory_pools.push((tags, memory_pool.clone()));
            for allocation in
                Self::allocate_resources_on_pool(memory_pool.clone(), &mut unallocated)?.into_iter()
            {
                allocations.push(allocation);
            }
        }

        assert!(unallocated.is_empty());

        allocations.sort_by_key(|&(index, _)| index);
        Ok(allocations.into_iter().map(|(_, value)| value).collect())
    }
}

impl DefaultMemoryManager {
    pub fn new(device: Arc<Device>) -> Self {
        let memory_pools = vec![];
        Self {
            device,
            memory_pools,
        }
    }

    #[inline(always)]
    pub fn device(&self) -> Arc<Device> {
        self.device.clone()
    }

    /// Allocate resources on the specified pool, ignoring resources that don't fit.
    fn allocate_resources_on_pool(
        memory_pool: Arc<MemoryPool>,
        unallocated: &mut Vec<(usize, UnallocatedResource)>,
    ) -> VulkanResult<smallvec::SmallVec<[(usize, AllocatedResource); 8]>> {
        let mut allocated =
            smallvec::SmallVec::<[(usize, AllocatedResource); 8]>::with_capacity(unallocated.len());

        let mut index = (unallocated.len() as i128) - 1;
        while index >= 0 {
            let (taken_index, taken_resource) = unallocated.remove(index as usize);

            match Self::allocate_resource_on_pool(memory_pool.clone(), taken_index, taken_resource)?
            {
                ResourceAllocationResult::AllocatedBuffer((allocated_index, allocated_buffer)) => {
                    allocated.push((allocated_index, AllocatedResource::Buffer(allocated_buffer)))
                }
                ResourceAllocationResult::AllocatedImage((allocated_index, allocated_image)) => {
                    allocated.push((allocated_index, AllocatedResource::Image(allocated_image)))
                }
                ResourceAllocationResult::Unallocated(failed_resource) => {
                    unallocated.push(failed_resource);
                }
            };

            // go to the preceding element
            index -= 1;
        }

        Ok(allocated)
    }

    fn allocate_resource_on_pool(
        memory_pool: Arc<MemoryPool>,
        index: usize,
        resource: UnallocatedResource,
    ) -> VulkanResult<ResourceAllocationResult> {
        match resource {
            UnallocatedResource::Buffer(buffer) => {
                match AllocatedBuffer::new(memory_pool.clone(), buffer) {
                    Ok(allocated_buffer) => Ok(ResourceAllocationResult::AllocatedBuffer((
                        index,
                        allocated_buffer,
                    ))),
                    Err(err) => match err {
                        VulkanError::Framework(fw_err) => match fw_err {
                            FrameworkError::MallocFail(failed_resource) => Ok(
                                ResourceAllocationResult::Unallocated((index, failed_resource)),
                            ),
                            fw_err => Err(VulkanError::Framework(fw_err)),
                        },
                        _ => Err(err),
                    },
                }
            }
            UnallocatedResource::Image(image) => {
                match AllocatedImage::new(memory_pool.clone(), image) {
                    Ok(allocated_image) => Ok(ResourceAllocationResult::AllocatedImage((
                        index,
                        allocated_image,
                    ))),
                    Err(err) => match err {
                        VulkanError::Framework(fw_err) => match fw_err {
                            FrameworkError::MallocFail(failed_resource) => Ok(
                                ResourceAllocationResult::Unallocated((index, failed_resource)),
                            ),
                            fw_err => Err(VulkanError::Framework(fw_err)),
                        },
                        _ => Err(err),
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

        let memory_heap =
            match self.find_suitable_heap(memory_type, requirements, allocator.total_size()) {
                Some(memory_heap) => memory_heap,
                None => MemoryHeap::new(
                    self.device.clone(),
                    ConcreteMemoryHeapDescriptor::new(*memory_type, allocator.total_size()),
                    *requirements,
                )?,
            };

        MemoryPool::new(memory_heap, Arc::new(allocator), *features)
    }

    fn find_suitable_heap(
        &self,
        memory_type: &MemoryType,
        requirements: &MemoryRequirements,
        minimum_size: u64,
    ) -> Option<Arc<MemoryHeap>> {
        for (_, memory_pool) in self.memory_pools.iter() {
            let memory_heap = memory_pool.get_parent_memory_heap();
            if !memory_heap.suitable_memory_type(requirements) {
                continue;
            } else if memory_heap.memory_type() != *memory_type {
                continue;
            } else if memory_heap.total_size() < minimum_size {
                continue;
            }

            return Some(memory_heap);
        }

        None
    }
}

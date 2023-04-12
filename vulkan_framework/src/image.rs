use crate::{
    device::{Device, DeviceOwned},
    prelude::{VulkanError, VulkanResult}, memory_allocator::{MemoryAllocator, AllocationResult}, memory_pool::{MemoryPoolBacked, MemoryPool},
};

use std::sync::Arc;

pub enum ImageDimensions {
    Image1D { x: u32 },
    Image2D { x: u32, y: u32 },
    Image3D { x: u32, y: u32, z: u32 }
}

pub struct ConcreteImageDescriptor {
    img_dimensions: ImageDimensions
}

pub trait ImageTrait {

}

pub struct Image<Allocator>
where
    Allocator: MemoryAllocator + Send + Sync,
{
    memory_pool: Arc<MemoryPool<Allocator>>,
    reserved_memory_from_pool: AllocationResult,
}

impl<Allocator> Drop for Image<Allocator>
where
    Allocator: MemoryAllocator + Send + Sync,
{
    fn drop(&mut self) {
        self.memory_pool.dealloc(&mut self.reserved_memory_from_pool)
    }
}

impl<Allocator> MemoryPoolBacked<Allocator> for Image<Allocator>
where
    Allocator: MemoryAllocator + Send + Sync,
{
    fn get_backing_memory_pool(&self) -> Arc<MemoryPool<Allocator>> {
        self.memory_pool.clone()
    }
}
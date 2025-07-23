use std::{path::PathBuf, sync::Arc};

use crate::{
    EmbeddedAssets,
    rendering::{MAX_TEXTURES, RenderingResult, mesh::MeshManager, texture::TextureManager},
};

use vulkan_framework::{
    buffer::{AllocatedBuffer, Buffer, BufferUsage, BufferUseAs, ConcreteBufferDescriptor},
    device::{Device, DeviceOwned},
    memory_allocator::DefaultAllocator,
    memory_heap::{ConcreteMemoryHeapDescriptor, MemoryHeap, MemoryHostVisibility, MemoryType},
    memory_pool::{MemoryPool, MemoryPoolBacked, MemoryPoolFeature, MemoryPoolFeatures},
    queue::Queue,
    queue_family::{QueueFamily, QueueFamilyOwned},
};

pub struct ResourceManager {
    device: Arc<Device>,
    queue_family: Arc<QueueFamily>,

    memory_pool: Arc<MemoryPool>,
    buffer: Arc<AllocatedBuffer>,

    mesh_manager: MeshManager,
    texture_manager: TextureManager,
}

impl ResourceManager {
    fn memory_pool_size(frames_in_flight: u32) -> u64 {
        1024u64 * 1024u64 * 32u64 * (frames_in_flight as u64)
    }

    pub fn new(queue_family: Arc<QueueFamily>, frames_in_flight: u32) -> RenderingResult<Self> {
        let device = queue_family.get_parent_device();

        let total_size = Self::memory_pool_size(frames_in_flight);

        let buffer = Buffer::new(
            device.clone(),
            ConcreteBufferDescriptor::new(
                BufferUsage::from([BufferUseAs::TransferSrc].as_slice()),
                1024u64 * 1024u64 * 24u64, // 24MiB
            ),
            None,
            Some("resource_management_buffer"),
        )?;

        let memory_heap = MemoryHeap::new(
            device.clone(),
            ConcreteMemoryHeapDescriptor::new(
                MemoryType::DeviceLocal(Some(MemoryHostVisibility::MemoryHostVisibile {
                    cached: false,
                })),
                total_size,
            ),
            &[&buffer],
        )?;

        let memory_pool = MemoryPool::new(
            memory_heap,
            Arc::new(DefaultAllocator::new(total_size)),
            MemoryPoolFeatures::from([MemoryPoolFeature::DeviceAddressable {}].as_slice()),
        )?;

        let buffer = AllocatedBuffer::new(memory_pool.clone(), buffer)?;

        memory_pool.write_raw_data(
            buffer.allocation_offset(),
            EmbeddedAssets::get("stub.dds").unwrap().data.as_ref(),
        )?;

        let mesh_manager = MeshManager::new(device.clone(), frames_in_flight)?;
        let texture_manager = TextureManager::new(
            queue_family.clone(),
            buffer.clone(),
            MAX_TEXTURES,
            frames_in_flight,
        )?;

        Ok(Self {
            device,
            queue_family,

            memory_pool,
            buffer,

            mesh_manager,
            texture_manager,
        })
    }

    pub fn load_object(&mut self, file: PathBuf) -> RenderingResult<()> {
        if !file.exists() {
            panic!("File doesn't exists!");
        }

        if !file.is_file() {
            panic!("Not a file");
        }

        Ok(())
    }
}

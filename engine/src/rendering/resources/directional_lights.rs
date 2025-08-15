use std::sync::{Arc, Mutex};

use vulkan_framework::{
    buffer::{
        AllocatedBuffer, Buffer, BufferSubresourceRange, BufferTrait, BufferUseAs,
        ConcreteBufferDescriptor,
    },
    device::DeviceOwned,
    memory_barriers::{BufferMemoryBarrier, MemoryAccessAs},
    memory_heap::MemoryType,
    memory_management::{MemoryManagementTagSize, MemoryManagementTags, MemoryManagerTrait},
    memory_pool::MemoryPoolFeatures,
    pipeline_stage::PipelineStage,
    queue::Queue,
    queue_family::{QueueFamily, QueueFamilyOwned},
};

use crate::{
    core::lights::directional::DirectionalLight,
    rendering::{
        MAX_DIRECTIONAL_LIGHTS, RenderingError, RenderingResult,
        resources::collection::LoadableResourcesCollection,
    },
};

pub struct DirectionalLights {
    debug_name: String,

    queue: Arc<Queue>,

    memory_manager: Arc<Mutex<dyn MemoryManagerTrait>>,

    lights: LoadableResourcesCollection<Arc<AllocatedBuffer>>,
}

impl DirectionalLights {
    pub fn new(
        queue_family: Arc<QueueFamily>,
        memory_manager: Arc<Mutex<dyn MemoryManagerTrait>>,
        debug_name: String,
    ) -> RenderingResult<Self> {
        let queue = Queue::new(queue_family.clone(), Some("texture_manager.queue"))?;

        let lights = LoadableResourcesCollection::new(
            queue_family,
            MAX_DIRECTIONAL_LIGHTS,
            String::from("texture_manager"),
        )?;

        Ok(Self {
            debug_name,

            queue,

            memory_manager,

            lights,
        })
    }

    pub fn load(&mut self, light: DirectionalLight) -> RenderingResult<u32> {
        assert_eq!(std::mem::size_of_val(&light), 4_usize * 6_usize);

        let Some(light_index) = self.lights.load(
            self.queue.clone(),
            |index| {
                let light_buffer = Buffer::new(
                    self.queue.get_parent_queue_family().get_parent_device(),
                    ConcreteBufferDescriptor::new(
                        [BufferUseAs::TransferSrc, BufferUseAs::TransferDst]
                            .as_slice()
                            .into(),
                        4u64 * 6u64,
                    ),
                    None,
                    Some(format!("{}->light[{index}]", self.debug_name).as_str()),
                )?;

                let mut mem_manager = self.memory_manager.lock().unwrap();
                mem_manager
                    .allocate_resources(
                        &MemoryType::device_local(),
                        &MemoryPoolFeatures::new(false),
                        vec![light_buffer.into()],
                        MemoryManagementTags::default()
                            .with_name("DirectionalLights".to_string())
                            .with_size(MemoryManagementTagSize::Small),
                    )
                    .map(|allocated| allocated[0].buffer())
                    .map_err(|err| err.into())
            },
            |recorder, _, buffer| {
                recorder.buffer_barriers(
                    [BufferMemoryBarrier::new(
                        [].as_slice().into(),
                        [].as_slice().into(),
                        [PipelineStage::Transfer].as_slice().into(),
                        [MemoryAccessAs::TransferWrite].as_slice().into(),
                        BufferSubresourceRange::new(
                            buffer.clone() as Arc<dyn BufferTrait>,
                            0,
                            4 * 6,
                        ),
                        self.queue.get_parent_queue_family(),
                        self.queue.get_parent_queue_family(),
                    )]
                    .as_slice(),
                );

                recorder.update_buffer(
                    buffer.clone() as Arc<dyn BufferTrait>,
                    0,
                    [light.direction(), light.albedo()].as_slice(),
                );

                recorder.buffer_barriers(
                    [BufferMemoryBarrier::new(
                        [PipelineStage::Transfer].as_slice().into(),
                        [MemoryAccessAs::TransferWrite].as_slice().into(),
                        [PipelineStage::BottomOfPipe].as_slice().into(),
                        [].as_slice().into(),
                        BufferSubresourceRange::new(
                            buffer.clone() as Arc<dyn BufferTrait>,
                            0,
                            4 * 6,
                        ),
                        self.queue.get_parent_queue_family(),
                        self.queue.get_parent_queue_family(),
                    )]
                    .as_slice(),
                );

                Ok(())
            },
        )?
        else {
            return Err(RenderingError::ResourceError(
                super::ResourceError::NoDirectionalLightingSlotAvailable,
            ));
        };

        Ok(light_index)
    }

    pub fn count(&self) -> u32 {
        let mut count = 0u32;
        self.foreach(|_| count += 1);

        count
    }

    pub fn foreach<F>(&self, fun: F)
    where
        F: FnMut(&Arc<AllocatedBuffer>),
    {
        self.lights.foreach_loaded_mut(fun);
    }

    #[inline]
    pub fn wait_blocking(&mut self) -> RenderingResult<()> {
        self.lights.wait_load_blocking()?;

        Ok(())
    }

    #[inline]
    pub fn wait_nonblocking(&mut self) -> RenderingResult<()> {
        self.lights.wait_load_nonblock()?;

        Ok(())
    }
}

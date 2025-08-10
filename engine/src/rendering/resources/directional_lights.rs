use std::sync::{Arc, Mutex, atomic::AtomicU64};

use vulkan_framework::{
    buffer::AllocatedBuffer, memory_management::MemoryManagerTrait, queue::Queue,
    queue_family::QueueFamily,
};

use crate::rendering::{
    MAX_DIRECTIONAL_LIGHTS, MAX_FRAMES_IN_FLIGHT_NO_MALLOC, RenderingResult,
    resources::collection::LoadableResourcesCollection,
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

    pub fn load(&mut self) -> RenderingResult<usize> {
        Ok(todo!())
    }

    pub fn count(&self) -> u32 {
        let mut count = 0u32;
        self.foreach(|_| {
            count += 1
        });

        count
    }

    pub fn foreach<F>(&self, fun: F)
    where
        F: FnMut(&Arc<AllocatedBuffer>),
    {
        self.lights.foreach_loaded_mut(fun);
    }
}

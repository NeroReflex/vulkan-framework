use std::sync::Arc;

use vulkan_framework::{
    command_buffer::{CommandBufferRecorder, PrimaryCommandBuffer},
    command_pool::CommandPool,
    device::DeviceOwned,
    fence::{Fence, FenceWaiter},
    queue::Queue,
    queue_family::QueueFamily,
};

use crate::rendering::RenderingResult;

enum LoadableResource<T> {
    Free(Arc<Fence>, Arc<PrimaryCommandBuffer>),
    Loaded(T),
    Loading(T, FenceWaiter),
}

type LoadableResourcesCollectionType<T> = smallvec::SmallVec<[LoadableResource<T>; 128]>;

pub struct LoadableResourcesCollection<T>
where
    T: Clone,
{
    debug_name: String,
    command_pool: Arc<CommandPool>,
    collection: LoadableResourcesCollectionType<T>,
    status: u64,
}

impl<T> LoadableResourcesCollection<T>
where
    T: Clone,
{
    #[inline]
    pub(crate) fn status(&self) -> u64 {
        self.status.to_owned()
    }

    pub(crate) fn wait_load_nonblock(&mut self) -> RenderingResult<usize> {
        let mut loaded = 0;

        for index in 0..self.collection.len() {
            let resource = match &self.collection[index] {
                LoadableResource::Loading(resource, fence_waiter) => {
                    // if the resource is still loading leave it alone
                    if !fence_waiter.complete()? {
                        continue;
                    }

                    resource
                }
                _ => continue,
            };

            self.collection[index] = LoadableResource::Loaded(resource.clone());
            loaded += 1;
        }

        if loaded > 0 {
            self.status += 1;
        }

        Ok(loaded)
    }

    pub(crate) fn load<CreateFn, LoadFn>(
        &mut self,
        queue: Arc<Queue>,
        creation_fun: CreateFn,
        loading_fun: LoadFn,
    ) -> RenderingResult<Option<u32>>
    where
        CreateFn: FnOnce() -> RenderingResult<T>,
        LoadFn: FnOnce(&mut CommandBufferRecorder, T) -> RenderingResult<()>,
    {
        for index in 0..self.collection.len() {
            // ensure this is an available slot for the new texture
            let (fence, command_buffer) = match &self.collection[index] {
                LoadableResource::Free(fence, command_buffer) => (fence, command_buffer),
                _ => continue,
            };

            let resource = creation_fun()?;

            command_buffer
                .record_one_time_submit(|recorder| loading_fun(recorder, resource.clone()))??;

            let fence_waiter = queue.submit(&[command_buffer.clone()], &[], &[], fence.clone())?;

            self.collection[index] = LoadableResource::Loading(resource, fence_waiter);

            // the image has been created
            return Ok(Some(index as u32));
        }

        Ok(None)
    }

    pub fn new(
        queue_family: Arc<QueueFamily>,
        max_elements: u32,
        debug_name: String,
    ) -> RenderingResult<Self> {
        let device = queue_family.get_parent_device();

        let command_pool = CommandPool::new(
            queue_family.clone(),
            Some(format!("{debug_name}.command_pool").as_str()),
        )?;

        let mut collection: LoadableResourcesCollectionType<T> =
            smallvec::SmallVec::with_capacity(max_elements as usize);
        for index in 0..max_elements as usize {
            collection.push(LoadableResource::Free(
                Fence::new(device.clone(), false, None)?,
                PrimaryCommandBuffer::new(
                    command_pool.clone(),
                    Some(format!("{debug_name}.command_buffers[{index}]").as_str()),
                )?,
            ));
        }

        let status = 0u64;

        Ok(Self {
            debug_name,
            command_pool,
            collection,
            status,
        })
    }
}

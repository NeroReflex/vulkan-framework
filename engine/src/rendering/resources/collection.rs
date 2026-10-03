use std::{sync::Arc, time::Duration};

use vulkan_framework::{
    command_buffer::{CommandBufferRecorder, PrimaryCommandBuffer},
    command_pool::CommandPool,
    device::DeviceOwned,
    fence::{Fence, FenceWaitFor, FenceWaiter},
    pipeline_stage::PipelineStage,
    queue::{Queue, SemaphoreSignalOp, SemaphoreWaitOp},
    queue_family::{QueueFamily, QueueFamilyOwned},
    semaphore::Semaphore,
};

use crate::rendering::{RenderingError, RenderingResult};

use super::ResourceError;

pub(super) fn wait_for_upload(fence: &Arc<Fence>) -> RenderingResult<()> {
    Fence::wait_for_fences(
        &[fence.clone()],
        FenceWaitFor::All,
        Duration::from_nanos(u64::MAX),
    )?;
    Ok(())
}

struct PendingUpload<T> {
    // Wait before destroying resources, including AS objects/scratch allocations
    // that the framework recorder does not retain itself.
    _waiter: FenceWaiter,
    fence: Arc<Fence>,
    resource: T,
}

enum LoadableResource<T> {
    Free,
    Loaded(T),
    Loading(PendingUpload<T>),
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
    upload_timeline: Arc<Semaphore>,
    submitted_uploads: u64,
}

impl<T> LoadableResourcesCollection<T>
where
    T: Clone,
{
    pub fn size(&self) -> usize {
        self.collection.len()
    }

    #[inline]
    pub fn fetch_loaded(&self, index: usize) -> Option<&T> {
        match self.collection.get(index)? {
            LoadableResource::Loaded(value) => Some(value),
            _ => None,
        }
    }

    #[inline]
    pub fn foreach_loaded_mut<F>(&self, mut function: F)
    where
        F: FnMut(&T),
    {
        for obj in &self.collection {
            if let LoadableResource::Loaded(loaded_obj) = obj {
                function(loaded_obj);
            }
        }
    }

    #[inline]
    pub fn foreach_loaded<F>(&self, function: F)
    where
        F: Fn(&T),
    {
        self.foreach_loaded_mut(function);
    }

    #[inline]
    pub(crate) fn status(&self) -> u64 {
        self.status
    }

    /// Include this wait in every consuming submission, even after a host wait.
    /// It publishes upload writes and image transitions to other queues in the
    /// same family. This collection does not implement queue-family transfers.
    pub(crate) fn upload_wait(&self) -> Option<SemaphoreWaitOp> {
        (self.submitted_uploads > 0).then(|| {
            SemaphoreWaitOp::Timeline(
                [PipelineStage::AllCommands].as_slice().into(),
                self.upload_timeline.clone(),
                self.submitted_uploads,
            )
        })
    }

    pub(crate) fn remove(&mut self, index: u32) -> RenderingResult<()> {
        let resource = self.collection.get_mut(index as usize).ok_or_else(|| {
            RenderingError::ResourceError(ResourceError::ResourceIndexOutOfRange(index as usize))
        })?;
        if let LoadableResource::Loading(upload) = resource {
            wait_for_upload(&upload.fence)?;
        }
        if !matches!(resource, LoadableResource::Free) {
            *resource = LoadableResource::Free;
            self.status += 1;
        }
        Ok(())
    }

    pub(crate) fn wait_load_blocking(&mut self) -> RenderingResult<usize> {
        let mut loaded = 0;
        for resource in &mut self.collection {
            let LoadableResource::Loading(upload) = resource else {
                continue;
            };
            // Do not publish readiness or discard the pending state on error.
            wait_for_upload(&upload.fence)?;
            *resource = LoadableResource::Loaded(upload.resource.clone());
            self.status += 1;
            loaded += 1;
        }
        Ok(loaded)
    }

    pub(crate) fn wait_load_nonblock(&mut self) -> RenderingResult<usize> {
        let mut loaded = 0;
        for resource in &mut self.collection {
            let LoadableResource::Loading(upload) = resource else {
                continue;
            };
            if !upload.fence.is_signaled()? {
                continue;
            }
            *resource = LoadableResource::Loaded(upload.resource.clone());
            self.status += 1;
            loaded += 1;
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
        CreateFn: FnOnce(usize) -> RenderingResult<T>,
        LoadFn: FnOnce(&mut CommandBufferRecorder, usize, T) -> RenderingResult<()>,
    {
        let Some(index) = self
            .collection
            .iter()
            .position(|resource| matches!(resource, LoadableResource::Free))
        else {
            return Ok(None);
        };
        self.upload_at(index, queue, creation_fun, loading_fun)?;
        Ok(Some(index as u32))
    }

    /// Submit an immutable replacement; preserve the previous resource on error.
    pub(crate) fn replace<CreateFn, LoadFn>(
        &mut self,
        index: u32,
        queue: Arc<Queue>,
        creation_fun: CreateFn,
        loading_fun: LoadFn,
    ) -> RenderingResult<()>
    where
        CreateFn: FnOnce(usize) -> RenderingResult<T>,
        LoadFn: FnOnce(&mut CommandBufferRecorder, usize, T) -> RenderingResult<()>,
    {
        match self.collection.get(index as usize) {
            Some(LoadableResource::Loaded(_) | LoadableResource::Loading(_)) => {}
            _ => {
                return Err(RenderingError::ResourceError(
                    ResourceError::ResourceIndexOutOfRange(index as usize),
                ));
            }
        }
        self.upload_at(index as usize, queue, creation_fun, loading_fun)
    }

    fn upload_at<CreateFn, LoadFn>(
        &mut self,
        index: usize,
        queue: Arc<Queue>,
        creation_fun: CreateFn,
        loading_fun: LoadFn,
    ) -> RenderingResult<()>
    where
        CreateFn: FnOnce(usize) -> RenderingResult<T>,
        LoadFn: FnOnce(&mut CommandBufferRecorder, usize, T) -> RenderingResult<()>,
    {
        let family = self.command_pool.get_parent_queue_family();
        assert!(Arc::ptr_eq(&family, &queue.get_parent_queue_family()));
        assert_eq!(
            family.get_parent_device().native_handle(),
            queue
                .get_parent_queue_family()
                .get_parent_device()
                .native_handle()
        );
        let next_upload = self
            .submitted_uploads
            .checked_add(1)
            .ok_or_else(|| RenderingError::ResourceError(ResourceError::UploadCounterExhausted))?;
        let fence = Fence::new(
            family.get_parent_device(),
            false,
            Some(format!("{}.fence[{index}]", self.debug_name).as_str()),
        )?;
        let command_buffer = PrimaryCommandBuffer::new(
            self.command_pool.clone(),
            Some(format!("{}.command_buffer[{index}]", self.debug_name).as_str()),
        )?;
        let resource = creation_fun(index)?;
        command_buffer
            .record_one_time_submit(|recorder| loading_fun(recorder, index, resource.clone()))??;
        let waits: Vec<_> = self.upload_wait().into_iter().collect();
        let waiter = queue.submit_mixed(
            &[command_buffer],
            &waits,
            &[SemaphoreSignalOp::Timeline(
                self.upload_timeline.clone(),
                next_upload,
            )],
            fence.clone(),
        )?;
        self.submitted_uploads = next_upload;
        self.collection[index] = LoadableResource::Loading(PendingUpload {
            _waiter: waiter,
            fence,
            resource,
        });
        Ok(())
    }

    pub fn new(
        queue_family: Arc<QueueFamily>,
        max_elements: u32,
        debug_name: String,
    ) -> RenderingResult<Self> {
        let upload_timeline = Semaphore::new_timeline(
            queue_family.get_parent_device(),
            0,
            Some(format!("{debug_name}.upload_timeline").as_str()),
        )?;
        let command_pool = CommandPool::new(
            queue_family,
            Some(format!("{debug_name}.command_pool").as_str()),
        )?;
        let collection = (0..max_elements).map(|_| LoadableResource::Free).collect();
        Ok(Self {
            debug_name,
            command_pool,
            collection,
            status: 0,
            upload_timeline,
            submitted_uploads: 0,
        })
    }
}

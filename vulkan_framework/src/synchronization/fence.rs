use std::{
    future::Future,
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
    time::Duration,
};

use crate::{
    command_buffer::CommandBufferTrait,
    fence::Fence,
    pipeline_stage::PipelineStages,
    prelude::{VulkanError, VulkanResult},
    queue::Queue,
    semaphore::Semaphore,
};

use super::thread::ThreadPool;

pub struct SpinlockFenceWaiter<T>
where
    T: Sized + Copy,
{
    result: T,
    fence: Option<Arc<Fence>>,
    queue: Option<Arc<Queue>>,
    semaphores: smallvec::SmallVec<[Arc<Semaphore>; 16]>,
    command_buffers: smallvec::SmallVec<[Arc<dyn CommandBufferTrait>; 8]>,
}

impl<T> Drop for SpinlockFenceWaiter<T>
where
    T: Sized + Copy,
{
    fn drop(&mut self) {
        if let Some(_fence) = &self.fence {
            panic!("Nooooooo you still have resources in use, please wait for the fence!!! :(");
        }
    }
}

impl<T> SpinlockFenceWaiter<T>
where
    T: Sized + Copy,
{
    pub fn empty(result: T) -> Self {
        Self {
            result,
            queue: None,
            fence: None,
            semaphores: smallvec::smallvec![],
            command_buffers: smallvec::smallvec![],
        }
    }

    pub fn new(
        queue: Option<Arc<Queue>>,
        command_buffers: &[Arc<dyn CommandBufferTrait>],
        semaphores: &[Arc<Semaphore>],
        fence: Arc<Fence>,
        result: T,
    ) -> Self {
        Self {
            result,
            fence: Some(fence),
            queue: queue,
            semaphores: semaphores
                .iter()
                .cloned()
                .collect::<smallvec::SmallVec<[Arc<Semaphore>; 16]>>(),
            command_buffers: command_buffers
                .iter()
                .cloned()
                .collect::<smallvec::SmallVec<[Arc<dyn CommandBufferTrait>; 8]>>(),
        }
    }

    pub fn new_by_submit(
        result: T,
        queue: Arc<Queue>,
        command_buffers: &[Arc<dyn CommandBufferTrait>],
        wait_semaphores: &[(PipelineStages, Arc<Semaphore>)],
        signal_semaphores: &[Arc<Semaphore>],
        fence: Arc<Fence>,
    ) -> VulkanResult<Self> {
        match queue.submit(
            command_buffers,
            wait_semaphores,
            signal_semaphores,
            fence.clone(),
        ) {
            Ok(()) => {
                let mut semaphores: smallvec::SmallVec<[Arc<Semaphore>; 16]> =
                    smallvec::smallvec![];

                for s in signal_semaphores.iter() {
                    semaphores.push(s.clone())
                }

                for s in wait_semaphores {
                    semaphores.push(s.1.clone())
                }

                Ok(Self::new(
                    Some(queue),
                    command_buffers,
                    semaphores.as_slice(),
                    fence,
                    result,
                ))
            }
            Err(err) => Err(err),
        }
    }

    /*
        pub fn from_fence(fence: Arc<Fence>) -> Self {
            Self {
                fence: Some(fence),
                queue: None,
                command_buffers: smallvec::smallvec![],
            }
        }
    */
}

impl<T> Future for SpinlockFenceWaiter<T>
where
    T: Sized + Copy,
{
    type Output = VulkanResult<T>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        match &self.fence {
            Some(fence) => {
                // Vulkan only allows polling of the fence status and not a callback-wake feature, so we have to use a spin future.
                // This is still better than blocking in async applications, since a smart-enough async engine
                // can choose to run some other tasks between probing this one.

                // Check if we are done without blocking
                match fence.is_signaled() {
                    Err(e) => return Poll::Ready(Err(e)),
                    Ok(status) => {
                        if status {
                            return match fence.reset() {
                                Ok(()) => {
                                    // here I am gonna destroy the fence and the list of occupied resources so that they can finally be free
                                    for cmd_buffer in self.command_buffers.iter() {
                                        cmd_buffer.flag_execution_as_finished();
                                    }

                                    let result = unsafe {
                                        let self_mut_ref = self.get_unchecked_mut();

                                        self_mut_ref.fence = None;
                                        self_mut_ref.queue = None;
                                        self_mut_ref.command_buffers.clear();
                                        self_mut_ref.semaphores.clear();

                                        self_mut_ref.result
                                    };

                                    Poll::Ready(Ok(result))
                                }
                                Err(err) => Poll::Ready(Err(err)),
                            };
                        }
                    }
                }

                // Otherwise spin
                cx.waker().wake_by_ref();
                Poll::Pending
            }
            None => Poll::Ready(Ok(self.result)),
        }
    }
}

// ====================================================================================================

pub struct ThreadedFenceWaiter<T>
where
    T: Sized + Copy,
{
    pool: Arc<ThreadPool>,
    result: T,
    fence: Option<Arc<Fence>>,
    queue: Option<Arc<Queue>>,
    semaphores: smallvec::SmallVec<[Arc<Semaphore>; 16]>,
    command_buffers: smallvec::SmallVec<[Arc<dyn CommandBufferTrait>; 8]>,
}

impl<T> Drop for ThreadedFenceWaiter<T>
where
    T: Sized + Copy,
{
    fn drop(&mut self) {
        if let Some(_fence) = &self.fence {
            panic!("Nooooooo you still have resources in use, please wait for the fence!!! :(");
        }
    }
}

impl<T> ThreadedFenceWaiter<T>
where
    T: Sized + Copy,
{
    pub fn empty(pool: Arc<ThreadPool>, result: T) -> Self {
        Self {
            pool,
            result,
            queue: None,
            fence: None,
            semaphores: smallvec::smallvec![],
            command_buffers: smallvec::smallvec![],
        }
    }

    pub fn new(
        pool: Arc<ThreadPool>,
        queue: Option<Arc<Queue>>,
        command_buffers: &[Arc<dyn CommandBufferTrait>],
        semaphores: &[Arc<Semaphore>],
        fence: Arc<Fence>,
        result: T,
    ) -> Self {
        Self {
            pool,
            result,
            fence: Some(fence),
            queue: queue,
            semaphores: semaphores
                .iter()
                .cloned()
                .collect::<smallvec::SmallVec<[Arc<Semaphore>; 16]>>(),
            command_buffers: command_buffers
                .iter()
                .cloned()
                .collect::<smallvec::SmallVec<[Arc<dyn CommandBufferTrait>; 8]>>(),
        }
    }

    pub fn new_by_submit(
        pool: Arc<ThreadPool>,
        result: T,
        queue: Arc<Queue>,
        command_buffers: &[Arc<dyn CommandBufferTrait>],
        wait_semaphores: &[(PipelineStages, Arc<Semaphore>)],
        signal_semaphores: &[Arc<Semaphore>],
        fence: Arc<Fence>,
    ) -> VulkanResult<Self> {
        match queue.submit(
            command_buffers,
            wait_semaphores,
            signal_semaphores,
            fence.clone(),
        ) {
            Ok(()) => {
                let mut semaphores: smallvec::SmallVec<[Arc<Semaphore>; 16]> =
                    smallvec::smallvec![];

                for s in signal_semaphores.iter() {
                    semaphores.push(s.clone())
                }

                for s in wait_semaphores {
                    semaphores.push(s.1.clone())
                }

                Ok(Self::new(
                    pool,
                    Some(queue),
                    command_buffers,
                    semaphores.as_slice(),
                    fence,
                    result,
                ))
            }
            Err(err) => Err(err),
        }
    }

    /*
        pub fn from_fence(fence: Arc<Fence>) -> Self {
            Self {
                fence: Some(fence),
                queue: None,
                command_buffers: smallvec::smallvec![],
            }
        }
    */
}

impl<T> Future for ThreadedFenceWaiter<T>
where
    T: Sized + Copy,
{
    type Output = VulkanResult<T>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        match &self.fence {
            Some(fence) => {
                // Vulkan only allows polling of the fence status and not a callback-wake feature, so we use a thread-waiting future.
                // This will have another thread wait for the fence while the user code can continue executing, and since the waiting thread
                // is derived from a pool the cost of the spawn is 0 and the only downside is the operation of telling one of those thread to execute
                // the blocking wait

                // Check if we are done without blocking
                match fence.is_signaled() {
                    Err(e) => return Poll::Ready(Err(e)),
                    Ok(status) => {
                        if status {
                            return match fence.reset() {
                                Ok(()) => {
                                    // here I am gonna destroy the fence and the list of occupied resources so that they can finally be free
                                    for cmd_buffer in self.command_buffers.iter() {
                                        cmd_buffer.flag_execution_as_finished();
                                    }

                                    let result = unsafe {
                                        let self_mut_ref = self.get_unchecked_mut();

                                        self_mut_ref.fence = None;
                                        self_mut_ref.queue = None;
                                        self_mut_ref.command_buffers.clear();
                                        self_mut_ref.semaphores.clear();

                                        self_mut_ref.result
                                    };

                                    Poll::Ready(Ok(result))
                                }
                                Err(err) => Poll::Ready(Err(err)),
                            };
                        }
                    }
                }

                // Otherwise spin
                let waker = cx.waker().clone();

                let fences_to_wait = [fence.clone()];
                self.pool.execute_retry(move || {
                    let waker_clone = waker.clone();

                    match Fence::wait_for_fences(
                        fences_to_wait.as_slice(),
                        crate::fence::FenceWaitFor::All,
                        Duration::from_nanos(100),
                    ) {
                        Ok(_) => {
                            waker_clone.clone().wake_by_ref();

                            true
                        }
                        Err(err) => {
                            // if the error is a timeout error schedule re-calling by returning false
                            if VulkanError::is_timeout(&err) {
                                return false;
                            }

                            true
                        }
                    }
                });

                Poll::Pending
            }
            None => Poll::Ready(Ok(self.result)),
        }
    }
}

use std::{
    future::Future,
    pin::Pin,
    sync::Arc,
    task::{Context, Poll}
};

use crate::{
    command_buffer::CommandBufferTrait,
    fence::{Fence},
    pipeline_stage::PipelineStages,
    prelude::VulkanResult,
    queue::Queue,
    semaphore::Semaphore,
};

pub struct SpinlockFenceWaiter<T>
where
    T: Sized + Copy
{

    result: T,
    fence: Option<Arc<Fence>>,
    queue: Option<Arc<Queue>>,
    command_buffers: smallvec::SmallVec<[Arc<dyn CommandBufferTrait>; 8]>,
}

impl<T> Drop for SpinlockFenceWaiter<T>
where
    T: Sized + Copy 
{
    fn drop(&mut self) {
        if let Some(_fence) = &self.fence {
            panic!("Nooooooo you still have resources in use, please wait for the fence!!! :(");
        }
    }
}

impl<T> SpinlockFenceWaiter<T>
where
    T: Sized + Copy 
{
    pub fn empty(result: T) -> Self {
        Self {
            result,
            queue: None,
            fence: None,
            command_buffers: smallvec::smallvec![],
        }
    }

    pub fn new(
        queue: Option<Arc<Queue>>,
        command_buffers: &[Arc<dyn CommandBufferTrait>],
        fence: Arc<Fence>,
        result: T
    ) -> Self {
        Self {
            result,
            fence: Some(fence),
            queue: queue,
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
            Ok(()) => Ok(Self {
                result,
                fence: Some(fence),
                queue: Some(queue),
                command_buffers: command_buffers
                    .iter()
                    .cloned()
                    .collect::<smallvec::SmallVec<[Arc<dyn CommandBufferTrait>; 8]>>(),
            }),
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
    T: Sized + Copy 
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

                                    let result =
                                    unsafe {
                                        let self_mut_ref = self.get_unchecked_mut();

                                        self_mut_ref.fence = None;
                                        self_mut_ref.queue = None;

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

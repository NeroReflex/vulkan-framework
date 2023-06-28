use std::{
    future::Future,
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
    time::Duration,
};

use crate::{
    command_buffer::CommandBufferTrait,
    fence::{Fence, FenceWaitFor},
    pipeline_stage::PipelineStages,
    prelude::VulkanResult,
    queue::Queue,
    semaphore::Semaphore,
};

pub struct FenceWaiter {
    fence: Option<Arc<Fence>>,
    queue: Option<Arc<Queue>>,
    command_buffers: smallvec::SmallVec<[Arc<dyn CommandBufferTrait>; 8]>,
}

impl Drop for FenceWaiter {
    fn drop(&mut self) {
        if let Some(_fence) = &self.fence {
            panic!("Nooooooo you still have resources in use, please wait for the fence!!! :(");
        }
    }
}

impl FenceWaiter {
    pub fn empty() -> Self {
        Self {
            queue: None,
            fence: None,
            command_buffers: smallvec::smallvec![],
        }
    }

    pub fn new_by_submit(
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

    pub fn from_fence(fence: Arc<Fence>) -> Self {
        Self {
            fence: Some(fence),
            queue: None,
            command_buffers: smallvec::smallvec![],
        }
    }

    pub fn wait(&mut self, device_timeout: Duration) -> VulkanResult<()> {
        if let Some(fence) = &self.fence {
            let fence_arr = [fence.clone()];

            match Fence::wait_for_fences(fence_arr.as_slice(), FenceWaitFor::All, device_timeout) {
                Ok(_) => {
                    match fence.reset() {
                        Ok(()) => {
                            // here I am gonna destroy the fence and the list of occupied resources so that they can finally be free
                            for cmd_buffer in self.command_buffers.iter() {
                                cmd_buffer.flag_execution_as_finished();
                            }
                            self.fence = None;
                            self.queue = None;

                            Ok(())
                        }
                        Err(err) => Err(err),
                    }
                }
                Err(err) => Err(err),
            }
        } else {
            // If this is empty then a successful wait operation has already been completed, therefore there is nothing to do.
            // It is safe to return another Ok() as bound resources have been already freed
            Ok(())
        }
    }
}

impl Future for FenceWaiter {
    type Output = VulkanResult<()>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        match &self.fence {
            Some(fence) => {
                // Vulkan only allows polling of the fence status, so we have to use a spin future.
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
                                    unsafe {
                                        let self_mut_ref = self.get_unchecked_mut();

                                        self_mut_ref.fence = None;
                                        self_mut_ref.queue = None;
                                    }

                                    Poll::Ready(Ok(()))
                                }
                                Err(err) => Poll::Ready(Err(err)),
                            };
                        }
                    }
                }

                // Otherwise spin
                //cx.waker().wake_by_ref();
                Poll::Pending
            }
            None => Poll::Ready(Ok(())),
        }
    }
}

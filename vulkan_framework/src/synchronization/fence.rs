use std::{
    future::Future,
    pin::Pin,
    sync::{Arc, Mutex},
    task::{Context, Poll, Waker},
    time::Duration,
};

use crate::{
    command_buffer::CommandBufferTrait,
    fence::{Fence, FenceWaitFor, FenceWaiter},
    pipeline_stage::PipelineStages,
    prelude::{VulkanError, VulkanResult},
    queue::Queue,
    semaphore::Semaphore,
};

use super::thread::ThreadPool;

// Queue submissions must retain the original token: it owns the fence reservation,
// rollback/completion semantics, command buffers and semaphores.
enum PendingFence {
    Submitted {
        fence: Arc<Fence>,
        _waiter: FenceWaiter,
        _queue: Arc<Queue>,
    },
    External(ExternalFence),
}

impl PendingFence {
    fn fence(&self) -> &Arc<Fence> {
        match self {
            Self::Submitted { fence, .. } => fence,
            Self::External(external) => &external.fence,
        }
    }
}

struct ExternalFence {
    fence: Arc<Fence>,
    _queue: Option<Arc<Queue>>,
    _semaphores: smallvec::SmallVec<[Arc<Semaphore>; 16]>,
    command_buffers: smallvec::SmallVec<[Arc<dyn CommandBufferTrait>; 8]>,
}

impl ExternalFence {
    fn new(
        queue: Option<Arc<Queue>>,
        command_buffers: &[Arc<dyn CommandBufferTrait>],
        semaphores: &[Arc<Semaphore>],
        fence: Arc<Fence>,
    ) -> Self {
        Self {
            fence,
            _queue: queue,
            _semaphores: semaphores.iter().cloned().collect(),
            command_buffers: command_buffers.iter().cloned().collect(),
        }
    }
}

impl Drop for ExternalFence {
    fn drop(&mut self) {
        loop {
            match Fence::wait_for_fences(
                std::slice::from_ref(&self.fence),
                FenceWaitFor::All,
                Duration::from_nanos(u64::MAX),
            ) {
                Ok(()) => break,
                Err(err) if err.is_timeout() => continue,
                Err(err) => panic!("Error while waiting for external fence: {err:?}"),
            }
        }
        for command_buffer in &self.command_buffers {
            command_buffer.mark_execution_complete().unwrap();
        }
        self.fence.reset().unwrap();
    }
}

pub struct SpinlockFenceWaiter<T: Sized + Copy> {
    result: T,
    pending: Option<PendingFence>,
}

// No field is structurally pinned; T is returned by value, never projected as pinned.
impl<T: Sized + Copy> Unpin for SpinlockFenceWaiter<T> {}

impl<T: Sized + Copy> SpinlockFenceWaiter<T> {
    pub fn empty(result: T) -> Self {
        Self {
            result,
            pending: None,
        }
    }

    /// Wait for externally initiated work (for example swapchain acquisition).
    /// The caller must exclusively transfer completion/reset responsibility and
    /// already-running command buffers. Do not use this with a Queue::submit token;
    /// use new_by_submit instead. Dropping an unfinished waiter blocks for safety.
    pub fn new(
        queue: Option<Arc<Queue>>,
        command_buffers: &[Arc<dyn CommandBufferTrait>],
        semaphores: &[Arc<Semaphore>],
        fence: Arc<Fence>,
        result: T,
    ) -> Self {
        Self {
            result,
            pending: Some(PendingFence::External(ExternalFence::new(
                queue,
                command_buffers,
                semaphores,
                fence,
            ))),
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
        let waiter = queue.submit(
            command_buffers,
            wait_semaphores,
            signal_semaphores,
            fence.clone(),
        )?;
        Ok(Self {
            result,
            pending: Some(PendingFence::Submitted {
                fence,
                _waiter: waiter,
                _queue: queue,
            }),
        })
    }
}

impl<T: Sized + Copy> Future for SpinlockFenceWaiter<T> {
    type Output = VulkanResult<T>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        if let Some(pending) = &this.pending {
            match pending.fence().is_signaled() {
                Err(err) => return Poll::Ready(Err(err)),
                Ok(false) => {
                    cx.waker().wake_by_ref();
                    return Poll::Pending;
                }
                Ok(true) => {}
            }
            // The owning token completes the buffers and releases the reservation once.
            drop(this.pending.take());
        }
        Poll::Ready(Ok(this.result))
    }
}

#[derive(Default)]
struct ThreadedWaitState {
    scheduled: bool,
    cancelled: bool,
    error: Option<VulkanError>,
    waker: Option<Waker>,
}

pub struct ThreadedFenceWaiter<T: Sized + Copy> {
    pool: Arc<ThreadPool>,
    result: T,
    pending: Option<PendingFence>,
    state: Arc<Mutex<ThreadedWaitState>>,
}

impl<T: Sized + Copy> Unpin for ThreadedFenceWaiter<T> {}

impl<T: Sized + Copy> Drop for ThreadedFenceWaiter<T> {
    fn drop(&mut self) {
        // Serialize cancellation with the worker's native wait before the token
        // resets its fence. A queued retry must not touch a subsequently reused fence.
        let mut state = self.state.lock().unwrap();
        state.cancelled = true;
        state.waker = None;
    }
}

impl<T: Sized + Copy> ThreadedFenceWaiter<T> {
    pub fn empty(pool: Arc<ThreadPool>, result: T) -> Self {
        Self {
            pool,
            result,
            pending: None,
            state: Arc::new(Mutex::new(ThreadedWaitState::default())),
        }
    }

    /// Same exclusive external-fence ownership contract as SpinlockFenceWaiter::new.
    /// Dropping an unfinished waiter blocks until its work completes.
    pub fn new(
        pool: Arc<ThreadPool>,
        queue: Option<Arc<Queue>>,
        command_buffers: &[Arc<dyn CommandBufferTrait>],
        semaphores: &[Arc<Semaphore>],
        fence: Arc<Fence>,
        result: T,
    ) -> Self {
        let mut this = Self::empty(pool, result);
        this.pending = Some(PendingFence::External(ExternalFence::new(
            queue,
            command_buffers,
            semaphores,
            fence,
        )));
        this
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
        let waiter = queue.submit(
            command_buffers,
            wait_semaphores,
            signal_semaphores,
            fence.clone(),
        )?;
        let mut this = Self::empty(pool, result);
        this.pending = Some(PendingFence::Submitted {
            fence,
            _waiter: waiter,
            _queue: queue,
        });
        Ok(this)
    }
}

impl<T: Sized + Copy> Future for ThreadedFenceWaiter<T> {
    type Output = VulkanResult<T>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        let Some(pending) = &this.pending else {
            return Poll::Ready(Ok(this.result));
        };
        let mut state = this.state.lock().unwrap();
        let status = match state.error.take() {
            Some(err) => Err(err),
            None => pending.fence().is_signaled(),
        };
        match status {
            Err(err) => {
                state.cancelled = true;
                state.waker = None;
                return Poll::Ready(Err(err));
            }
            Ok(true) => {
                state.cancelled = true;
                state.waker = None;
                drop(state);
                drop(this.pending.take());
                return Poll::Ready(Ok(this.result));
            }
            Ok(false) => {}
        }
        state.waker = Some(cx.waker().clone());
        if !state.scheduled {
            state.scheduled = true;
            let shared_state = this.state.clone();
            let fence = pending.fence().clone();
            this.pool.execute_retry(move || {
                let mut state = shared_state.lock().unwrap();
                if state.cancelled {
                    return true;
                }
                match Fence::wait_for_fences(
                    std::slice::from_ref(&fence),
                    FenceWaitFor::All,
                    Duration::from_millis(1),
                ) {
                    Err(err) if err.is_timeout() => false,
                    result => {
                        state.error = result.err();
                        state.scheduled = false;
                        let waker = state.waker.take();
                        drop(state);
                        // Wake on both completion and terminal errors, using the latest waker.
                        if let Some(waker) = waker {
                            waker.wake();
                        }
                        true
                    }
                }
            });
        }
        Poll::Pending
    }
}

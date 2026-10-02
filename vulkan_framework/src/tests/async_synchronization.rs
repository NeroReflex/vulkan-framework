use std::{
    future::Future,
    pin::Pin,
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        mpsc, Arc,
    },
    task::{Context, Poll, Wake, Waker},
    time::{Duration, Instant},
};

use crate::{
    command_buffer::{CommandBufferTrait, PrimaryCommandBuffer, SubmittableCommandBufferTrait},
    command_pool::{CommandPool, CommandPoolOwned},
    device::Device,
    fence::{Fence, FenceWaitFor},
    pipeline_stage::{PipelineStage, PipelineStages},
    prelude::VulkanResult,
    queue::{Queue, SemaphoreWaitOp},
    queue_family::QueueFamily,
    semaphore::Semaphore,
    synchronization::{
        fence::{SpinlockFenceWaiter, ThreadedFenceWaiter},
        thread::ThreadPool,
    },
};

#[derive(Default)]
struct WakeCounter(AtomicUsize);
impl Wake for WakeCounter {
    fn wake(self: Arc<Self>) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

#[test]
fn empty_async_waiters_are_ready_even_for_non_unpin_results() -> VulkanResult<()> {
    #[derive(Clone, Copy)]
    struct ResultWithPin(std::marker::PhantomPinned);
    let waker = Waker::from(Arc::new(WakeCounter::default()));
    let mut cx = Context::from_waker(&waker);
    let mut spin = SpinlockFenceWaiter::empty(ResultWithPin(std::marker::PhantomPinned));
    assert!(matches!(
        Pin::new(&mut spin).poll(&mut cx),
        Poll::Ready(Ok(_))
    ));
    let pool = ThreadPool::new(1)?;
    let mut threaded = ThreadedFenceWaiter::empty(pool, ResultWithPin(std::marker::PhantomPinned));
    assert!(matches!(
        Pin::new(&mut threaded).poll(&mut cx),
        Poll::Ready(Ok(_))
    ));
    Ok(())
}

#[test]
fn thread_pool_rejects_zero_workers() {
    assert!(ThreadPool::new(0).is_err());
}

#[test]
fn thread_pool_keeps_more_than_eight_retries_and_progresses_with_idle_workers() -> VulkanResult<()>
{
    for workers in [1, 4] {
        let pool = ThreadPool::new(workers)?;
        let release = Arc::new(AtomicBool::new(false));
        let (started_tx, started_rx) = mpsc::channel();
        let (done_tx, done_rx) = mpsc::channel();
        for _ in 0..32 {
            let release = release.clone();
            let started = AtomicBool::new(false);
            let started_tx = started_tx.clone();
            let done_tx = done_tx.clone();
            pool.execute_retry(move || {
                if !started.swap(true, Ordering::SeqCst) {
                    started_tx.send(()).unwrap();
                }
                if release.load(Ordering::SeqCst) {
                    done_tx.send(()).unwrap();
                    true
                } else {
                    false
                }
            });
        }
        for _ in 0..32 {
            started_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        }
        release.store(true, Ordering::SeqCst);
        for _ in 0..32 {
            done_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        }
        assert!(
            done_rx.try_recv().is_err(),
            "a retry must complete only once"
        );
    }
    Ok(())
}

struct CountedCommandBuffer {
    inner: Arc<PrimaryCommandBuffer>,
    completions: AtomicUsize,
    cancellations: AtomicUsize,
}
impl CountedCommandBuffer {
    fn new(pool: Arc<CommandPool>) -> VulkanResult<Arc<Self>> {
        let inner = PrimaryCommandBuffer::new(pool, None)?;
        inner.record_one_time_submit(|_| {})?;
        Ok(Arc::new(Self {
            inner,
            completions: AtomicUsize::new(0),
            cancellations: AtomicUsize::new(0),
        }))
    }
}
impl SubmittableCommandBufferTrait for CountedCommandBuffer {
    fn mark_execution_begin(&self) -> VulkanResult<()> {
        self.inner.mark_execution_begin()
    }
    fn mark_execution_complete(&self) -> VulkanResult<()> {
        self.inner.mark_execution_complete()?;
        self.completions.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
    fn mark_execution_cancel(&self) -> VulkanResult<()> {
        self.inner.mark_execution_cancel()?;
        self.cancellations.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}
impl CommandPoolOwned for CountedCommandBuffer {
    fn get_parent_command_pool(&self) -> Arc<CommandPool> {
        self.inner.get_parent_command_pool()
    }
}
impl CommandBufferTrait for CountedCommandBuffer {
    fn native_handle(&self) -> u64 {
        self.inner.native_handle()
    }
}

type TestFuture = Pin<Box<dyn Future<Output = VulkanResult<u32>> + Send>>;
fn submit(
    threaded: bool,
    queue: Arc<Queue>,
    command_buffers: &[Arc<dyn CommandBufferTrait>],
    waits: &[(PipelineStages, Arc<Semaphore>)],
    fence: Arc<Fence>,
) -> VulkanResult<TestFuture> {
    if threaded {
        Ok(Box::pin(ThreadedFenceWaiter::new_by_submit(
            ThreadPool::new(2)?,
            42,
            queue,
            command_buffers,
            waits,
            &[],
            fence,
        )?))
    } else {
        Ok(Box::pin(SpinlockFenceWaiter::new_by_submit(
            42,
            queue,
            command_buffers,
            waits,
            &[],
            fence,
        )?))
    }
}
fn setup() -> VulkanResult<(Arc<Device>, Arc<Queue>, Arc<CommandPool>)> {
    let (_, device) = super::common::setup_test_device()?;
    let family = QueueFamily::new(device.clone(), 0)?;
    let queue = Queue::new(family.clone(), None)?;
    let pool = CommandPool::new(family, None)?;
    Ok((device, queue, pool))
}
fn finish(future: &mut TestFuture) -> VulkanResult<u32> {
    let waker = Waker::from(Arc::new(WakeCounter::default()));
    let mut cx = Context::from_waker(&waker);
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        if let Poll::Ready(result) = future.as_mut().poll(&mut cx) {
            return result;
        }
        assert!(Instant::now() < deadline, "async fence timed out");
        std::thread::sleep(Duration::from_millis(1));
    }
}

#[test]
fn async_submit_preserves_rollback_fence_ownership_and_exactly_once_completion() -> VulkanResult<()>
{
    for threaded in [false, true] {
        let (device, queue, pool) = setup()?;
        let cb = CountedCommandBuffer::new(pool.clone())?;
        let unrecorded = PrimaryCommandBuffer::new(pool, None)?;
        let fence = Fence::new(device.clone(), false, None)?;
        let bad_batch: Vec<Arc<dyn CommandBufferTrait>> = vec![cb.clone(), unrecorded];
        assert!(submit(threaded, queue.clone(), &bad_batch, &[], fence.clone()).is_err());
        assert_eq!(cb.cancellations.load(Ordering::SeqCst), 1);
        assert_eq!(cb.completions.load(Ordering::SeqCst), 0);
        fence.reset()?;
        let good_batch: Vec<Arc<dyn CommandBufferTrait>> = vec![cb.clone()];
        let mut future = submit(threaded, queue.clone(), &good_batch, &[], fence.clone())?;
        assert!(fence.reset().is_err());
        assert!(queue.submit(&[], &[], &[], fence.clone()).is_err());
        assert_eq!(finish(&mut future)?, 42);
        assert_eq!(finish(&mut future)?, 42);
        drop(future);
        assert_eq!(cb.completions.load(Ordering::SeqCst), 1);
        assert!(cb.inner.mark_execution_begin().is_err());
        assert!(!fence.is_signaled()?);
        drop(queue.submit(&[], &[], &[], fence)?);
    }
    Ok(())
}

#[test]
fn async_pending_retains_resources_and_threaded_wait_uses_latest_waker() -> VulkanResult<()> {
    for threaded in [false, true] {
        let (device, queue, pool) = setup()?;
        let cb = CountedCommandBuffer::new(pool)?;
        let timeline = Semaphore::new_timeline(device.clone(), 0, None)?;
        let binary = Semaphore::new(device.clone(), None)?;
        let stages: PipelineStages = [PipelineStage::AllCommands].as_slice().into();
        // Signal the binary semaphore independently of the unresolved timeline wait.
        // A binary wait requires all dependencies of its signal to be submitted.
        drop(queue.submit(
            &[],
            &[],
            &[binary.clone()],
            Fence::new(device.clone(), false, None)?,
        )?);
        let blocker = queue.submit_mixed(
            &[],
            &[SemaphoreWaitOp::Timeline(stages, timeline.clone(), 1)],
            &[],
            Fence::new(device.clone(), false, None)?,
        )?;
        let fence = Fence::new(device.clone(), false, None)?;
        let weak_cb = Arc::downgrade(&cb);
        let weak_binary = Arc::downgrade(&binary);
        let mut future = submit(
            threaded,
            queue.clone(),
            &[cb.clone()],
            &[(stages, binary.clone())],
            fence.clone(),
        )?;
        drop(cb);
        drop(binary);
        let old_counter = Arc::new(WakeCounter::default());
        let old_waker = Waker::from(old_counter.clone());
        let latest_counter = Arc::new(WakeCounter::default());
        let latest_waker = Waker::from(latest_counter.clone());
        assert!(future
            .as_mut()
            .poll(&mut Context::from_waker(&old_waker))
            .is_pending());
        for _ in 0..16 {
            assert!(future
                .as_mut()
                .poll(&mut Context::from_waker(&latest_waker))
                .is_pending());
        }
        assert!(weak_cb.upgrade().is_some());
        assert!(weak_binary.upgrade().is_some());
        assert!(fence.reset().is_err());
        assert_eq!(
            weak_cb
                .upgrade()
                .unwrap()
                .completions
                .load(Ordering::SeqCst),
            0
        );
        unsafe {
            device.ash_handle().signal_semaphore(
                &ash::vk::SemaphoreSignalInfo::default()
                    .semaphore(timeline.ash_handle())
                    .value(1),
            )?;
        }
        drop(blocker);
        if threaded {
            let deadline = Instant::now() + Duration::from_secs(5);
            while latest_counter.0.load(Ordering::SeqCst) == 0 {
                assert!(Instant::now() < deadline, "threaded waiter lost its wakeup");
                std::thread::sleep(Duration::from_millis(1));
            }
            assert_eq!(old_counter.0.load(Ordering::SeqCst), 0);
            assert_eq!(
                latest_counter.0.load(Ordering::SeqCst),
                1,
                "polling must not schedule duplicate waits"
            );
        }
        assert_eq!(finish(&mut future)?, 42);
        assert!(weak_cb.upgrade().is_none());
        assert!(weak_binary.upgrade().is_none());
        drop(queue.submit(&[], &[], &[], fence)?);
    }
    Ok(())
}

#[test]
fn dropping_unpolled_or_pending_async_submit_completes_safely() -> VulkanResult<()> {
    for threaded in [false, true] {
        for poll_first in [false, true] {
            let (device, queue, pool) = setup()?;
            let cb = CountedCommandBuffer::new(pool)?;
            let timeline = Semaphore::new_timeline(device.clone(), 0, None)?;
            let stages: PipelineStages = [PipelineStage::AllCommands].as_slice().into();
            let blocker = queue.submit_mixed(
                &[],
                &[SemaphoreWaitOp::Timeline(stages, timeline.clone(), 1)],
                &[],
                Fence::new(device.clone(), false, None)?,
            )?;
            let fence = Fence::new(device.clone(), false, None)?;
            let mut future = submit(threaded, queue.clone(), &[cb.clone()], &[], fence.clone())?;
            if poll_first {
                let waker = Waker::from(Arc::new(WakeCounter::default()));
                assert!(future
                    .as_mut()
                    .poll(&mut Context::from_waker(&waker))
                    .is_pending());
            }
            let (tx, rx) = mpsc::channel();
            let dropping = std::thread::spawn(move || {
                drop(future);
                tx.send(()).unwrap();
            });
            let premature = rx.recv_timeout(Duration::from_millis(20)).is_ok();
            assert!(fence.reset().is_err());
            unsafe {
                device.ash_handle().signal_semaphore(
                    &ash::vk::SemaphoreSignalInfo::default()
                        .semaphore(timeline.ash_handle())
                        .value(1),
                )?;
            }
            rx.recv_timeout(Duration::from_secs(5)).unwrap();
            dropping.join().unwrap();
            drop(blocker);
            assert!(
                !premature,
                "drop must not release pending submission resources"
            );
            assert_eq!(cb.completions.load(Ordering::SeqCst), 1);
            drop(queue.submit(&[], &[], &[], fence)?);
        }
    }
    Ok(())
}

#[test]
fn external_async_waiters_reset_unowned_fences_and_finish_buffers_once() -> VulkanResult<()> {
    for threaded in [false, true] {
        for poll_first in [false, true] {
            let (device, _, pool) = setup()?;
            let cb = CountedCommandBuffer::new(pool)?;
            cb.mark_execution_begin()?;
            let fence = Fence::new(device, true, None)?;
            let mut future: TestFuture = if threaded {
                Box::pin(ThreadedFenceWaiter::new(
                    ThreadPool::new(1)?,
                    None,
                    &[cb.clone()],
                    &[],
                    fence.clone(),
                    42,
                ))
            } else {
                Box::pin(SpinlockFenceWaiter::new(
                    None,
                    &[cb.clone()],
                    &[],
                    fence.clone(),
                    42,
                ))
            };
            if poll_first {
                assert_eq!(finish(&mut future)?, 42);
            }
            drop(future);
            assert_eq!(cb.completions.load(Ordering::SeqCst), 1);
            assert!(!fence.is_signaled()?);
            fence.reset()?;
            Fence::wait_for_fences(&[], FenceWaitFor::All, Duration::ZERO)?;
        }
    }
    Ok(())
}

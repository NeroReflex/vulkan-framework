use crate::prelude::VulkanResult;
use std::sync::{mpsc::RecvTimeoutError, Arc, Mutex};
use std::time::Duration;

type JobOnce = Box<dyn FnOnce() + 'static + Send>;
type JobRetry = Box<dyn Fn() -> bool + 'static + Send>;

enum Message {
    NewJob(JobOnce),
    NewRetryingJob(JobRetry),
    Quit,
}

pub struct ThreadPool {
    workers: Vec<std::thread::JoinHandle<()>>,
    sender: std::sync::mpsc::Sender<Message>,
}

fn worker_loop(receiver: Arc<Mutex<std::sync::mpsc::Receiver<Message>>>) {
    let mut scheduled_retry_jobs: Vec<JobRetry> = Vec::new();
    loop {
        scheduled_retry_jobs.retain(|job| !job());

        // Never wait behind an idle worker holding the receiver lock: this worker
        // may have fence retries that must progress even with no new messages.
        let message = match receiver.try_lock() {
            Ok(rx) => rx.recv_timeout(Duration::from_millis(1)),
            Err(std::sync::TryLockError::WouldBlock) => {
                std::thread::sleep(Duration::from_micros(100));
                continue;
            }
            Err(std::sync::TryLockError::Poisoned(_)) => break,
        };
        match message {
            Ok(Message::NewJob(job)) => job(),
            Ok(Message::NewRetryingJob(job)) => {
                if !job() {
                    scheduled_retry_jobs.push(job);
                }
            }
            Ok(Message::Quit) | Err(RecvTimeoutError::Disconnected) => break,
            Err(RecvTimeoutError::Timeout) => {}
        }
    }
}

impl ThreadPool {
    pub fn new(max_workers: usize) -> VulkanResult<Arc<Self>> {
        if max_workers == 0 {
            return Err(ash::vk::Result::ERROR_INITIALIZATION_FAILED.into());
        }
        let (tx, rx) = std::sync::mpsc::channel::<Message>();
        let receiver = Arc::new(Mutex::new(rx));
        let mut workers = Vec::with_capacity(max_workers);

        for i in 0..max_workers {
            let recv = receiver.clone();
            let handle = std::thread::Builder::new()
                .name(format!("vulkan-pool-{}", i))
                .spawn(move || worker_loop(recv))
                .map_err(|_| ash::vk::Result::ERROR_INITIALIZATION_FAILED)?;
            workers.push(handle);
        }

        Ok(Arc::new(Self {
            workers,
            sender: tx,
        }))
    }

    pub fn execute_once<F>(&self, f: F)
    where
        F: FnOnce() + 'static + Send,
    {
        let _ = self.sender.send(Message::NewJob(Box::new(f)));
    }

    pub fn execute_retry<F>(&self, f: F)
    where
        F: Fn() -> bool + 'static + Send,
    {
        let _ = self.sender.send(Message::NewRetryingJob(Box::new(f)));
    }
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        for _ in &self.workers {
            let _ = self.sender.send(Message::Quit);
        }
        // Handles detach; workers exit on Quit or disconnection. Joining here can
        // deadlock if the final pool Arc is released inside one of its own jobs.
    }
}

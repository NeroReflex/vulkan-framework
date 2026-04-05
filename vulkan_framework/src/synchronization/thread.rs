use crate::prelude::VulkanResult;
use std::sync::{Arc, Mutex};
use std::time::Duration;

type JobOnce = Box<dyn FnOnce() + 'static + Send>;
type JobRetry = Box<dyn Fn() -> bool + 'static + Send>;

enum Message {
    NewJob(JobOnce),
    NewRetryingJob(JobRetry),
    Quit,
}

pub struct ThreadPool {
    _workers: Vec<std::thread::JoinHandle<()>>,
    sender: std::sync::mpsc::Sender<Message>,
}

fn worker_loop(receiver: Arc<Mutex<std::sync::mpsc::Receiver<Message>>>) {
    let mut scheduled_retry_jobs: Vec<Option<JobRetry>> = (0..8).map(|_| None).collect();
    loop {
        // Try one pending retry job before blocking for new work.
        let mut completed = None;
        for (i, maybe_job) in scheduled_retry_jobs.iter().enumerate() {
            if let Some(job) = maybe_job {
                if job() {
                    completed = Some(i);
                    break;
                }
            }
        }
        if let Some(i) = completed {
            scheduled_retry_jobs[i] = None;
        }

        let has_retry = scheduled_retry_jobs.iter().any(|j| j.is_some());
        let msg = {
            let rx = receiver.lock().unwrap();
            if has_retry {
                rx.recv_timeout(Duration::from_micros(100)).ok()
            } else {
                rx.recv().ok()
            }
        };

        match msg {
            Some(Message::NewJob(job)) => job(),
            Some(Message::NewRetryingJob(job)) => {
                if !job() {
                    for slot in &mut scheduled_retry_jobs {
                        if slot.is_none() {
                            *slot = Some(job);
                            break;
                        }
                    }
                }
            }
            Some(Message::Quit) | None => {
                if !has_retry {
                    break;
                }
            }
        }
    }
}

impl ThreadPool {
    pub fn new(max_workers: usize) -> VulkanResult<Arc<Self>> {
        let (tx, rx) = std::sync::mpsc::channel::<Message>();
        let receiver = Arc::new(Mutex::new(rx));
        let mut workers = Vec::with_capacity(max_workers);

        for i in 0..max_workers {
            let recv = receiver.clone();
            let handle = std::thread::Builder::new()
                .name(format!("vulkan-pool-{}", i))
                .spawn(move || worker_loop(recv))
                .expect("failed to spawn vulkan thread pool worker");
            workers.push(handle);
        }

        Ok(Arc::new(Self {
            _workers: workers,
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
        for _ in &self._workers {
            let _ = self.sender.send(Message::Quit);
        }
        // Workers are joined when the JoinHandles drop.
    }
}

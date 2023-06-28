use std::{
    collections::LinkedList,
    sync::{mpsc, Arc, Mutex},
    time::Duration,
};

use crate::prelude::VulkanResult;

type JobOnce = Box<dyn FnOnce() + 'static + Send>;
type JobRetry = Box<dyn Fn() -> bool + 'static + Send>;

enum Message {
    Close,
    NewJob(JobOnce),
    NewRetryingJob(JobRetry),
}

struct Worker {
    _id: usize,
    t: Option<std::thread::JoinHandle<()>>,
}

impl Worker {
    fn new(id: usize, receiver: Arc<Mutex<mpsc::Receiver<Message>>>) -> Worker {
        let t = std::thread::spawn(move || {
            let mut scheduled_retry_job: Vec<Option<JobRetry>> =
                vec![None, None, None, None, None, None, None, None];

            loop {
                match receiver
                    .lock()
                    .unwrap()
                    .recv_timeout(Duration::from_nanos(1))
                {
                    Ok(msg) => match msg {
                        Message::NewRetryingJob(job) => {
                            #[cfg(debug_assertions)]
                            println!("do job from worker[{}]", id);

                            if !job() {
                                #[cfg(debug_assertions)]
                                println!("job from worker[{}] asked to be executed again, scheduling execution", id);

                                let mut recycled = Option::<usize>::None;
                                for a in 0..scheduled_retry_job.len() {
                                    if scheduled_retry_job[a].is_none() {
                                        recycled = Some(a);
                                        break;
                                    }
                                }

                                match recycled {
                                    Some(idx) => scheduled_retry_job[idx] = Some(job),
                                    None => scheduled_retry_job.push(Some(job)),
                                }
                            }
                        }
                        Message::NewJob(job) => {
                            #[cfg(debug_assertions)]
                            println!("do job from worker[{}]", id);

                            job();
                        }
                        Message::Close => {
                            for job in scheduled_retry_job.iter_mut() {
                                match job {
                                    Some(job_fn) => while !job_fn() {},
                                    None => {}
                                }
                            }

                            #[cfg(debug_assertions)]
                            println!("Closing worker[{}]", id);

                            break;
                        }
                    },
                    Err(_timeout_err) => {
                        let mut completed = Option::<usize>::None;

                        'try_to_complete_one: for (job_idx, maybe_job) in
                            scheduled_retry_job.iter().enumerate()
                        {
                            match maybe_job {
                                Some(job) => {
                                    if job() {
                                        completed = Some(job_idx);
                                        // remove the current job and break the loop
                                        break 'try_to_complete_one;
                                    }
                                }
                                None => {}
                            }
                        }

                        if let Some(idx_to_remove) = completed {
                            scheduled_retry_job[idx_to_remove] = None;
                        }
                    }
                }
            }
        });

        Worker {
            _id: id,
            t: Some(t),
        }
    }
}

pub struct ThreadPool {
    workers: Vec<Worker>,
    max_workers: usize,
    sender: mpsc::Sender<Message>,
}

impl ThreadPool {
    pub fn new(max_workers: usize) -> VulkanResult<Arc<Self>> {
        if max_workers == 0 {
            panic!("max_workers must be greater than zero!")
        }

        let (tx, rx) = mpsc::channel();

        let mut workers = Vec::with_capacity(max_workers);
        let receiver = Arc::new(Mutex::new(rx));
        for i in 0..max_workers {
            workers.push(Worker::new(i, Arc::clone(&receiver)));
        }

        Ok(Arc::new(Self {
            workers: workers,
            max_workers: max_workers,
            sender: tx,
        }))
    }

    pub fn execute_once<F>(&self, f: F)
    where
        F: FnOnce() + 'static + Send,
    {
        let job = Message::NewJob(Box::new(f));
        self.sender.send(job).unwrap();
    }

    pub fn execute_retry<F>(&self, f: F)
    where
        F: Fn() -> bool + 'static + Send,
    {
        let job = Message::NewRetryingJob(Box::new(f));
        self.sender.send(job).unwrap();
    }
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        for _ in 0..self.max_workers {
            self.sender.send(Message::Close).unwrap();
        }
        for w in self.workers.iter_mut() {
            if let Some(t) = w.t.take() {
                t.join().unwrap();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn it_works() {
        let p = ThreadPool::new(4).unwrap();
        p.execute_once(|| println!("do new job1"));
        p.execute_once(|| println!("do new job2"));
        p.execute_once(|| println!("do new job3"));
        p.execute_once(|| println!("do new job4"));
    }
}

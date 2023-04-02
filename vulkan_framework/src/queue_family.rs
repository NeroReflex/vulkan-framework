#[derive(Copy, Clone)]
pub enum QueueFamilySupportedOperationType {
    Compute,
    Graphics,
    Transfer,
    Present,
}

#[derive(Clone)]
pub struct ConcreteQueueFamilyDescriptor {
    supported_operations: Vec<QueueFamilySupportedOperationType>,
    queue_priorities: Vec<f32>,
}

impl ConcreteQueueFamilyDescriptor {
    pub fn new(
        supported_operations: &[QueueFamilySupportedOperationType],
        queue_priorities: &[f32],
    ) -> Self {
        Self {
            supported_operations: supported_operations.iter().map(|el| el.clone()).collect(),
            queue_priorities: queue_priorities.iter().map(|a| a.clone()).collect(),
        }
    }

    pub fn max_queues(&self) -> u32 {
        self.queue_priorities.len() as u32
    }

    pub fn get_queue_priorities(&self) -> &[f32] {
        self.queue_priorities.as_slice()
    }
}

use crate::{
    queue_family::*,
    result::VkError,
};

pub struct Queue<'queue> {
    queue_family: &'queue QueueFamily<'queue>,

}

impl<'queue> QueueFamilyOwned<'queue> for Queue<'queue> {
    fn get_parent_queue_family(&self) -> &'queue QueueFamily<'queue> {
        self.queue_family
    }
}

impl<'queue> Drop for Queue<'queue> {
    fn drop(&mut self) {
        // Nothing to be done here
    }
}

impl<'queue> Queue<'queue> {
    pub fn get_priority() -> f32 {
        todo!()
    }
    
    pub fn new(device: &'queue QueueFamily<'queue>) -> Result<Self, VkError> {
        todo!()
    }


}
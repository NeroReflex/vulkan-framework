pub type VulkanResult<T> = Result<T, VulkanError>;

pub struct VulkanError {
    error_name: String,
}

impl VulkanError {
    pub(crate) fn new() -> VulkanError {
        todo!()
    }
}

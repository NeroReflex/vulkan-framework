pub type VulkanResult<T> = Result<T, VulkanError>;

pub struct FrameworkError {
    error_name: String,
}

pub enum VulkanError {
    Framework(FrameworkError),
    Vulkan,
    Unspecified,
}

impl VulkanError {
    pub(crate) fn new() -> VulkanError {
        todo!()
    }
}

pub type VulkanResult<T> = Result<T, VulkanError>;

#[derive(Debug)]
pub enum FrameworkError {
    MallocFail,
    IncompatibleMemoryHeapType,
    Unknown(Option<String>)
}

#[derive(Debug)]
pub enum VulkanError {
    Framework(FrameworkError),
    Vulkan(i32, Option<String>),
    MissingExtension(String),
    Unspecified,
}

impl VulkanError {
    pub fn is_timeout(&self) -> bool {
        if let Self::Vulkan(err_code, _) = self {
            return *err_code == 2;
        }

        false
    }
}

/*impl VulkanError {
    pub(crate) fn new() -> VulkanError {
        todo!()
    }
}*/

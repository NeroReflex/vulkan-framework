pub type VulkanResult<T> = Result<T, VulkanError>;

#[derive(Debug)]
pub enum FrameworkError {
    MallocFail,
    IncompatibleMemoryHeapType,
    UserInput(Option<String>),
    //QueueFamilyUnavailable,
    NoSuitableDeviceFound,
    NoSuitableMemoryHeapFound,
    ResourceFromIncompatibleDevice,
    CannotLoadVulkan,
    CannotCreateVulkanInstance,
    MapMemoryError(i32),
    Unknown(Option<String>),
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

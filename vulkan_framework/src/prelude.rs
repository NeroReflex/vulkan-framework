use std::fmt::Display;

pub type VulkanResult<T> = Result<T, VulkanError>;

#[derive(Debug, Clone)]
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
    MapMemoryError,
    DescriptorSetBindingOutOfRange,
    DescriptorSetBindingDuplicated,
    MalformedRenderpassDefinition,
    MemoryHeapAndResourceNotFromTheSameDevice,
    Unknown(Option<String>),
}

impl Display for FrameworkError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FrameworkError::MallocFail => write!(f, "MallocFail"),
            FrameworkError::IncompatibleMemoryHeapType => {
                write!(f, "IncompatibleMemoryHeapType")
            }
            FrameworkError::UserInput(_maybe_details) => {
                write!(f, "UserInput")
            }
            FrameworkError::NoSuitableDeviceFound => {
                write!(f, "NoSuitableDeviceFound")
            }
            FrameworkError::NoSuitableMemoryHeapFound => {
                write!(f, "NoSuitableMemoryHeapFound")
            }
            FrameworkError::ResourceFromIncompatibleDevice => {
                write!(f, "ResourceFromIncompatibleDevice")
            }
            FrameworkError::CannotLoadVulkan => write!(f, "CannotLoadVulkan"),
            FrameworkError::CannotCreateVulkanInstance => {
                write!(f, "CannotCreateVulkanInstance")
            }
            FrameworkError::MapMemoryError => write!(f, "MapMemoryError"),
            FrameworkError::Unknown(_details) => write!(f, "Unknown"),
            FrameworkError::MalformedRenderpassDefinition => {
                write!(f, "Malformed renderpass definition")
            }
            FrameworkError::DescriptorSetBindingOutOfRange => {
                write!(f, "descriptor set binding is out of range")
            }
            FrameworkError::DescriptorSetBindingDuplicated => write!(
                f,
                "descriptor set binding is being used twice in the same write"
            ),
            FrameworkError::MemoryHeapAndResourceNotFromTheSameDevice => write!(
                f,
                "memory heap and resource to be allocated are not from the same device"
            ),
        }
    }
}

#[derive(Debug, Clone)]
pub enum VulkanError {
    Framework(FrameworkError),
    Vulkan(i32, Option<String>),
    MissingExtension(String),
}

impl Display for VulkanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VulkanError::Framework(error) => write!(f, "Framework error: {error}"),
            VulkanError::Vulkan(code, maybe_str) => match maybe_str {
                Some(str) => write!(f, "Vulkan error ({}): {}", code, str),
                None => write!(f, "Vulkan error ({})", code),
            },
            VulkanError::MissingExtension(name) => write!(f, "Missing vulkan extension: {}", name),
        }
    }
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

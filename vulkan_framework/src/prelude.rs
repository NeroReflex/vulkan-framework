use std::fmt::Display;

use crate::instance::InstanceAPIVersion;

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
    MapMemoryError,
    IncompatibleInstanceVersion(InstanceAPIVersion, InstanceAPIVersion),
    Unknown(Option<String>),
}

#[derive(Debug)]
pub enum VulkanError {
    Framework(FrameworkError),
    Vulkan(i32, Option<String>),
    MissingExtension(String),
}

impl Display for VulkanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VulkanError::Framework(error) => {
                write!(f, "Framework error");
                match error {
                    FrameworkError::MallocFail => write!(f, " MallocFail"),
                    FrameworkError::IncompatibleMemoryHeapType => {
                        write!(f, " IncompatibleMemoryHeapType")
                    }
                    FrameworkError::UserInput(maybe_details) => write!(f, " UserInput"),
                    FrameworkError::NoSuitableDeviceFound => write!(f, " NoSuitableDeviceFound"),
                    FrameworkError::NoSuitableMemoryHeapFound => {
                        write!(f, " NoSuitableMemoryHeapFound")
                    }
                    FrameworkError::ResourceFromIncompatibleDevice => {
                        write!(f, " ResourceFromIncompatibleDevice")
                    }
                    FrameworkError::CannotLoadVulkan => write!(f, " CannotLoadVulkan"),
                    FrameworkError::CannotCreateVulkanInstance => {
                        write!(f, " CannotCreateVulkanInstance")
                    }
                    FrameworkError::MapMemoryError => write!(f, " MapMemoryError"),
                    FrameworkError::IncompatibleInstanceVersion(
                        current_version,
                        wanted_version,
                    ) => write!(f, " IncompatibleInstanceVersion"),
                    FrameworkError::Unknown(details) => write!(f, " Unknown"),
                }
            }
            VulkanError::Vulkan(code, maybe_str) => {
                write!(f, "Vulkan error ({})", code);
                match maybe_str {
                    Some(str) => write!(f, ": {}", str),
                    None => write!(f, ""),
                }
            }
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

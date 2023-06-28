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
            VulkanError::Framework(error) => match error {
                FrameworkError::MallocFail => write!(f, "Framework error: MallocFail"),
                FrameworkError::IncompatibleMemoryHeapType => {
                    write!(f, "Framework error: IncompatibleMemoryHeapType")
                }
                FrameworkError::UserInput(_maybe_details) => {
                    write!(f, "Framework error: UserInput")
                }
                FrameworkError::NoSuitableDeviceFound => {
                    write!(f, "Framework error: NoSuitableDeviceFound")
                }
                FrameworkError::NoSuitableMemoryHeapFound => {
                    write!(f, "Framework error: NoSuitableMemoryHeapFound")
                }
                FrameworkError::ResourceFromIncompatibleDevice => {
                    write!(f, "Framework error: ResourceFromIncompatibleDevice")
                }
                FrameworkError::CannotLoadVulkan => write!(f, "Framework error: CannotLoadVulkan"),
                FrameworkError::CannotCreateVulkanInstance => {
                    write!(f, "Framework error: CannotCreateVulkanInstance")
                }
                FrameworkError::MapMemoryError => write!(f, "Framework error: MapMemoryError"),
                FrameworkError::IncompatibleInstanceVersion(_current_version, _wanted_version) => {
                    write!(f, "Framework error: IncompatibleInstanceVersion")
                }
                FrameworkError::Unknown(_details) => write!(f, "Framework error: Unknown"),
            },
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

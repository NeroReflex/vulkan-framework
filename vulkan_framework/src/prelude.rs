use thiserror::Error;

use crate::memory_management::UnallocatedResource;

#[derive(Debug, Error)]
pub enum FrameworkError {
    #[error("Memory allocation failed")]
    MallocFail(UnallocatedResource),
    #[error("Incompatible memory heap type")]
    IncompatibleMemoryHeapType,
    #[error("Error creating the descriptor set: bindings are not starting from zero")]
    InvalidBindingsNotStartingFromZero,
    #[error("In a subpass one input attachment is specified to be {0}, but only {1} attachments have beed defined")]
    InvalidInputAttachment(usize, usize),
    #[error("In a subpass one color attachment is specified to be {0}, but only {1} attachments have beed defined")]
    InvalidColorAttachment(usize, usize),
    #[error("Error in queue search: no queue descriptor(s) have been specified")]
    MissingQueueDescriptor,
    #[error("From this QueueFamily the number of created Queue(s) is {0} out of a maximum supported number of {1} has already been created")]
    TooManyQueues(usize, usize),
    #[error("The queue family with index {0} has already been created once and there can only be one QueueFamily for requested queue capabilies")]
    QueueFamilyAlreadyCreated(usize),
    #[error("A queue family with index {0} does not exists, at device creation time only {1} queue families were requested")]
    TooManyQueueFamilies(usize, usize),
    #[error("Error creating image: number of layers must be at least 1")]
    NoImageLayersSpecified,
    #[error("Error creating image: number of mipmap levels must be at least 1 (the base one)")]
    NoMipMapLevelsSpecified,
    #[error("Unsuitable image dimensions for the resource")]
    UnsuitableImageDimensions,
    #[error("Unsuitable memory heap: the given allocator will manage {0} bytes, but the selected memory heap only has {1} bytes available")]
    UnsuitableMemoryHeapForAllocator(u64, u64),
    #[error("No suitable devices found")]
    NoSuitableDeviceFound,
    #[error("No suitable memory heap has been found")]
    NoSuitableMemoryHeapFound,
    #[error("Resources are from an incompatible device")]
    ResourceFromIncompatibleDevice,
    #[error("Error loading vulkan")]
    CannotLoadVulkan,
    #[error("Error creating the vulkan instance")]
    CannotCreateVulkanInstance,
    #[error("Error mapping memory")]
    MapMemoryError,
    #[error("Descriptor set binding is out of range")]
    DescriptorSetBindingOutOfRange,
    #[error("Descriptor set binding is being used twice in the same write")]
    DescriptorSetBindingDuplicated,
    #[error("Malformed renderpass definition")]
    MalformedRenderpassDefinition,
    #[error("Memory heap and resource to be allocated are not from the same device")]
    MemoryHeapAndResourceNotFromTheSameDevice,
    #[error("Resources have no memory type requirements in common and are therefore incompatible")]
    IncompatibleResources,
    #[error("Missing feature on MemoryPool: device_addressable need to be set")]
    MemoryPoolNotAddressable,
    #[error("The memory pool is already mapped in host memory")]
    MemoryPoolAlreadyMapped,
    #[error("Error creating the descriptor set layout: no binding descriptors specified")]
    NoBindingDescriptorsSpecified,
    #[error("Mutex error: {0}")]
    MutexError(String),
    #[error("Invalid descriptor set usage")]
    InvalidDescriptorSetUsage,
    #[error("Swapchain already exists")]
    SwapchainAlreadyExists,
    #[error("Invalid swapchain image index {0}, the swapchain currently holds {1} images")]
    InvalidSwapchainImageIndex(usize, usize),
    #[error("Command buffer was submitted while no commands were registered")]
    CommandBufferSubmitNoCommands,
    #[error("Command buffer was submitted while commands were still recording")]
    CommandBufferSubmitRecording,
    #[error("Command buffer was submitted while already running")]
    CommandBufferSubmitAlreadyRunning,
    #[error("Command buffer record attempt while commands were still recording")]
    CommandBufferRecordRecording,
    #[error("Command buffer record attempt while commands were still running")]
    CommandBufferRecordRunning,
    #[error("Command buffer internal error: {0}")]
    CommandBufferInternalError(u32),
    #[error("Command buffer is in an invalid state")]
    CommandBufferInvalidState,
}

#[derive(Debug, Error)]
pub enum VulkanError {
    #[error("Framework error: {0}")]
    Framework(#[from] FrameworkError),

    #[error("Framework error: {0}")]
    Vulkan(#[from] crate::ash::vk::Result),

    #[error("Missing extension: {0}")]
    MissingExtension(String),
}

pub type VulkanResult<T> = Result<T, VulkanError>;

impl VulkanError {
    pub fn is_timeout(&self) -> bool {
        if let Self::Vulkan(result) = self {
            return result.as_raw() == ash::vk::Result::TIMEOUT.as_raw();
        }

        false
    }
}

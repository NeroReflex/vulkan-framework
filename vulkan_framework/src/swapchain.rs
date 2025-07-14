use std::{sync::Arc, time::Duration};

use ash::vk::Handle;

#[cfg(feature = "async")]
use crate::synchronization::{
    fence::{SpinlockFenceWaiter, ThreadedFenceWaiter},
    thread::ThreadPool,
};

use crate::{
    device::{Device, DeviceOwned},
    fence::Fence,
    image::{Image1DTrait, Image2DDimensions, Image2DTrait, ImageFlags, ImageFormat, ImageUsage},
    instance::InstanceOwned,
    prelude::{VulkanError, VulkanResult},
    queue::Queue,
    queue_family::{QueueFamily, QueueFamilyOwned},
    semaphore::Semaphore,
    surface::Surface,
};

/**
 * Swapchain present modes as defined in vulkan.
 *
 * Immediate = VK_PRESENT_MODE_IMMEDIATE_KHR. This one is the one that results in visible tearing
 * Mailbox = VK_PRESENT_MODE_MAILBOX_KHR
 * FIFO = VK_PRESENT_MODE_FIFO_KHR this cannot generate tearing and is always supported
 * FIFORelaxed = VK_PRESENT_MODE_FIFO_RELAXED_KHR
 */
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum PresentModeSwapchainKHR {
    Immediate,
    Mailbox,
    FIFO,
    FIFORelaxed,
}

impl PresentModeSwapchainKHR {
    pub(crate) fn ash_value(&self) -> ash::vk::PresentModeKHR {
        match self {
            PresentModeSwapchainKHR::Immediate => ash::vk::PresentModeKHR::IMMEDIATE,
            PresentModeSwapchainKHR::Mailbox => ash::vk::PresentModeKHR::MAILBOX,
            PresentModeSwapchainKHR::FIFO => ash::vk::PresentModeKHR::FIFO,
            PresentModeSwapchainKHR::FIFORelaxed => ash::vk::PresentModeKHR::FIFO_RELAXED,
        }
    }
}

pub enum SurfaceColorspaceSwapchainKHR {
    SRGBNonlinear,
    // TODO: VK_AMD_display_native_hdr for freesync
}

impl SurfaceColorspaceSwapchainKHR {
    pub(crate) fn ash_colorspace(&self) -> ash::vk::ColorSpaceKHR {
        match self {
            SurfaceColorspaceSwapchainKHR::SRGBNonlinear => ash::vk::ColorSpaceKHR::SRGB_NONLINEAR,
        }
    }
}

#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum CompositeAlphaSwapchainKHR {
    Opaque = 0x00000001u32,
    PreMultiplied = 0x00000002u32,
    PostMultiplied = 0x00000004u32,
    Inherit = 0x00000008u32,
}

impl CompositeAlphaSwapchainKHR {
    pub(crate) fn ash_alpha(&self) -> ash::vk::CompositeAlphaFlagsKHR {
        match self {
            Self::Opaque => ash::vk::CompositeAlphaFlagsKHR::OPAQUE,
            Self::PreMultiplied => ash::vk::CompositeAlphaFlagsKHR::PRE_MULTIPLIED,
            Self::PostMultiplied => ash::vk::CompositeAlphaFlagsKHR::POST_MULTIPLIED,
            Self::Inherit => ash::vk::CompositeAlphaFlagsKHR::INHERIT,
        }
    }
}

#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum SurfaceTransformSwapchainKHR {
    Identity = 0x00000001u32,
    Rotate90 = 0x00000002u32,
    Rotate180 = 0x00000004u32,
    Rotate270 = 0x00000008u32,
    HorizontalMirror = 0x00000010u32,
    HorizontalMirrorRotate90 = 0x00000020,
    HorizontalMirrorRotate180 = 0x00000040u32,
    HorizontalMirrorRotate270 = 0x00000080u32,
    Inherit = 0x00000100u32,
}

impl SurfaceTransformSwapchainKHR {
    pub(crate) fn ash_transform(&self) -> ash::vk::SurfaceTransformFlagsKHR {
        match self {
            Self::Identity => ash::vk::SurfaceTransformFlagsKHR::IDENTITY,
            Self::Rotate90 => ash::vk::SurfaceTransformFlagsKHR::ROTATE_90,
            Self::Rotate180 => ash::vk::SurfaceTransformFlagsKHR::ROTATE_180,
            Self::Rotate270 => ash::vk::SurfaceTransformFlagsKHR::ROTATE_270,
            Self::HorizontalMirror => ash::vk::SurfaceTransformFlagsKHR::HORIZONTAL_MIRROR,
            Self::HorizontalMirrorRotate90 => {
                ash::vk::SurfaceTransformFlagsKHR::HORIZONTAL_MIRROR_ROTATE_90
            }
            Self::HorizontalMirrorRotate180 => {
                ash::vk::SurfaceTransformFlagsKHR::HORIZONTAL_MIRROR_ROTATE_180
            }
            Self::HorizontalMirrorRotate270 => {
                ash::vk::SurfaceTransformFlagsKHR::HORIZONTAL_MIRROR_ROTATE_270
            }
            Self::Inherit => ash::vk::SurfaceTransformFlagsKHR::INHERIT,
        }
    }
}

pub struct DeviceSurfaceInfo {
    device: Arc<Device>,
    surface: Arc<Surface>,
    surface_capabilities: ash::vk::SurfaceCapabilitiesKHR,
    surface_present_modes: smallvec::SmallVec<[ash::vk::PresentModeKHR; 4]>,
    surface_formats: smallvec::SmallVec<[ash::vk::SurfaceFormatKHR; 8]>,
}

impl DeviceOwned for DeviceSurfaceInfo {
    fn get_parent_device(&self) -> Arc<Device> {
        self.device.clone()
    }
}

impl DeviceSurfaceInfo {
    pub fn image_count_supported(&self, count: u32) -> bool {
        (count >= self.surface_capabilities.min_image_count)
            && ((count < self.surface_capabilities.max_image_count)
                || (self.surface_capabilities.max_image_count == 0))
    }

    /**
     * Result is either 0 (no limits) or a number >= min_image_count()
     *
     * Make sure to use the function image_count_supported to check if the desired number is supported!
     */
    pub fn max_image_count(&self) -> u32 {
        self.surface_capabilities.max_image_count
    }

    pub fn min_image_count(&self) -> u32 {
        self.surface_capabilities.min_image_count
    }

    pub fn present_mode_supported(&self, mode: &PresentModeSwapchainKHR) -> bool {
        self.surface_present_modes.contains(&mode.ash_value())
    }

    pub fn format_supported(
        &self,
        color_space: &SurfaceColorspaceSwapchainKHR,
        format: &ImageFormat,
    ) -> bool {
        let fmt = ash::vk::SurfaceFormatKHR::default()
            .format(format.ash_format())
            .color_space(color_space.ash_colorspace());

        self.surface_formats.contains(&fmt)
    }

    pub fn new(device: Arc<Device>, surface: Arc<Surface>) -> VulkanResult<Self> {
        match device.get_parent_instance().get_surface_khr_extension() {
            Some(sfc_ext) => {
                let surface_capabilities_result = unsafe {
                    sfc_ext.get_physical_device_surface_capabilities(
                        device.ash_physical_device_handle().to_owned(),
                        surface.ash_handle().to_owned(),
                    )
                };

                if let Err(err) = surface_capabilities_result {
                    return Err(VulkanError::Vulkan(
                        err.as_raw(),
                        Some(format!("Error in fetching surface capabilities: {}", err)),
                    ));
                }

                let surface_present_modes_result = unsafe {
                    sfc_ext.get_physical_device_surface_present_modes(
                        device.ash_physical_device_handle().to_owned(),
                        surface.ash_handle().to_owned(),
                    )
                };

                if let Err(err) = surface_present_modes_result {
                    return Err(VulkanError::Vulkan(
                        err.as_raw(),
                        Some(format!("Error in fetching surface present modes: {}", err)),
                    ));
                }

                let surface_formats_result = unsafe {
                    sfc_ext.get_physical_device_surface_formats(
                        device.ash_physical_device_handle().to_owned(),
                        surface.ash_handle().to_owned(),
                    )
                };

                if let Err(err) = surface_present_modes_result {
                    return Err(VulkanError::Vulkan(
                        err.as_raw(),
                        Some(format!(
                            "Error in fetching surface supported formats: {}",
                            err
                        )),
                    ));
                }

                let surface_capabilities = surface_capabilities_result.unwrap();
                let surface_present_modes = surface_present_modes_result.unwrap();
                let surface_formats = surface_formats_result.unwrap();

                Ok(Self {
                    device,
                    surface,
                    surface_capabilities,
                    surface_present_modes: surface_present_modes
                        .into_iter()
                        .collect::<smallvec::SmallVec<[ash::vk::PresentModeKHR; 4]>>(),
                    surface_formats: surface_formats
                        .into_iter()
                        .collect::<smallvec::SmallVec<[ash::vk::SurfaceFormatKHR; 8]>>(),
                })
            }
            None => Err(VulkanError::MissingExtension(String::from(
                "VK_KHR_swapchain",
            ))),
        }
    }
}

pub struct SwapchainKHR {
    device: Arc<Device>,
    surface: Arc<Surface>,
    swapchain: ash::vk::SwapchainKHR,
    image_format: ImageFormat,
    image_usage: ImageUsage,
    extent: Image2DDimensions,
    transform: SurfaceTransformSwapchainKHR,
    composite_alpha: CompositeAlphaSwapchainKHR,
    min_image_count: u32,
    image_layers: u32,
}

pub trait SwapchainKHROwned {
    fn get_parent_swapchain(&self) -> Arc<SwapchainKHR>;
}

impl DeviceOwned for SwapchainKHR {
    #[inline]
    fn get_parent_device(&self) -> Arc<Device> {
        self.device.clone()
    }
}

impl Drop for SwapchainKHR {
    #[inline]
    fn drop(&mut self) {
        match self.device.ash_ext_swapchain_khr() {
            Some(ext) => unsafe {
                ext.destroy_swapchain(
                    self.swapchain,
                    self.device.get_parent_instance().get_alloc_callbacks(),
                )
            },
            None => {
                panic!("Swapchain extension is not available anymore. This should not happend. If you read this main developer of this crate made something bad.");
            }
        }
    }
}

impl SwapchainKHR {
    #[inline]
    pub(crate) fn ash_handle(&self) -> ash::vk::SwapchainKHR {
        self.swapchain
    }

    pub fn surface(&self) -> Arc<Surface> {
        self.surface.clone()
    }

    pub fn transform(&self) -> SurfaceTransformSwapchainKHR {
        self.transform
    }

    pub fn composite_alpha(&self) -> CompositeAlphaSwapchainKHR {
        self.composite_alpha
    }

    pub fn min_image_count(&self) -> u32 {
        self.min_image_count
    }

    pub fn images_flags(&self) -> ImageFlags {
        ImageFlags::Unmanaged(0)
    }

    pub fn images_usage(&self) -> ImageUsage {
        self.image_usage
    }

    #[inline]
    pub fn images_format(&self) -> crate::image::ImageFormat {
        self.image_format
    }

    #[inline]
    pub fn images_extent(&self) -> crate::image::Image2DDimensions {
        self.extent
    }

    #[inline]
    pub fn images_layers_count(&self) -> u32 {
        self.image_layers
    }

    pub fn queue_present(
        &self,
        queue: Arc<Queue>,
        index: u32,
        semaphores: &[Arc<Semaphore>],
    ) -> VulkanResult<()> {
        if self.get_parent_device() != queue.get_parent_queue_family().get_parent_device() {
            return Err(VulkanError::Framework(
                crate::prelude::FrameworkError::ResourceFromIncompatibleDevice,
            ));
        }

        let native_semaphores = semaphores
            .iter()
            .map(|sem| {
                // TODO: check self.device == sem.device

                sem.ash_handle()
            })
            .collect::<smallvec::SmallVec<[ash::vk::Semaphore; 8]>>();

        let swapchains = [self.ash_handle()];
        let indexes = [index];
        let present_info = ash::vk::PresentInfoKHR::default()
            .swapchains(&swapchains)
            .image_indices(&indexes)
            .wait_semaphores(native_semaphores.as_slice());

        match self.get_parent_device().ash_ext_swapchain_khr() {
            Option::Some(ext) => {
                match unsafe { ext.queue_present(queue.ash_handle(), &present_info) } {
                    Ok(_suboptimal) => Ok(()),
                    Err(err) => Err(VulkanError::Vulkan(err.as_raw(), None)),
                }
            }
            Option::None => Err(VulkanError::MissingExtension(String::from(
                "VK_KHR_swapchain",
            ))),
        }
    }

    #[cfg(feature = "async")]
    pub fn async_threaded_acquire_next_image_index(
        &self,
        pool: Arc<ThreadPool>,
        timeout: Duration,
        maybe_semaphore: Option<Arc<Semaphore>>,
        fence: Arc<Fence>,
    ) -> VulkanResult<ThreadedFenceWaiter<(u32, bool)>> {
        match self.get_parent_device().ash_ext_swapchain_khr() {
            Option::Some(ext) => {
                match unsafe {
                    let nanos = timeout.as_nanos();
                    ext.acquire_next_image(
                        self.swapchain,
                        if nanos > (u64::MAX as u128) {
                            u64::MAX
                        } else {
                            nanos as u64
                        },
                        match &maybe_semaphore {
                            Option::Some(semaphore) => semaphore.ash_handle(),
                            Option::None => ash::vk::Semaphore::null(),
                        },
                        fence.ash_handle(),
                    )
                } {
                    Ok(result) => match &maybe_semaphore {
                        Some(sem) => Ok(ThreadedFenceWaiter::new(
                            pool,
                            None,
                            &[],
                            &[sem.clone()],
                            fence,
                            result,
                        )),
                        None => Ok(ThreadedFenceWaiter::new(
                            pool,
                            None,
                            &[],
                            &[],
                            fence,
                            result,
                        )),
                    },
                    Err(err) => Err(VulkanError::Vulkan(
                        err.as_raw(),
                        Some(format!("Error creating the swapchain: {}", err)),
                    )),
                }
            }
            Option::None => Err(VulkanError::MissingExtension(String::from(
                "VK_KHR_swapchain",
            ))),
        }
    }

    #[cfg(feature = "async")]
    pub fn async_spinlock_acquire_next_image_index(
        &self,
        timeout: Duration,
        maybe_semaphore: Option<Arc<Semaphore>>,
        fence: Arc<Fence>,
    ) -> VulkanResult<SpinlockFenceWaiter<(u32, bool)>> {
        match self.get_parent_device().ash_ext_swapchain_khr() {
            Option::Some(ext) => {
                match unsafe {
                    let nanos = timeout.as_nanos();
                    ext.acquire_next_image(
                        self.swapchain,
                        if nanos > (u64::MAX as u128) {
                            u64::MAX
                        } else {
                            nanos as u64
                        },
                        match maybe_semaphore {
                            Option::Some(semaphore) => semaphore.ash_handle(),
                            Option::None => ash::vk::Semaphore::null(),
                        },
                        fence.ash_handle(),
                    )
                } {
                    Ok(result) => Ok(SpinlockFenceWaiter::new(None, &[], &[], fence, result)),
                    Err(err) => Err(VulkanError::Vulkan(
                        err.as_raw(),
                        Some(format!("Error creating the swapchain: {}", err)),
                    )),
                }
            }
            Option::None => Err(VulkanError::MissingExtension(String::from(
                "VK_KHR_swapchain",
            ))),
        }
    }

    /// Retrieve the index of the next available presentable image.
    ///
    /// @param timeout how long the function waits if no image is available
    /// @param maybe_semaphore the semaphore to signal
    /// @param maybe_fence the fence to signal
    /// @returns a tuple of the index and a boolean true or a boolean false if the image is suboptimal
    ///
    pub fn acquire_next_image_index(
        &self,
        timeout: Duration,
        maybe_semaphore: Option<Arc<Semaphore>>,
        maybe_fence: Option<Arc<Fence>>,
    ) -> VulkanResult<(u32, bool)> {
        match self.get_parent_device().ash_ext_swapchain_khr() {
            Option::Some(ext) => {
                match unsafe {
                    let nanos = timeout.as_nanos();
                    ext.acquire_next_image(
                        self.swapchain,
                        if nanos > (u64::MAX as u128) {
                            u64::MAX
                        } else {
                            nanos as u64
                        },
                        match maybe_semaphore {
                            Option::Some(semaphore) => semaphore.ash_handle(),
                            Option::None => ash::vk::Semaphore::null(),
                        },
                        match maybe_fence {
                            Option::Some(fence) => fence.ash_handle(),
                            Option::None => ash::vk::Fence::null(),
                        },
                    )
                } {
                    Ok(result) => Ok(result),
                    Err(err) => Err(VulkanError::Vulkan(
                        err.as_raw(),
                        Some(format!("Error creating the swapchain: {}", err)),
                    )),
                }
            }
            Option::None => Err(VulkanError::MissingExtension(String::from(
                "VK_KHR_swapchain",
            ))),
        }
    }

    pub fn new(
        device_info: &DeviceSurfaceInfo,
        queue_families: &[Arc<QueueFamily>],
        old_swapchain: Option<Arc<Self>>,
        present_mode: PresentModeSwapchainKHR,
        color_space: SurfaceColorspaceSwapchainKHR,
        composite_alpha: CompositeAlphaSwapchainKHR,
        transform: SurfaceTransformSwapchainKHR,
        clipped: bool,
        image_format: ImageFormat,
        image_usage: ImageUsage,
        extent: Image2DDimensions,
        min_image_count: u32,
        image_layers: u32,
    ) -> VulkanResult<Arc<Self>> {
        let queue_family_indexes: Vec<u32> = queue_families
            .iter()
            .map(|family| family.get_family_index())
            .collect();

        let device = device_info.device.clone();
        let surface = device_info.surface.clone();

        match device.ash_ext_swapchain_khr() {
            Some(ext) => {
                assert!(
                    device_info.format_supported(&color_space, &image_format)
                        && device_info.present_mode_supported(&present_mode)
                );

                let create_info = ash::vk::SwapchainCreateInfoKHR::default()
                    .old_swapchain(match &old_swapchain {
                        Some(old) => old.swapchain,
                        None => ash::vk::SwapchainKHR::from_raw(0),
                    })
                    .image_extent(
                        ash::vk::Extent2D::default()
                            .height(extent.height())
                            .width(extent.width()),
                    )
                    .surface(*surface.ash_handle())
                    .queue_family_indices(queue_family_indexes.as_slice())
                    .image_sharing_mode(match queue_family_indexes.len() <= 1 {
                        true => ash::vk::SharingMode::EXCLUSIVE,
                        false => ash::vk::SharingMode::CONCURRENT,
                    })
                    .image_usage(image_usage.ash_usage())
                    .image_array_layers(image_layers)
                    .image_format(image_format.ash_format())
                    .image_color_space(color_space.ash_colorspace())
                    .present_mode(present_mode.ash_value())
                    .clipped(clipped)
                    .pre_transform(transform.ash_transform())
                    .composite_alpha(composite_alpha.ash_alpha())
                    .min_image_count(min_image_count);

                //let surface_capabilities =

                match unsafe {
                    ext.create_swapchain(
                        &create_info,
                        device.get_parent_instance().get_alloc_callbacks(),
                    )
                } {
                    Ok(swapchain) => {
                        // the old swapchain (or at least my handle of it) must be removed AFTER the creation of the new one, not BEFORE!
                        std::mem::drop(old_swapchain);

                        Ok(Arc::new(Self {
                            device,
                            surface,
                            swapchain,
                            min_image_count,
                            transform,
                            composite_alpha,
                            image_format,
                            image_usage,
                            extent,
                            image_layers,
                        }))
                    }
                    Err(err) => Err(VulkanError::Vulkan(
                        err.as_raw(),
                        Some(format!("Error creating the swapchain: {}", err)),
                    )),
                }
            }
            None => Err(VulkanError::MissingExtension(String::from(
                "VK_KHR_swapchain",
            ))),
        }
    }
}

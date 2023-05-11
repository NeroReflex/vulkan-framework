use std::sync::Arc;

use ash::vk::Handle;

use crate::{device::{Device, DeviceOwned}, prelude::{VulkanResult, VulkanError}, instance::InstanceOwned, image::{Image2DDimensions, Image1DTrait, Image2DTrait, ImageUsageSpecifier, ImageUsage, ImageFormat}, queue_family::QueueFamily, surface::Surface};

pub struct SwapchainKHR {
    device: Arc<Device>,
    surface: Arc<Surface>,
    swapchain: ash::vk::SwapchainKHR,
    image_count: u32,
}

/**
 * Swapchain present modes as defined in vulkan.
 * 
 * Immediate = VK_PRESENT_MODE_IMMEDIATE_KHR. This one is the one that results in visible tearing
 * Mailbox = VK_PRESENT_MODE_MAILBOX_KHR
 * FIFO = VK_PRESENT_MODE_FIFO_KHR this cannot generate tearing and is always supported
 * FIFORelaxed = VK_PRESENT_MODE_FIFO_RELAXED_KHR
 */
#[derive(Copy, Clone, PartialEq, Eq)]
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
            SurfaceColorspaceSwapchainKHR::SRGBNonlinear => ash::vk::ColorSpaceKHR::SRGB_NONLINEAR
        }
    }
}

pub struct DeviceSurfaceInfo {
    device: Arc<Device>,
    surface: Arc<Surface>,
    surface_capabilities: ash::vk::SurfaceCapabilitiesKHR,
    surface_present_modes: Vec<ash::vk::PresentModeKHR>,
    surface_formats: Vec<ash::vk::SurfaceFormatKHR>,
}

impl DeviceOwned for DeviceSurfaceInfo {
    fn get_parent_device(&self) -> Arc<Device> {
        self.device.clone()
    }
}

impl DeviceSurfaceInfo {
    pub fn present_mode_supported(&self, mode: &PresentModeSwapchainKHR) -> bool {
        self.surface_present_modes.contains(&mode.ash_value())
    }

    pub fn format_supported(&self, color_space: &SurfaceColorspaceSwapchainKHR, format: &ImageFormat) -> bool {
        let fmt = ash::vk::SurfaceFormatKHR::builder()
            .format(format.ash_format())
            .color_space(color_space.ash_colorspace())
            .build();

        self.surface_formats.contains(&fmt)
    }

    pub fn new(
        device: Arc<Device>,
        surface: Arc<Surface>,
    ) -> VulkanResult<Self> {
        match device.get_parent_instance().get_surface_khr_extension() {
            Some(sfc_ext) => {
                let surface_capabilities_result = unsafe { sfc_ext.get_physical_device_surface_capabilities(device.ash_physical_device_handle().to_owned(), surface.ash_handle().to_owned()) };
                let surface_present_modes_result = unsafe { sfc_ext.get_physical_device_surface_present_modes(device.ash_physical_device_handle().to_owned(), surface.ash_handle().to_owned()) };
                let surface_formats_result = unsafe { sfc_ext.get_physical_device_surface_formats(device.ash_physical_device_handle().to_owned(), surface.ash_handle().to_owned()) };

                match (surface_capabilities_result, surface_present_modes_result, surface_formats_result) {
                    (Ok(surface_capabilities), Ok(surface_present_modes), Ok(surface_formats)) => {
                        Ok(
                            Self {
                                device,
                                surface,
                                surface_capabilities,
                                surface_present_modes,
                                surface_formats,
                            }
                        )
                    },
                    _ => Err(VulkanError::Unspecified)
                }
            },
            Nome => {
                Err(VulkanError::Unspecified)
            }
        }
    }
}

impl DeviceOwned for SwapchainKHR {
    fn get_parent_device(&self) -> Arc<Device> {
        self.device.clone()
    }
}

impl Drop for SwapchainKHR {
    fn drop(&mut self) {
        match self.device.ash_ext_swapchain_khr() {
            Some(ext) => {
                unsafe {
                    ext.destroy_swapchain(self.swapchain, self.device.get_parent_instance().get_alloc_callbacks())
                }
            },
            None => {
                panic!("Swapchain extension is not available anymore. This should not happend. If you read this main developer of this crate made something bad.");
            }
        }
        
    }
}

impl SwapchainKHR {
    
    pub fn new(
        device_info: &DeviceSurfaceInfo,
        queue_families: &[Arc<QueueFamily>],
        old_swapchain: Option<Arc<Self>>,
        present_mode: PresentModeSwapchainKHR,
        color_space: SurfaceColorspaceSwapchainKHR,
        clipped: bool,
        image_format: ImageFormat,
        image_usage: ImageUsage,
        extent: Image2DDimensions,
        image_count: u32,
        image_layers: u32,
    ) -> VulkanResult<Arc<Self>> {
        let queue_family_indexes: Vec<u32> = queue_families.iter().map(|family| family.get_family_index()).collect();

        let device = device_info.device.clone();
        let surface = device_info.surface.clone();

        match device.ash_ext_swapchain_khr() {
            Some(ext) => {

                assert!(device_info.format_supported(&color_space, &image_format) && device_info.present_mode_supported(&present_mode));

                let create_info = ash::vk::SwapchainCreateInfoKHR::builder()
                    .old_swapchain(
                        match &old_swapchain {
                            Some(old) => old.swapchain,
                            None => ash::vk::SwapchainKHR::from_raw(0)
                        }
                    )
                    .image_extent(
                        ash::vk::Extent2D::builder()
                            .height(extent.height())
                            .width(extent.width())
                            .build()
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
                    .build();
                
                //let surface_capabilities =

                match unsafe { ext.create_swapchain(&create_info, device.get_parent_instance().get_alloc_callbacks()) } {
                    Ok(swapchain) => {
                        // the old swapchain (or at least my handle of it) must be removed AFTER the creation of the new one, not BEFORE!
                        std::mem::drop(old_swapchain);
                        
                        Ok(Arc::new(Self {
                            device,
                            surface,
                            swapchain,
                            image_count,
                        }))
                    },
                    Err(err) => {
                        #[cfg(debug_assertions)]
                        {
                            panic!("Error creating the swapchain: {}", err)
                        }

                        Err(VulkanError::Unspecified)
                    }
                }
            },
            None => {
                return Err(VulkanError::MissingExtension(String::from("VK_KHR_swapchain")))
            }
        }
    }
}
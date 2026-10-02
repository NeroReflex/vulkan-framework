use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::Duration,
};

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
    prelude::{FrameworkError, VulkanError, VulkanResult},
    queue::Queue,
    queue_family::{QueueFamily, QueueFamilyOwned},
    semaphore::Semaphore,
    surface::Surface,
    swapchain_image::ImageSwapchainKHR,
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

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
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

pub(crate) fn image_count_supported(
    capabilities: &ash::vk::SurfaceCapabilitiesKHR,
    count: u32,
) -> bool {
    count >= capabilities.min_image_count
        && (capabilities.max_image_count == 0 || count <= capabilities.max_image_count)
}

pub(crate) fn image_extent(
    capabilities: &ash::vk::SurfaceCapabilitiesKHR,
    preferred: Image2DDimensions,
) -> Image2DDimensions {
    if capabilities.current_extent.width != u32::MAX {
        Image2DDimensions::new(
            capabilities.current_extent.width,
            capabilities.current_extent.height,
        )
    } else {
        Image2DDimensions::new(
            preferred.width().clamp(
                capabilities.min_image_extent.width,
                capabilities.max_image_extent.width,
            ),
            preferred.height().clamp(
                capabilities.min_image_extent.height,
                capabilities.max_image_extent.height,
            ),
        )
    }
}

pub(crate) fn validate_image_capabilities(
    capabilities: &ash::vk::SurfaceCapabilitiesKHR,
    usage: ImageUsage,
    extent: Image2DDimensions,
    min_image_count: u32,
    image_layers: u32,
    transform: SurfaceTransformSwapchainKHR,
    alpha: CompositeAlphaSwapchainKHR,
) -> VulkanResult<()> {
    if image_layers == 0 {
        return Err(FrameworkError::NoImageLayersSpecified.into());
    }
    if extent.width() == 0 || extent.height() == 0 || image_extent(capabilities, extent) != extent {
        return Err(FrameworkError::UnsuitableImageDimensions.into());
    }
    let usage: ash::vk::ImageUsageFlags = usage.into();
    if image_layers > capabilities.max_image_array_layers
        || !image_count_supported(capabilities, min_image_count)
        || usage.is_empty()
        || !capabilities.supported_usage_flags.contains(usage)
        || !capabilities
            .supported_transforms
            .contains(transform.ash_transform())
        || !capabilities
            .supported_composite_alpha
            .contains(alpha.ash_alpha())
    {
        return Err(ash::vk::Result::ERROR_UNKNOWN.into());
    }
    Ok(())
}

// Once vkCreateSwapchainKHR is called, oldSwapchain is retired even on failure.
// Always release the retired handle; release the replacement too if enumeration fails.
pub(crate) fn replace_native_swapchain(
    ext: &ash::khr::swapchain::Device,
    callbacks: Option<&ash::vk::AllocationCallbacks<'_>>,
    create_info: &ash::vk::SwapchainCreateInfoKHR<'_>,
    swapchain: &mut ash::vk::SwapchainKHR,
    images: &mut Vec<ash::vk::Image>,
) -> VulkanResult<()> {
    let old = std::mem::replace(swapchain, ash::vk::SwapchainKHR::null());
    images.clear();
    let result = unsafe { ext.create_swapchain(create_info, callbacks) }.and_then(|new| {
        match unsafe { ext.get_swapchain_images(new) } {
            Ok(new_images) if !new_images.is_empty() => Ok((new, new_images)),
            result => {
                unsafe { ext.destroy_swapchain(new, callbacks) };
                Err(result
                    .err()
                    .unwrap_or(ash::vk::Result::ERROR_INITIALIZATION_FAILED))
            }
        }
    });
    if old != ash::vk::SwapchainKHR::null() {
        unsafe { ext.destroy_swapchain(old, callbacks) };
    }
    let (new, new_images) = result?;
    *swapchain = new;
    *images = new_images;
    Ok(())
}

pub(crate) fn acquire_native_image(
    ext: &ash::khr::swapchain::Device,
    swapchain: ash::vk::SwapchainKHR,
    timeout: Duration,
    semaphore: ash::vk::Semaphore,
    fence: ash::vk::Fence,
) -> VulkanResult<(u32, bool)> {
    let (index, suboptimal) = unsafe {
        ext.acquire_next_image(
            swapchain,
            timeout.as_nanos().min(u64::MAX as u128) as u64,
            semaphore,
            fence,
        )
    }?;
    Ok((index, !suboptimal))
}

// Reserve only the host acquisition call. The synchronous API leaves waiting/resetting
// to the caller, and existing async waiters reset an unowned fence on completion.
// Keeping a submission reservation after returning would prevent either path resetting it.
pub(crate) struct AcquisitionFenceGuard<'a> {
    fence: &'a Fence,
}

impl<'a> AcquisitionFenceGuard<'a> {
    pub(crate) fn new(fence: &'a Fence) -> VulkanResult<Self> {
        fence.reserve_submission()?;
        Ok(Self { fence })
    }
}

impl Drop for AcquisitionFenceGuard<'_> {
    fn drop(&mut self) {
        let _ = self.fence.cancel_submission();
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
        image_count_supported(&self.surface_capabilities, count)
    }

    /// Use the surface's fixed current extent, or clamp the preferred extent to its limits.
    /// A zero result (for example, a minimized window) cannot be used to create a swapchain.
    pub fn image_extent(&self, preferred: Image2DDimensions) -> Image2DDimensions {
        image_extent(&self.surface_capabilities, preferred)
    }

    pub fn transform_supported(&self, transform: &SurfaceTransformSwapchainKHR) -> bool {
        self.surface_capabilities
            .supported_transforms
            .contains(transform.ash_transform())
    }

    pub fn composite_alpha_supported(&self, alpha: &CompositeAlphaSwapchainKHR) -> bool {
        self.surface_capabilities
            .supported_composite_alpha
            .contains(alpha.ash_alpha())
    }

    /// The surface's current presentation transform.
    pub fn current_transform(&self) -> SurfaceTransformSwapchainKHR {
        // Vulkan reports exactly one of these bits as currentTransform.
        [
            SurfaceTransformSwapchainKHR::Identity,
            SurfaceTransformSwapchainKHR::Rotate90,
            SurfaceTransformSwapchainKHR::Rotate180,
            SurfaceTransformSwapchainKHR::Rotate270,
            SurfaceTransformSwapchainKHR::HorizontalMirror,
            SurfaceTransformSwapchainKHR::HorizontalMirrorRotate90,
            SurfaceTransformSwapchainKHR::HorizontalMirrorRotate180,
            SurfaceTransformSwapchainKHR::HorizontalMirrorRotate270,
            SurfaceTransformSwapchainKHR::Inherit,
        ]
        .into_iter()
        .find(|transform| transform.ash_transform() == self.surface_capabilities.current_transform)
        .expect("Vulkan returned an invalid current surface transform")
    }

    #[inline]
    pub fn surface(&self) -> Arc<Surface> {
        self.surface.clone()
    }

    /**
     * Result is either 0 (no limits) or a number >= min_image_count()
     *
     * Make sure to use the function image_count_supported to check if the desired number is supported!
     */
    #[inline]
    pub fn max_image_count(&self) -> u32 {
        self.surface_capabilities.max_image_count
    }

    #[inline]
    pub fn min_image_count(&self) -> u32 {
        self.surface_capabilities.min_image_count
    }

    #[inline]
    pub fn present_mode_supported(&self, mode: &PresentModeSwapchainKHR) -> bool {
        self.surface_present_modes.contains(&mode.ash_value())
    }

    #[inline]
    pub fn format_supported(
        &self,
        color_space: &SurfaceColorspaceSwapchainKHR,
        format: &ImageFormat,
    ) -> bool {
        let fmt = ash::vk::SurfaceFormatKHR::default()
            .format(format.to_owned().into())
            .color_space(color_space.ash_colorspace());

        self.surface_formats.iter().any(|supported| {
            supported.color_space == fmt.color_space
                && (supported.format == fmt.format
                    || supported.format == ash::vk::Format::UNDEFINED)
        })
    }

    pub fn new(device: Arc<Device>, surface: Arc<Surface>) -> VulkanResult<Self> {
        if device.get_parent_instance().native_handle()
            != surface.get_parent_instance().native_handle()
        {
            return Err(FrameworkError::ResourceFromIncompatibleDevice.into());
        }
        match device.get_parent_instance().get_surface_khr_extension() {
            Some(sfc_ext) => {
                let surface_capabilities = unsafe {
                    sfc_ext.get_physical_device_surface_capabilities(
                        device.ash_physical_device_handle().to_owned(),
                        surface.ash_handle().to_owned(),
                    )
                }?;

                let surface_present_modes = unsafe {
                    sfc_ext.get_physical_device_surface_present_modes(
                        device.ash_physical_device_handle().to_owned(),
                        surface.ash_handle().to_owned(),
                    )
                }?;

                let surface_formats = unsafe {
                    sfc_ext.get_physical_device_surface_formats(
                        device.ash_physical_device_handle().to_owned(),
                        surface.ash_handle().to_owned(),
                    )
                }?;

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
                "VK_KHR_surface",
            ))),
        }
    }
}

// Own the device's reservation until a fully initialized swapchain takes over.
pub(crate) struct SwapchainReservation<'a> {
    exists: &'a AtomicBool,
    pub(crate) committed: bool,
}

impl<'a> SwapchainReservation<'a> {
    pub(crate) fn new(exists: &'a AtomicBool) -> VulkanResult<Self> {
        exists
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .map_err(|_| FrameworkError::SwapchainAlreadyExists)?;
        Ok(Self {
            exists,
            committed: false,
        })
    }
}

impl Drop for SwapchainReservation<'_> {
    fn drop(&mut self) {
        if !self.committed {
            self.exists.store(false, Ordering::SeqCst);
        }
    }
}

type QueueFamiliesType = smallvec::SmallVec<[Arc<QueueFamily>; 4]>;

pub struct SwapchainKHR {
    device: Arc<Device>,
    queue_families: QueueFamiliesType,
    surface: Arc<Surface>,
    swapchain: ash::vk::SwapchainKHR,
    image_format: ImageFormat,
    image_usage: ImageUsage,
    extent: Image2DDimensions,
    transform: SurfaceTransformSwapchainKHR,
    composite_alpha: CompositeAlphaSwapchainKHR,
    min_image_count: u32,
    image_layers: u32,
    present_mode: PresentModeSwapchainKHR,
    color_space: SurfaceColorspaceSwapchainKHR,
    clipped: bool,
    images: Vec<ash::vk::Image>,
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
        let Some(ext) = self.device.ash_ext_swapchain_khr() else {
            panic!("Swapchain extension is not available anymore. This should not happend. If you read this main developer of this crate made something bad.");
        };

        if self.swapchain != ash::vk::SwapchainKHR::null() {
            unsafe {
                ext.destroy_swapchain(
                    self.swapchain,
                    self.device.get_parent_instance().get_alloc_callbacks(),
                )
            }
        }
        self.device.swapchain_exists.store(false, Ordering::SeqCst);
    }
}

impl SwapchainKHR {
    #[inline]
    pub fn queue_families(&self) -> &[Arc<QueueFamily>] {
        self.queue_families.as_slice()
    }

    #[inline]
    pub(crate) fn ash_handle(&self) -> ash::vk::SwapchainKHR {
        self.swapchain
    }

    /// Get a new reference to the swapchain image with the provided index.
    /// The caller should avoid calling this at every frame,
    /// and should opt to cache resulting images.
    pub fn image(swapchain: Arc<Self>, index: u32) -> VulkanResult<Arc<ImageSwapchainKHR>> {
        let requested_index = index as usize;
        let Some(image_handle) = swapchain.images.get(requested_index) else {
            return Err(VulkanError::Framework(
                FrameworkError::InvalidSwapchainImageIndex(requested_index, swapchain.images.len()),
            ));
        };

        let swapchain_cloned = swapchain.clone();
        let image = ImageSwapchainKHR::new(
            swapchain_cloned,
            ImageFlags::Unmanaged(0),
            swapchain.images_usage(),
            swapchain.images_format(),
            swapchain.images_extent(),
            swapchain.images_layers_count(),
            1,
            ash::vk::Image::from_raw(image_handle.as_raw()),
        );

        Ok(Arc::new(image))
    }

    #[inline]
    pub fn surface(&self) -> Arc<Surface> {
        self.surface.clone()
    }

    #[inline]
    pub fn transform(&self) -> SurfaceTransformSwapchainKHR {
        self.transform
    }

    #[inline]
    pub fn composite_alpha(&self) -> CompositeAlphaSwapchainKHR {
        self.composite_alpha
    }

    /// The minimum image count requested at creation, not necessarily the actual count.
    #[inline]
    pub fn min_image_count(&self) -> u32 {
        self.min_image_count
    }

    /// The actual number of images allocated by the driver (possibly above the minimum).
    /// Returns zero after a failed recreation that retired the previous swapchain.
    #[inline]
    pub fn images_count(&self) -> u32 {
        self.images.len() as u32
    }

    #[inline]
    pub fn images_flags(&self) -> ImageFlags {
        ImageFlags::Unmanaged(0)
    }

    #[inline]
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

    /// Present an acquired image, waiting on binary semaphores from this device.
    /// Returns `true` for a suboptimal presentation, and `false` for an optimal one.
    /// Unlike acquisition, this follows Vulkan/ash's suboptimal boolean convention.
    /// The caller must keep wait semaphores alive until presentation has consumed them.
    pub fn queue_present(
        &self,
        queue: Arc<Queue>,
        index: u32,
        semaphores: &[Arc<Semaphore>],
    ) -> VulkanResult<bool> {
        self.ensure_live()?;
        let family = queue.get_parent_queue_family();
        if self.device != family.get_parent_device() {
            return Err(FrameworkError::ResourceFromIncompatibleDevice.into());
        }
        if index as usize >= self.images.len() {
            return Err(FrameworkError::InvalidSwapchainImageIndex(
                index as usize,
                self.images.len(),
            )
            .into());
        }
        if !self
            .queue_families
            .iter()
            .any(|supported| supported.get_family_index() == family.get_family_index())
            || !self.family_supports_present(&family)?
        {
            return Err(ash::vk::Result::ERROR_UNKNOWN.into());
        }

        let mut native_semaphores = smallvec::SmallVec::<[ash::vk::Semaphore; 8]>::new();
        for semaphore in semaphores {
            Self::validate_semaphore(&self.device, semaphore)?;
            if native_semaphores.contains(&semaphore.ash_handle()) {
                return Err(ash::vk::Result::ERROR_UNKNOWN.into());
            }
            native_semaphores.push(semaphore.ash_handle());
        }

        let swapchains = [self.ash_handle()];
        let indexes = [index];
        let present_info = ash::vk::PresentInfoKHR::default()
            .swapchains(&swapchains)
            .image_indices(&indexes)
            .wait_semaphores(native_semaphores.as_slice());

        let ext = self
            .device
            .ash_ext_swapchain_khr()
            .as_ref()
            .ok_or_else(|| VulkanError::MissingExtension(String::from("VK_KHR_swapchain")))?;
        let _guard = queue.lock()?;
        Ok(unsafe { ext.queue_present(queue.ash_handle(), &present_info) }?)
    }

    /// Acquire an image and asynchronously wait for its fence. The result boolean is
    /// `true` for an optimal image and `false` for a suboptimal image.
    ///
    /// Acquisition itself may block up to `timeout`; only the subsequent fence wait is async.
    /// The waiter retains the fence and optional semaphore until completion and resets the
    /// fence. Do not reset/reuse the fence through other references until then. Keep the
    /// swapchain alive and do not recreate it while acquisition is pending. The existing
    /// waiter is not cancellation-safe: await it to completion rather than dropping it.
    #[cfg(feature = "async")]
    pub fn async_threaded_acquire_next_image_index(
        &self,
        pool: Arc<ThreadPool>,
        timeout: Duration,
        maybe_semaphore: Option<Arc<Semaphore>>,
        fence: Arc<Fence>,
    ) -> VulkanResult<ThreadedFenceWaiter<(u32, bool)>> {
        let result =
            self.acquire_next_image_index(timeout, maybe_semaphore.clone(), Some(fence.clone()))?;
        let semaphores: Vec<_> = maybe_semaphore.into_iter().collect();
        Ok(ThreadedFenceWaiter::new(
            pool,
            None,
            &[],
            &semaphores,
            fence,
            result,
        ))
    }

    /// Acquire an image and asynchronously poll its fence. The result boolean is
    /// `true` for an optimal image and `false` for a suboptimal image.
    ///
    /// Acquisition itself may block up to `timeout`; only the subsequent fence wait is async.
    /// The waiter retains the fence and optional semaphore until completion and resets the
    /// fence. Do not reset/reuse the fence through other references until then. Keep the
    /// swapchain alive and do not recreate it while acquisition is pending. The existing
    /// waiter is not cancellation-safe: await it to completion rather than dropping it.
    #[cfg(feature = "async")]
    pub fn async_spinlock_acquire_next_image_index(
        &self,
        timeout: Duration,
        maybe_semaphore: Option<Arc<Semaphore>>,
        fence: Arc<Fence>,
    ) -> VulkanResult<SpinlockFenceWaiter<(u32, bool)>> {
        let result =
            self.acquire_next_image_index(timeout, maybe_semaphore.clone(), Some(fence.clone()))?;
        let semaphores: Vec<_> = maybe_semaphore.into_iter().collect();
        Ok(SpinlockFenceWaiter::new(
            None,
            &[],
            &semaphores,
            fence,
            result,
        ))
    }

    /// Retrieve the index of the next available presentable image.
    ///
    /// Waits up to `timeout` for an image, signaling the optional binary semaphore
    /// and/or unsignaled fence, which must belong to this device. At least one is required.
    /// Keep the swapchain and synchronization objects alive until acquisition completes.
    /// The fence must not be owned by a queue submission. Its host use is reserved during
    /// the native acquisition call, but this API does not return a fence-ownership token:
    /// after it returns, the caller must not reset/reuse the fence until it has signaled.
    /// Externally synchronize semaphore/fence aliases; a semaphore must be unsignaled and
    /// have no pending signal or wait operations.
    ///
    /// Returns `(index, optimal)`: `true` means optimal, `false` means suboptimal.
    /// This intentionally inverts Vulkan/ash's suboptimal boolean convention.
    pub fn acquire_next_image_index(
        &self,
        timeout: Duration,
        maybe_semaphore: Option<Arc<Semaphore>>,
        maybe_fence: Option<Arc<Fence>>,
    ) -> VulkanResult<(u32, bool)> {
        self.ensure_live()?;
        if maybe_semaphore.is_none() && maybe_fence.is_none() {
            return Err(ash::vk::Result::ERROR_UNKNOWN.into());
        }
        if let Some(semaphore) = &maybe_semaphore {
            Self::validate_semaphore(&self.device, semaphore)?;
        }
        if let Some(fence) = &maybe_fence {
            if fence.get_parent_device() != self.device {
                return Err(FrameworkError::ResourceFromIncompatibleDevice.into());
            }
        }
        let ext = self
            .device
            .ash_ext_swapchain_khr()
            .as_ref()
            .ok_or_else(|| VulkanError::MissingExtension(String::from("VK_KHR_swapchain")))?;
        let _fence_guard = maybe_fence
            .as_deref()
            .map(AcquisitionFenceGuard::new)
            .transpose()?;
        acquire_native_image(
            ext,
            self.swapchain,
            timeout,
            maybe_semaphore
                .as_ref()
                .map_or(ash::vk::Semaphore::null(), |sem| sem.ash_handle()),
            maybe_fence
                .as_ref()
                .map_or(ash::vk::Fence::null(), |fence| fence.ash_handle()),
        )
    }

    /// Recreate using fresh surface capabilities and the current surface extent.
    /// For surfaces with a variable extent, the previous image extent is the preference;
    /// use `recreate_with_extent` to provide the new drawable size instead.
    ///
    /// Before calling, wait for all GPU work and call `Device::wait_idle()` to finish
    /// presentation, then drop cached swapchain images, image views and command-buffer
    /// references so that `Arc::get_mut()` can provide exclusive access to the swapchain.
    /// Rebuild those resources after successful recreation; image count may change.
    ///
    /// Capability/query/validation errors leave the old swapchain untouched. A driver
    /// creation/enumeration error retires and destroys the old swapchain: this wrapper
    /// then has no images, rejects acquisition/presentation with `ERROR_OUT_OF_DATE_KHR`,
    /// and retains its device reservation until a retry succeeds or it is dropped.
    pub fn recreate(&mut self) -> VulkanResult<()> {
        self.recreate_with_extent(self.extent)
    }

    /// Recreate with a preferred drawable extent, honoring fresh current/min/max extents.
    /// Retains the present mode, format, color space, usage, layers and clipping preference.
    /// Adjusts the minimum image count to fresh limits and retains transform/alpha when
    /// supported, otherwise choosing the current transform and a supported alpha mode.
    /// Has the same idle/resource-release requirements and failure semantics as `recreate`.
    /// A zero current extent is rejected; retry once the window is drawable again.
    pub fn recreate_with_extent(&mut self, preferred: Image2DDimensions) -> VulkanResult<()> {
        let info = DeviceSurfaceInfo::new(self.device.clone(), self.surface.clone())?;
        let extent = info.image_extent(preferred);
        let mut min_image_count = self.min_image_count.max(info.min_image_count());
        if info.max_image_count() != 0 {
            min_image_count = min_image_count.min(info.max_image_count());
        }
        let transform = if info.transform_supported(&self.transform) {
            self.transform
        } else {
            info.current_transform()
        };
        let alpha = if info.composite_alpha_supported(&self.composite_alpha) {
            self.composite_alpha
        } else {
            [
                CompositeAlphaSwapchainKHR::Opaque,
                CompositeAlphaSwapchainKHR::PreMultiplied,
                CompositeAlphaSwapchainKHR::PostMultiplied,
                CompositeAlphaSwapchainKHR::Inherit,
            ]
            .into_iter()
            .find(|alpha| info.composite_alpha_supported(alpha))
            .ok_or(ash::vk::Result::ERROR_UNKNOWN)?
        };
        self.replace_swapchain(&info, extent, min_image_count, transform, alpha)
    }

    fn ensure_live(&self) -> VulkanResult<()> {
        if self.swapchain == ash::vk::SwapchainKHR::null() {
            return Err(ash::vk::Result::ERROR_OUT_OF_DATE_KHR.into());
        }
        Ok(())
    }

    pub(crate) fn validate_semaphore(
        device: &Arc<Device>,
        semaphore: &Semaphore,
    ) -> VulkanResult<()> {
        if semaphore.get_parent_device() != *device {
            return Err(FrameworkError::ResourceFromIncompatibleDevice.into());
        }
        if semaphore.is_timeline() {
            return Err(ash::vk::Result::ERROR_UNKNOWN.into());
        }
        Ok(())
    }

    fn family_supports_present(&self, family: &QueueFamily) -> VulkanResult<bool> {
        let instance = self.device.get_parent_instance();
        let ext = instance
            .get_surface_khr_extension()
            .ok_or_else(|| VulkanError::MissingExtension(String::from("VK_KHR_surface")))?;
        Ok(unsafe {
            ext.get_physical_device_surface_support(
                *self.device.ash_physical_device_handle(),
                family.get_family_index(),
                *self.surface.ash_handle(),
            )
        }?)
    }

    fn replace_swapchain(
        &mut self,
        info: &DeviceSurfaceInfo,
        extent: Image2DDimensions,
        min_image_count: u32,
        transform: SurfaceTransformSwapchainKHR,
        alpha: CompositeAlphaSwapchainKHR,
    ) -> VulkanResult<()> {
        if !info.format_supported(&self.color_space, &self.image_format) {
            return Err(ash::vk::Result::ERROR_FORMAT_NOT_SUPPORTED.into());
        }
        if !info.present_mode_supported(&self.present_mode) || self.queue_families.is_empty() {
            return Err(ash::vk::Result::ERROR_UNKNOWN.into());
        }
        validate_image_capabilities(
            &info.surface_capabilities,
            self.image_usage,
            extent,
            min_image_count,
            self.image_layers,
            transform,
            alpha,
        )?;
        let mut indexes = Vec::with_capacity(self.queue_families.len());
        let mut can_present = false;
        for family in &self.queue_families {
            if family.get_parent_device() != self.device {
                return Err(FrameworkError::ResourceFromIncompatibleDevice.into());
            }
            let index = family.get_family_index();
            if indexes.contains(&index) {
                return Err(ash::vk::Result::ERROR_UNKNOWN.into());
            }
            indexes.push(index);
            can_present |= self.family_supports_present(family)?;
        }
        if !can_present {
            return Err(ash::vk::Result::ERROR_UNKNOWN.into());
        }
        let create_info = ash::vk::SwapchainCreateInfoKHR::default()
            .surface(*self.surface.ash_handle())
            .old_swapchain(self.swapchain)
            .image_extent(ash::vk::Extent2D {
                width: extent.width(),
                height: extent.height(),
            })
            .queue_family_indices(&indexes)
            .image_sharing_mode(if indexes.len() == 1 {
                ash::vk::SharingMode::EXCLUSIVE
            } else {
                ash::vk::SharingMode::CONCURRENT
            })
            .image_usage(self.image_usage.into())
            .image_array_layers(self.image_layers)
            .image_format(self.image_format.into())
            .image_color_space(self.color_space.ash_colorspace())
            .present_mode(self.present_mode.ash_value())
            .clipped(self.clipped)
            .pre_transform(transform.ash_transform())
            .composite_alpha(alpha.ash_alpha())
            .min_image_count(min_image_count);
        let ext = self
            .device
            .ash_ext_swapchain_khr()
            .as_ref()
            .ok_or_else(|| VulkanError::MissingExtension(String::from("VK_KHR_swapchain")))?;
        replace_native_swapchain(
            ext,
            self.device.get_parent_instance().get_alloc_callbacks(),
            &create_info,
            &mut self.swapchain,
            &mut self.images,
        )?;
        self.extent = extent;
        self.min_image_count = min_image_count;
        self.transform = transform;
        self.composite_alpha = alpha;
        Ok(())
    }

    /// Create a swapchain after validating surface capabilities and queue families.
    /// Use `DeviceSurfaceInfo::image_extent` to resolve a preferred drawable size before
    /// passing `extent`; invalid or zero extents are rejected rather than silently changed.
    /// `min_image_count` is a request: use `images_count()` for the allocated image count.
    pub fn new(
        device_info: &DeviceSurfaceInfo,
        queue_families: &[Arc<QueueFamily>],
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
        if device_info.device.ash_ext_swapchain_khr().is_none() {
            return Err(VulkanError::MissingExtension(String::from(
                "VK_KHR_swapchain",
            )));
        }
        let mut reservation = SwapchainReservation::new(&device_info.device.swapchain_exists)?;
        let mut swapchain = Self {
            device: device_info.device.clone(),
            queue_families: queue_families.iter().cloned().collect(),
            surface: device_info.surface.clone(),
            swapchain: ash::vk::SwapchainKHR::null(),
            min_image_count,
            transform,
            composite_alpha,
            image_format,
            image_usage,
            extent,
            image_layers,
            present_mode,
            color_space,
            clipped,
            images: Vec::new(),
        };
        // From here, SwapchainKHR::drop releases the reservation on every error path.
        reservation.committed = true;
        swapchain.replace_swapchain(
            device_info,
            extent,
            min_image_count,
            transform,
            composite_alpha,
        )?;
        Ok(Arc::new(swapchain))
    }
}

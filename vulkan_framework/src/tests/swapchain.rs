use super::common;
use crate::{
    fence::Fence,
    image::Image2DDimensions,
    prelude::{FrameworkError, VulkanError, VulkanResult},
    semaphore::Semaphore,
    swapchain::{
        acquire_native_image, image_count_supported, image_extent, replace_native_swapchain,
        validate_image_capabilities, AcquisitionFenceGuard, CompositeAlphaSwapchainKHR,
        SurfaceTransformSwapchainKHR, SwapchainKHR, SwapchainReservation,
    },
};
use ash::vk::{self, Handle};
use std::{
    cell::RefCell,
    ffi::{c_void, CStr},
    sync::atomic::{AtomicBool, Ordering},
    time::Duration,
};

fn capabilities() -> vk::SurfaceCapabilitiesKHR {
    vk::SurfaceCapabilitiesKHR::default()
        .min_image_count(2)
        .max_image_count(4)
        .current_extent(vk::Extent2D {
            width: u32::MAX,
            height: u32::MAX,
        })
        .min_image_extent(vk::Extent2D {
            width: 64,
            height: 32,
        })
        .max_image_extent(vk::Extent2D {
            width: 1920,
            height: 1080,
        })
        .max_image_array_layers(2)
        .supported_usage_flags(vk::ImageUsageFlags::COLOR_ATTACHMENT)
        .supported_transforms(vk::SurfaceTransformFlagsKHR::IDENTITY)
        .current_transform(vk::SurfaceTransformFlagsKHR::IDENTITY)
        .supported_composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
}

#[test]
fn image_count_limit_is_inclusive_and_zero_maximum_is_unlimited() {
    let mut caps = capabilities();
    for count in [2, 3, 4] {
        assert!(image_count_supported(&caps, count));
    }
    for count in [0, 1, 5, u32::MAX] {
        assert!(!image_count_supported(&caps, count));
    }
    caps.max_image_count = 2;
    assert!(image_count_supported(&caps, 2));
    caps.max_image_count = 0;
    assert!(image_count_supported(&caps, u32::MAX));
    assert!(!image_count_supported(&caps, 1));
}

#[test]
fn extent_uses_current_extent_or_clamps_each_preferred_dimension() {
    let mut caps = capabilities();
    assert_eq!(
        image_extent(&caps, Image2DDimensions::new(800, 600)),
        Image2DDimensions::new(800, 600)
    );
    assert_eq!(
        image_extent(&caps, Image2DDimensions::new(0, u32::MAX)),
        Image2DDimensions::new(64, 1080)
    );
    assert_eq!(
        image_extent(&caps, Image2DDimensions::new(u32::MAX, 0)),
        Image2DDimensions::new(1920, 32)
    );
    caps.current_extent = vk::Extent2D {
        width: 1024,
        height: 768,
    };
    assert_eq!(
        image_extent(&caps, Image2DDimensions::new(800, 600)),
        Image2DDimensions::new(1024, 768)
    );
    caps.current_extent = vk::Extent2D {
        width: 0,
        height: 0,
    };
    assert_eq!(
        image_extent(&caps, Image2DDimensions::new(800, 600)),
        Image2DDimensions::new(0, 0)
    );
}

fn validate(
    caps: &vk::SurfaceCapabilitiesKHR,
    extent: Image2DDimensions,
    count: u32,
    layers: u32,
    usage: vk::ImageUsageFlags,
) -> VulkanResult<()> {
    validate_image_capabilities(
        caps,
        usage.as_raw().into(),
        extent,
        count,
        layers,
        SurfaceTransformSwapchainKHR::Identity,
        CompositeAlphaSwapchainKHR::Opaque,
    )
}

#[test]
fn creation_validates_counts_layers_usage_extents_and_supported_bits() {
    let mut caps = capabilities();
    let extent = Image2DDimensions::new(800, 600);
    let usage = vk::ImageUsageFlags::COLOR_ATTACHMENT;
    assert!(validate(&caps, extent, 4, 2, usage).is_ok());
    for count in [0, 1, 5] {
        assert!(validate(&caps, extent, count, 1, usage).is_err());
    }
    for layers in [0, 3] {
        assert!(validate(&caps, extent, 2, layers, usage).is_err());
    }
    for usage in [
        vk::ImageUsageFlags::empty(),
        usage | vk::ImageUsageFlags::STORAGE,
    ] {
        assert!(validate(&caps, extent, 2, 1, usage).is_err());
    }
    for extent in [
        Image2DDimensions::new(0, 600),
        Image2DDimensions::new(800, 0),
        Image2DDimensions::new(63, 600),
        Image2DDimensions::new(800, 1081),
    ] {
        assert!(validate(&caps, extent, 2, 1, usage).is_err());
    }
    caps.current_extent = vk::Extent2D {
        width: 1024,
        height: 768,
    };
    assert!(validate(&caps, extent, 2, 1, usage).is_err());
    caps.current_extent = vk::Extent2D {
        width: 0,
        height: 0,
    };
    assert!(validate(&caps, Image2DDimensions::new(0, 0), 2, 1, usage).is_err());
    caps = capabilities();
    caps.supported_transforms = vk::SurfaceTransformFlagsKHR::ROTATE_90;
    assert!(validate(&caps, extent, 2, 1, usage).is_err());
    caps = capabilities();
    caps.supported_composite_alpha = vk::CompositeAlphaFlagsKHR::INHERIT;
    assert!(validate(&caps, extent, 2, 1, usage).is_err());
}

#[test]
fn reservation_rolls_back_errors_and_does_not_clear_another_owner() {
    let exists = AtomicBool::new(false);
    {
        let _reservation = SwapchainReservation::new(&exists).unwrap();
        assert!(exists.load(Ordering::SeqCst));
        assert!(matches!(
            SwapchainReservation::new(&exists),
            Err(VulkanError::Framework(
                FrameworkError::SwapchainAlreadyExists
            ))
        ));
        assert!(exists.load(Ordering::SeqCst));
    }
    assert!(!exists.load(Ordering::SeqCst));
    let mut reservation = SwapchainReservation::new(&exists).unwrap();
    reservation.committed = true;
    drop(reservation);
    assert!(exists.load(Ordering::SeqCst));
}

// Mock only the swapchain extension dispatch: no Vulkan loader, window or engine runs.
struct MockDriver {
    create_result: vk::Result,
    images_result: vk::Result,
    acquire_result: vk::Result,
    image_count: u32,
    old_swapchains: Vec<vk::SwapchainKHR>,
    destroyed: Vec<vk::SwapchainKHR>,
    acquire_args: Option<(u64, vk::Semaphore, vk::Fence)>,
}

impl Default for MockDriver {
    fn default() -> Self {
        Self {
            create_result: vk::Result::SUCCESS,
            images_result: vk::Result::SUCCESS,
            acquire_result: vk::Result::SUCCESS,
            image_count: 5,
            old_swapchains: Vec::new(),
            destroyed: Vec::new(),
            acquire_args: None,
        }
    }
}

thread_local! {
    static DRIVER: RefCell<MockDriver> = RefCell::new(MockDriver::default());
}

unsafe extern "system" fn create_swapchain(
    _device: vk::Device,
    info: *const vk::SwapchainCreateInfoKHR<'_>,
    _callbacks: *const vk::AllocationCallbacks<'_>,
    swapchain: *mut vk::SwapchainKHR,
) -> vk::Result {
    DRIVER.with(|driver| {
        let mut driver = driver.borrow_mut();
        driver.old_swapchains.push((*info).old_swapchain);
        if driver.create_result == vk::Result::SUCCESS {
            *swapchain = vk::SwapchainKHR::from_raw(20);
        }
        driver.create_result
    })
}

unsafe extern "system" fn destroy_swapchain(
    _device: vk::Device,
    swapchain: vk::SwapchainKHR,
    _callbacks: *const vk::AllocationCallbacks<'_>,
) {
    DRIVER.with(|driver| driver.borrow_mut().destroyed.push(swapchain));
}

unsafe extern "system" fn get_images(
    _device: vk::Device,
    _swapchain: vk::SwapchainKHR,
    count: *mut u32,
    images: *mut vk::Image,
) -> vk::Result {
    DRIVER.with(|driver| {
        let driver = driver.borrow();
        if driver.images_result != vk::Result::SUCCESS {
            return driver.images_result;
        }
        if !images.is_null() {
            for index in 0..driver.image_count.min(*count) {
                *images.add(index as usize) = vk::Image::from_raw(100 + index as u64);
            }
        }
        *count = driver.image_count;
        vk::Result::SUCCESS
    })
}

unsafe extern "system" fn acquire_image(
    _device: vk::Device,
    _swapchain: vk::SwapchainKHR,
    timeout: u64,
    semaphore: vk::Semaphore,
    fence: vk::Fence,
    index: *mut u32,
) -> vk::Result {
    DRIVER.with(|driver| {
        let mut driver = driver.borrow_mut();
        driver.acquire_args = Some((timeout, semaphore, fence));
        *index = 3;
        driver.acquire_result
    })
}

unsafe extern "system" fn get_device_proc_addr(
    _device: vk::Device,
    name: *const std::ffi::c_char,
) -> vk::PFN_vkVoidFunction {
    let address = match CStr::from_ptr(name).to_bytes() {
        b"vkCreateSwapchainKHR" => create_swapchain as *const (),
        b"vkDestroySwapchainKHR" => destroy_swapchain as *const (),
        b"vkGetSwapchainImagesKHR" => get_images as *const (),
        b"vkAcquireNextImageKHR" => acquire_image as *const (),
        _ => std::ptr::null(),
    };
    std::mem::transmute(address)
}

fn mock_extension() -> ash::khr::swapchain::Device {
    DRIVER.with(|driver| *driver.borrow_mut() = MockDriver::default());
    unsafe {
        let instance = ash::Instance::load_with(
            |name| {
                if name.to_bytes() == b"vkGetDeviceProcAddr" {
                    get_device_proc_addr as *const () as *const c_void
                } else {
                    std::ptr::null()
                }
            },
            vk::Instance::from_raw(1),
        );
        let device = ash::Device::load_with(|_| std::ptr::null(), vk::Device::from_raw(1));
        ash::khr::swapchain::Device::new(&instance, &device)
    }
}

fn replace(
    ext: &ash::khr::swapchain::Device,
    handle: &mut vk::SwapchainKHR,
    images: &mut Vec<vk::Image>,
) -> VulkanResult<()> {
    let info = vk::SwapchainCreateInfoKHR::default()
        .old_swapchain(*handle)
        .min_image_count(2);
    replace_native_swapchain(ext, None, &info, handle, images)
}

#[test]
fn successful_replacement_releases_only_old_and_uses_actual_driver_image_count() {
    let ext = mock_extension();
    let old = vk::SwapchainKHR::from_raw(10);
    let mut handle = old;
    let mut images = vec![vk::Image::from_raw(1)];
    replace(&ext, &mut handle, &mut images).unwrap();
    assert_eq!(handle, vk::SwapchainKHR::from_raw(20));
    assert_eq!(images.len(), 5); // The driver allocated more than the requested minimum of 2.
    assert_eq!(images[4], vk::Image::from_raw(104));
    DRIVER.with(|driver| {
        let driver = driver.borrow();
        assert_eq!(driver.old_swapchains, [old]);
        assert_eq!(driver.destroyed, [old]);
    });
}

#[test]
fn create_failure_releases_old_and_retry_uses_null_old_swapchain() {
    let ext = mock_extension();
    let old = vk::SwapchainKHR::from_raw(10);
    let mut handle = old;
    let mut images = vec![vk::Image::from_raw(1)];
    DRIVER.with(|driver| driver.borrow_mut().create_result = vk::Result::ERROR_OUT_OF_DATE_KHR);
    assert!(matches!(
        replace(&ext, &mut handle, &mut images),
        Err(VulkanError::Vulkan(vk::Result::ERROR_OUT_OF_DATE_KHR))
    ));
    assert_eq!(handle, vk::SwapchainKHR::null());
    assert!(images.is_empty());
    DRIVER.with(|driver| {
        let mut driver = driver.borrow_mut();
        assert_eq!(driver.destroyed, [old]);
        driver.create_result = vk::Result::SUCCESS;
    });
    replace(&ext, &mut handle, &mut images).unwrap();
    DRIVER.with(|driver| {
        let driver = driver.borrow();
        assert_eq!(driver.old_swapchains, [old, vk::SwapchainKHR::null()]);
        assert_eq!(driver.destroyed, [old]);
    });
}

#[test]
fn enumeration_failure_releases_both_handles_and_clears_images() {
    let ext = mock_extension();
    let old = vk::SwapchainKHR::from_raw(10);
    let mut handle = old;
    let mut images = vec![vk::Image::from_raw(1)];
    DRIVER.with(|driver| driver.borrow_mut().images_result = vk::Result::ERROR_OUT_OF_HOST_MEMORY);
    assert!(matches!(
        replace(&ext, &mut handle, &mut images),
        Err(VulkanError::Vulkan(vk::Result::ERROR_OUT_OF_HOST_MEMORY))
    ));
    assert_eq!(handle, vk::SwapchainKHR::null());
    assert!(images.is_empty());
    DRIVER.with(|driver| {
        assert_eq!(
            driver.borrow().destroyed,
            [vk::SwapchainKHR::from_raw(20), old]
        )
    });
}

#[test]
fn initial_creation_errors_rollback_reservation_and_do_not_destroy_null() {
    for enumerate_failure in [false, true] {
        let ext = mock_extension();
        DRIVER.with(|driver| {
            let mut driver = driver.borrow_mut();
            if enumerate_failure {
                driver.images_result = vk::Result::ERROR_OUT_OF_HOST_MEMORY;
            } else {
                driver.create_result = vk::Result::ERROR_OUT_OF_HOST_MEMORY;
            }
        });
        let exists = AtomicBool::new(false);
        let mut handle = vk::SwapchainKHR::null();
        let mut images = Vec::new();
        let result = (|| {
            let _reservation = SwapchainReservation::new(&exists)?;
            replace(&ext, &mut handle, &mut images)
        })();
        assert!(result.is_err());
        assert!(!exists.load(Ordering::SeqCst));
        assert!(SwapchainReservation::new(&exists).is_ok());
        DRIVER.with(|driver| {
            let driver = driver.borrow();
            assert!(!driver.destroyed.contains(&vk::SwapchainKHR::null()));
            assert_eq!(driver.destroyed.len(), usize::from(enumerate_failure));
        });
    }
}

#[test]
fn empty_driver_image_list_does_not_leave_a_live_swapchain() {
    let ext = mock_extension();
    DRIVER.with(|driver| driver.borrow_mut().image_count = 0);
    let mut handle = vk::SwapchainKHR::null();
    let mut images = Vec::new();
    assert!(replace(&ext, &mut handle, &mut images).is_err());
    assert_eq!(handle, vk::SwapchainKHR::null());
    DRIVER.with(|driver| assert_eq!(driver.borrow().destroyed, [vk::SwapchainKHR::from_raw(20)]));
}

#[test]
fn acquisition_inverts_suboptimal_boolean_and_saturates_timeout() {
    let ext = mock_extension();
    let swapchain = vk::SwapchainKHR::from_raw(10);
    let semaphore = vk::Semaphore::from_raw(11);
    let fence = vk::Fence::from_raw(12);
    for (result, optimal) in [
        (vk::Result::SUCCESS, true),
        (vk::Result::SUBOPTIMAL_KHR, false),
    ] {
        DRIVER.with(|driver| driver.borrow_mut().acquire_result = result);
        assert_eq!(
            acquire_native_image(&ext, swapchain, Duration::MAX, semaphore, fence).unwrap(),
            (3, optimal)
        );
        DRIVER.with(|driver| {
            assert_eq!(
                driver.borrow().acquire_args,
                Some((u64::MAX, semaphore, fence))
            )
        });
    }
    DRIVER.with(|driver| driver.borrow_mut().acquire_result = vk::Result::TIMEOUT);
    assert!(
        acquire_native_image(&ext, swapchain, Duration::ZERO, semaphore, fence)
            .unwrap_err()
            .is_timeout()
    );
}

#[test]
fn acquisition_fence_guard_rejects_owned_and_signaled_fences() -> VulkanResult<()> {
    let (_instance, device) = common::setup_test_device()?;
    let fence = Fence::new(device.clone(), false, None)?;
    fence.reserve_submission()?;
    assert!(!fence.is_signaled()?);
    assert!(AcquisitionFenceGuard::new(&fence).is_err());
    // Rejection must not release the existing owner's reservation.
    assert!(fence.reset().is_err());
    fence.cancel_submission()?;

    let guard = AcquisitionFenceGuard::new(&fence)?;
    assert!(fence.reset().is_err());
    assert!(Fence::reset_fences(&[fence.clone()]).is_err());
    assert!(fence.reserve_submission().is_err());
    assert!(AcquisitionFenceGuard::new(&fence).is_err());
    drop(guard);
    fence.reset()?;

    let signaled = Fence::new(device, true, None)?;
    assert!(AcquisitionFenceGuard::new(&signaled).is_err());
    signaled.reset()?;
    Ok(())
}

#[test]
fn acquisition_releases_host_fence_reservation_on_success_and_error() -> VulkanResult<()> {
    let (_instance, device) = common::setup_test_device()?;
    let fence = Fence::new(device, false, None)?;
    let ext = mock_extension();
    for result in [
        vk::Result::SUCCESS,
        vk::Result::SUBOPTIMAL_KHR,
        vk::Result::TIMEOUT,
        vk::Result::ERROR_OUT_OF_DATE_KHR,
    ] {
        DRIVER.with(|driver| driver.borrow_mut().acquire_result = result);
        let acquired = (|| {
            let _guard = AcquisitionFenceGuard::new(&fence)?;
            assert!(fence.reset().is_err());
            acquire_native_image(
                &ext,
                vk::SwapchainKHR::from_raw(10),
                Duration::ZERO,
                vk::Semaphore::null(),
                fence.ash_handle(),
            )
        })();
        assert_eq!(
            acquired.is_ok(),
            result == vk::Result::SUCCESS || result == vk::Result::SUBOPTIMAL_KHR
        );
        // The mock never signals the fence; this checks only the host-call reservation.
        fence.reset()?;
        fence.reserve_submission()?;
        fence.cancel_submission()?;
    }
    Ok(())
}

#[test]
fn semaphore_validation_rejects_timeline_and_foreign_devices() -> VulkanResult<()> {
    let (_instance, device) = common::setup_test_device()?;
    let binary = Semaphore::new(device.clone(), None)?;
    let timeline = Semaphore::new_timeline(device.clone(), 0, None)?;
    assert!(SwapchainKHR::validate_semaphore(&device, &binary).is_ok());
    assert!(SwapchainKHR::validate_semaphore(&device, &timeline).is_err());
    let (_other_instance, other_device) = common::setup_test_device()?;
    assert!(matches!(
        SwapchainKHR::validate_semaphore(&other_device, &binary),
        Err(VulkanError::Framework(
            FrameworkError::ResourceFromIncompatibleDevice
        ))
    ));
    Ok(())
}

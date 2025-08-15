use crate::prelude::{FrameworkError, VulkanError, VulkanResult};
use crate::{instance::*, queue_family::*};

use ash;

use std::ffi::CString;
use std::os::raw::c_char;
use std::sync::atomic::AtomicBool;

#[cfg(feature = "better_mutex")]
use parking_lot::{const_mutex, Mutex};

#[cfg(not(feature = "better_mutex"))]
use std::sync::Mutex;

use std::vec::Vec;

use std::sync::Arc;

pub trait DeviceOwned {
    fn get_parent_device(&self) -> Arc<Device>;
}

struct DeviceExtensions {
    debug_utils_khr_ext: Option<Arc<ash::ext::debug_utils::Device>>,
    swapchain_khr_ext: Option<ash::khr::swapchain::Device>,
    deferred_host_operation_khr_ext: Option<ash::khr::deferred_host_operations::Device>,
    raytracing_pipeline_khr_ext: Option<ash::khr::ray_tracing_pipeline::Device>,
    raytracing_maintenance_khr_ext: Option<ash::khr::ray_tracing_maintenance1::Device>,
    acceleration_structure_khr_ext: Option<ash::khr::acceleration_structure::Device>,
}

struct DeviceData<'a> {
    selected_physical_device: ash::vk::PhysicalDevice,
    selected_device_features: ash::vk::PhysicalDeviceFeatures,
    selected_queues: Vec<ash::vk::DeviceQueueCreateInfo<'a>>,
    required_family_collection: Vec<Option<(u32, ConcreteQueueFamilyDescriptor)>>,
    supported_extension_names: Vec<String>,
    enabled_extensions: Vec<Arc<CString>>,
}

pub struct RaytracingInfo {
    min_acceleration_structure_scratch_offset_alignment: u32,
    shader_group_handle_size: u32,
    max_ray_dispatch_invocation_count: u32,
    max_ray_hit_attribute_size: u32,
    max_ray_recursion_depth: u32,
    max_shader_group_stride: u32,
    shader_group_base_alignment: u32,
    shader_group_handle_alignment: u32,
}

impl RaytracingInfo {
    pub fn print_info(&self) {
        println!(
            "    AccelerationStructure min_acceleration_structure_scratch_offset_alignment: {}",
            self.min_acceleration_structure_scratch_offset_alignment
        );
        println!(
            "    RayTracing shader_group_handle_size: {}",
            self.shader_group_handle_size
        );
        println!(
            "    RayTracing max_ray_dispatch_invocation_count: {}",
            self.max_ray_dispatch_invocation_count
        );
        println!(
            "    RayTracing max_ray_hit_attribute_size: {}",
            self.max_ray_hit_attribute_size
        );
        println!(
            "    RayTracing max_ray_recursion_depth: {}",
            self.max_ray_recursion_depth
        );
        println!(
            "    RayTracing max_shader_group_stride: {}",
            self.max_shader_group_stride
        );
        println!(
            "    RayTracing shader_group_base_alignment: {}",
            self.shader_group_base_alignment
        );
        println!(
            "    RayTracing shader_group_handle_alignment: {}",
            self.shader_group_handle_alignment
        );
    }

    pub(crate) fn from<'a, 'b>(
        accel_structure_properties: &ash::vk::PhysicalDeviceAccelerationStructurePropertiesKHR<'a>,
        ray_tracing_pipeline_properties: &ash::vk::PhysicalDeviceRayTracingPipelinePropertiesKHR<
            'b,
        >,
    ) -> Self {
        Self {
            min_acceleration_structure_scratch_offset_alignment: accel_structure_properties
                .min_acceleration_structure_scratch_offset_alignment,
            shader_group_handle_size: ray_tracing_pipeline_properties.shader_group_handle_size,
            max_ray_dispatch_invocation_count: ray_tracing_pipeline_properties
                .max_ray_dispatch_invocation_count,
            max_ray_hit_attribute_size: ray_tracing_pipeline_properties.max_ray_hit_attribute_size,
            max_ray_recursion_depth: ray_tracing_pipeline_properties.max_ray_recursion_depth,
            max_shader_group_stride: ray_tracing_pipeline_properties.max_shader_group_stride,
            shader_group_base_alignment: ray_tracing_pipeline_properties
                .shader_group_base_alignment,
            shader_group_handle_alignment: ray_tracing_pipeline_properties
                .shader_group_handle_alignment,
        }
    }

    pub fn min_acceleration_structure_scratch_offset_alignment(&self) -> u32 {
        self.min_acceleration_structure_scratch_offset_alignment
    }

    pub fn shader_group_handle_size(&self) -> u32 {
        self.shader_group_handle_size
    }

    pub fn max_ray_dispatch_invocation_count(&self) -> u32 {
        self.max_ray_dispatch_invocation_count
    }

    pub fn max_ray_hit_attribute_size(&self) -> u32 {
        self.max_ray_hit_attribute_size
    }

    pub fn max_ray_recursion_depth(&self) -> u32 {
        self.max_ray_recursion_depth
    }

    pub fn max_shader_group_stride(&self) -> u32 {
        self.max_shader_group_stride
    }

    pub fn shader_group_base_alignment(&self) -> u32 {
        self.shader_group_base_alignment
    }

    pub fn shader_group_handle_alignment(&self) -> u32 {
        self.shader_group_handle_alignment
    }
}

pub struct Device {
    //_name_bytes: Vec<u8>,
    required_family_collection: Mutex<Vec<Option<(u32, ConcreteQueueFamilyDescriptor)>>>,
    instance: Arc<Instance>,
    supported_extension_names: Vec<String>,
    extensions: DeviceExtensions,
    device: ash::Device,
    pub(crate) physical_device: ash::vk::PhysicalDevice,
    ray_tracing_info: Option<RaytracingInfo>,
    pub(crate) swapchain_exists: AtomicBool,
}

impl PartialEq for Device {
    fn eq(&self, other: &Self) -> bool {
        ash::vk::Handle::as_raw(self.device.handle())
            == ash::vk::Handle::as_raw(other.device.handle())
    }
}

impl InstanceOwned for Device {
    fn get_parent_instance(&self) -> Arc<Instance> {
        self.instance.clone()
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        self.wait_idle().unwrap();
        let alloc_callbacks = self.instance.get_alloc_callbacks();
        unsafe { self.device.destroy_device(alloc_callbacks) }
    }
}

impl Device {
    pub fn native_handle(&self) -> u64 {
        ash::vk::Handle::as_raw(self.device.handle())
    }

    pub(crate) fn ash_ext_debug_utils_ext(&self) -> &Option<Arc<ash::ext::debug_utils::Device>> {
        &self.extensions.debug_utils_khr_ext
    }

    pub(crate) fn ash_ext_swapchain_khr(&self) -> &Option<ash::khr::swapchain::Device> {
        &self.extensions.swapchain_khr_ext
    }

    pub(crate) fn ash_ext_raytracing_maintenance1_khr(
        &self,
    ) -> &Option<ash::khr::ray_tracing_maintenance1::Device> {
        &self.extensions.raytracing_maintenance_khr_ext
    }

    pub(crate) fn ash_ext_raytracing_pipeline_khr(
        &self,
    ) -> &Option<ash::khr::ray_tracing_pipeline::Device> {
        &self.extensions.raytracing_pipeline_khr_ext
    }

    pub(crate) fn ash_ext_deferred_host_operation_khr(
        &self,
    ) -> &Option<ash::khr::deferred_host_operations::Device> {
        &self.extensions.deferred_host_operation_khr_ext
    }

    pub(crate) fn ash_ext_acceleration_structure_khr(
        &self,
    ) -> &Option<ash::khr::acceleration_structure::Device> {
        &self.extensions.acceleration_structure_khr_ext
    }

    pub fn ray_tracing_info(&self) -> &Option<RaytracingInfo> {
        &self.ray_tracing_info
    }

    pub(crate) fn ash_physical_device_handle(&self) -> &ash::vk::PhysicalDevice {
        &self.physical_device
    }

    pub(crate) fn ash_handle(&self) -> &ash::Device {
        &self.device
    }

    pub fn supported_extensions(&self) -> Vec<String> {
        self.supported_extension_names.clone()
    }

    pub fn wait_idle(&self) -> VulkanResult<()> {
        Ok(unsafe { self.device.device_wait_idle() }?)
    }

    /**
     * Check if the given queue family supports at least specified operations.
     * The return value is a score representing a fit, 0 is for best-fit, the greather the number the worst is the fit.
     *
     * @param operations slice with the set of requested capabilities to support (framework)
     * @param instance the vulkan low-level instance (native)
     * @param device the vulkan low-level physical device (native)
     * @param queue_family the current queue family properties (native)
     * @param family_index the current queue family index (native)
     *
     * @return Some(score) iif all requested operations are supported for the given queue family, None otherwise.
     */
    fn corresponds<'a, I>(
        operations: I,
        _instance: &ash::Instance,
        device: &ash::vk::PhysicalDevice,
        surface_extension: Option<&ash::khr::surface::Instance>,
        queue_family: &ash::vk::QueueFamilyProperties,
        family_index: u32,
        _max_queues: u32,
    ) -> Option<u16>
    where
        I: Iterator<Item = &'a QueueFamilySupportedOperationType>,
    {
        /*
        // this was undocumented and I forgot what it was supposed to mean.
        if max_queues < queue_family.queue_count {
            return None;
        }
        */

        let mut score = 0;

        for feature in operations {
            // get an initial score based on support of stuff
            match queue_family
                .queue_flags
                .contains(ash::vk::QueueFlags::TRANSFER)
            {
                true => score += 1,
                false => score += 0,
            };

            match queue_family
                .queue_flags
                .contains(ash::vk::QueueFlags::COMPUTE)
            {
                true => score += 1,
                false => score += 0,
            };

            match queue_family
                .queue_flags
                .contains(ash::vk::QueueFlags::GRAPHICS)
            {
                true => score += 1,
                false => score += 0,
            };

            match feature {
                QueueFamilySupportedOperationType::Transfer => {
                    if !queue_family
                        .queue_flags
                        .contains(ash::vk::QueueFlags::TRANSFER)
                    {
                        return None;
                    }

                    score -= 1;
                }
                QueueFamilySupportedOperationType::Present(surface_handle) => {
                    match surface_extension {
                        Some(ext) => {
                            match unsafe {
                                ext.get_physical_device_surface_support(
                                    device.to_owned(),
                                    family_index,
                                    surface_handle.ash_handle().to_owned(),
                                )
                            } {
                                Ok(support) => match support {
                                    true => {
                                        score -= 1;
                                    }
                                    false => {
                                        return None;
                                    }
                                },
                                Err(_err) => {
                                    return None;
                                }
                            }
                        }
                        None => {
                            #[cfg(debug_assertions)]
                            {
                                println!("SurfaceKHR extension not available, have you forgotten to specify it on instance creation?")
                            }
                            return None;
                        }
                    }
                }
                QueueFamilySupportedOperationType::Graphics => {
                    if !queue_family
                        .queue_flags
                        .contains(ash::vk::QueueFlags::GRAPHICS)
                    {
                        return None;
                    }

                    score -= 1;
                }
                QueueFamilySupportedOperationType::Compute => {
                    if !queue_family
                        .queue_flags
                        .contains(ash::vk::QueueFlags::COMPUTE)
                    {
                        return None;
                    }

                    score -= 1;
                }
            }
        }

        Some(score)
    }

    fn ffi_strings(native_strings: &[String]) -> Vec<Arc<CString>> {
        native_strings
            .iter()
            .map(|ext_name| Arc::new(CString::new(ext_name.as_str()).unwrap()))
            .collect::<Vec<Arc<CString>>>()
    }

    fn strings_ffi(native_strings: Vec<Vec<std::ffi::c_char>>) -> Vec<String> {
        native_strings
            .iter()
            .map(|ext| match ext.iter().position(|n| *n == 0) {
                Some(ext_len) => String::from(unsafe {
                    std::str::from_utf8_unchecked(
                        std::ptr::slice_from_raw_parts(
                            ext.as_ptr() as *const u8,
                            ext_len.min(ext.len()),
                        )
                        .as_ref()
                        .unwrap(),
                    )
                }),
                None => panic!("Error in utf8"),
            })
            .collect()
    }

    /**
     * Creates a new device from the given instance if a suitable one is found.
     *
     * This method will fail if the given amount of queue families is not available or it is impossible to find
     * one or more suitable queue family for each family descriptor.
     *
     * For each requested extension that is also supported by the framework an handle will be created and the user
     * should be using it in case low-level operations are to be performed with native handles.
     *
     * If the device is created successfully the given queue family descriptor index will be used to obtain the
     * corresponding queue family on the opened device.
     *
     * Supported extensions are:
     *   - VK_KHR_swapchain
     *
     * @param instance a reference to the vulkan instance, will be stored inside the resulting Device (if any)
     * @param queue_descriptors a slice that specifies queue family minimal capabilities and maximum numer of queues that can be instantiated
     * @param device_extensions a list of device extensions to be enabled
     * @param device_layers a list of device layers to be enabled
     */
    pub fn new(
        instance: Arc<Instance>,
        queue_descriptors: &[ConcreteQueueFamilyDescriptor],
        device_extensions: &[String],
        debug_name: Option<&str>,
    ) -> VulkanResult<Arc<Self>> {
        // queue cannot be capable of nothing...
        if queue_descriptors.is_empty() {
            return Err(VulkanError::Framework(
                FrameworkError::MissingQueueDescriptor,
            ));
        }

        unsafe {
            let physical_devices = instance.ash_handle().enumerate_physical_devices()?;
            let mut best_physical_device_score: i128 = -1;
            let mut selected_physical_device: Option<DeviceData> = None;

            if physical_devices.is_empty() {
                return Err(VulkanError::Framework(
                    FrameworkError::NoSuitableDeviceFound,
                ));
            }

            let enabled_extensions = Self::ffi_strings(device_extensions);

            'suitable_device_search: for phy_device in physical_devices.iter() {
                let phy_device_features = instance
                    .ash_handle()
                    .get_physical_device_features(phy_device.to_owned());
                let phy_device_properties = instance
                    .ash_handle()
                    .get_physical_device_properties(phy_device.to_owned());
                let phy_device_extensions = instance
                    .ash_handle()
                    .enumerate_device_extension_properties(phy_device.to_owned());
                let phy_device_name = std::str::from_utf8_unchecked(
                    std::ptr::slice_from_raw_parts(
                        phy_device_properties.to_owned().device_name.as_ptr() as *const u8,
                        phy_device_properties.to_owned().device_name.len(),
                    )
                    .as_ref()
                    .unwrap(),
                );

                let supported_extension_names = match phy_device_extensions {
                    Ok(supported_extensions) => {
                        let supproted_extensions_map = supported_extensions
                            .iter()
                            .map(|f| f.extension_name.to_vec())
                            .collect::<Vec<Vec<i8>>>();

                        let supported_extensions_strings =
                            Self::strings_ffi(supproted_extensions_map);

                        for requested_extension in device_extensions.iter() {
                            if !supported_extensions_strings
                                .iter()
                                .any(|supported_ext| requested_extension == supported_ext)
                            {
                                #[cfg(debug_assertions)]
                                {
                                    println!("Requested extension {requested_extension} is not supported by physical device {phy_device_name}. This device won't be selected.");
                                }
                                continue 'suitable_device_search;
                            }
                        }

                        supported_extensions_strings
                    }
                    Err(err) => {
                        println!("Error enumerating device extensions for device {phy_device_name}: {err}. Will skip this device.");
                        continue 'suitable_device_search;
                    }
                };

                let msbytes = match phy_device_properties.device_type {
                    ash::vk::PhysicalDeviceType::DISCRETE_GPU => 0xC000000000000000u64,
                    ash::vk::PhysicalDeviceType::INTEGRATED_GPU => 0x8000000000000000u64,
                    ash::vk::PhysicalDeviceType::CPU => 0x4000000000000000u64,
                    _ => 0x0000000000000000u64,
                };

                let current_score = msbytes;

                // get queues properties (used to check if all requested queues are available on this device)
                let queue_family_properties = instance
                    .ash_handle()
                    .get_physical_device_queue_family_properties(phy_device.to_owned());

                // Check if all requested queues are supported
                let mut selected_queues: Vec<ash::vk::DeviceQueueCreateInfo> = vec![];
                let mut required_family_collection = vec![];

                let mut available_queue_families: Vec<(usize, &ash::vk::QueueFamilyProperties)> =
                    queue_family_properties.iter().enumerate().collect();

                for current_requested_queue_family_descriptor in queue_descriptors.iter() {
                    // this is the currently selected queue family (queue_family, score)
                    let mut selected_queue_family: Option<(usize, u16)> = None;

                    // the following for loop will search for the best fit for requested capabilities
                    /*'suitable_queue_family_search:*/
                    for (family_index, current_descriptor) in available_queue_families.clone() {
                        if let Some(score) = Self::corresponds(
                            current_requested_queue_family_descriptor
                                .get_supported_operations()
                                .iter(),
                            instance.ash_handle(),
                            phy_device,
                            instance.get_surface_khr_extension(),
                            current_descriptor,
                            family_index as u32,
                            current_requested_queue_family_descriptor.max_queues() as u32,
                        ) {
                            // Found a suitable queue family.
                            // Use this queue family if it's a better fit than the previous one
                            match selected_queue_family {
                                Some((_, best_fit_queue_score)) => {
                                    if best_fit_queue_score > score {
                                        selected_queue_family = Some((family_index, score))
                                    }
                                }
                                None => selected_queue_family = Some((family_index, score)),
                            }

                            // Stop the search. This changes the algorithm from a "best fit" to a "first fit"
                            //break 'suitable_queue_family_search;
                        }
                    }

                    // if any of the queue is not supported continue the search for a suitable device
                    // otherwise remove the current best fit from the queue of available queue_families to avoid choosing it two times
                    match selected_queue_family {
                        Some((family_index, _)) => {
                            let queue_create_info = ash::vk::DeviceQueueCreateInfo::default()
                                .queue_family_index(family_index as u32)
                                .queue_priorities(
                                    current_requested_queue_family_descriptor
                                        .get_queue_priorities(),
                                );

                            selected_queues.push(queue_create_info);
                            required_family_collection.push(Option::Some((
                                family_index as u32,
                                current_requested_queue_family_descriptor.clone(),
                            )));

                            available_queue_families = available_queue_families.iter().filter_map(|(queue_family_index, queue_family_properties)| -> Option<(usize, &ash::vk::QueueFamilyProperties)> {
                                        if *queue_family_index != family_index {
                                            return Some((*queue_family_index, queue_family_properties))
                                        }

                                        None
                                    }).collect();
                        }
                        None => {
                            println!("No suitable queue family found on device {phy_device_name}");
                            continue 'suitable_device_search;
                        }
                    }
                }

                let currently_selected_device_data = DeviceData {
                    selected_physical_device: *phy_device,
                    selected_device_features: phy_device_features,
                    selected_queues,
                    required_family_collection,
                    supported_extension_names,
                    enabled_extensions: enabled_extensions.clone(),
                };

                assert!(
                    currently_selected_device_data
                        .selected_device_features
                        .texture_compression_bc
                        != 0
                );

                #[cfg(debug_assertions)]
                {
                    println!("Found suitable device: {phy_device_name}");
                }

                match selected_physical_device {
                    Some(currently_selected_device) => {
                        if best_physical_device_score < current_score as i128 {
                            best_physical_device_score = current_score as i128;
                            selected_physical_device = Some(currently_selected_device_data);
                        } else {
                            selected_physical_device = Some(currently_selected_device);
                        }
                    }
                    None => {
                        best_physical_device_score = current_score as i128;
                        selected_physical_device = Some(currently_selected_device_data);
                    }
                }
            }

            let Some(selected_device) = selected_physical_device else {
                return Err(VulkanError::Framework(
                    FrameworkError::NoSuitableDeviceFound,
                ));
            };

            let extensions_ptr = selected_device
                .enabled_extensions
                .iter()
                .map(|str| str.as_ptr())
                .collect::<Vec<*const c_char>>();

            let mut device_create_info_builder = ash::vk::DeviceCreateInfo::default()
                .queue_create_infos(selected_device.selected_queues.as_slice())
                .enabled_extension_names(extensions_ptr.as_slice());

            let acceleration_structure_enabled = device_extensions.iter().any(|ext| {
                ext.as_str()
                    == ash::khr::acceleration_structure::NAME
                        .to_str()
                        .unwrap_or("")
            });

            let ray_tracing_enabled = device_extensions.iter().any(|ext| {
                ext.as_str() == ash::khr::ray_tracing_pipeline::NAME.to_str().unwrap_or("")
            });

            // prepare the list of features that the driver will fill, declaring what features it supports
            let mut features2 = ash::vk::PhysicalDeviceFeatures2::default();
            let mut get_synchronization2_features =
                ash::vk::PhysicalDeviceSynchronization2Features::default().synchronization2(false);
            let mut get_dynamic_rendering_features =
                ash::vk::PhysicalDeviceDynamicRenderingFeatures::default().dynamic_rendering(false);
            let mut accel_structure_features =
                ash::vk::PhysicalDeviceAccelerationStructureFeaturesKHR::default();
            let mut ray_tracing_pipeline_features =
                ash::vk::PhysicalDeviceRayTracingPipelineFeaturesKHR::default();
            let mut get_device_address_features =
                ash::vk::PhysicalDeviceBufferDeviceAddressFeatures::default();
            let mut get_imageless_framebuffer_features =
                ash::vk::PhysicalDeviceImagelessFramebufferFeatures::default();

            let mut properties2 = ash::vk::PhysicalDeviceProperties2::default();
            let mut accel_structure_properties =
                ash::vk::PhysicalDeviceAccelerationStructurePropertiesKHR::default();
            let mut ray_tracing_pipeline_properties =
                ash::vk::PhysicalDeviceRayTracingPipelinePropertiesKHR::default();
            ray_tracing_pipeline_properties.p_next = &mut accel_structure_properties
                as *mut ash::vk::PhysicalDeviceAccelerationStructurePropertiesKHR
                as *mut std::ffi::c_void;

            // Enable raytracing if required extensions have been requested
            if acceleration_structure_enabled {
                if ray_tracing_enabled {
                    properties2 = properties2.push_next(&mut ray_tracing_pipeline_properties);

                    accel_structure_features.p_next = &mut ray_tracing_pipeline_features
                        as *mut ash::vk::PhysicalDeviceRayTracingPipelineFeaturesKHR
                        as *mut std::ffi::c_void;

                    ray_tracing_pipeline_features.p_next = &mut get_device_address_features
                        as *mut ash::vk::PhysicalDeviceBufferDeviceAddressFeatures
                        as *mut std::ffi::c_void;
                }
                features2.p_next = &mut accel_structure_features as *mut _ as *mut std::ffi::c_void;
            }

            get_imageless_framebuffer_features.p_next = features2.p_next;
            features2.p_next =
                &mut get_imageless_framebuffer_features as *mut _ as *mut std::ffi::c_void;

            get_synchronization2_features.p_next = features2.p_next;
            features2.p_next =
                &mut get_synchronization2_features as *mut _ as *mut std::ffi::c_void;

            get_dynamic_rendering_features.p_next = get_synchronization2_features.p_next;
            get_synchronization2_features.p_next =
                &mut get_dynamic_rendering_features as *mut _ as *mut std::ffi::c_void;

            instance.ash_handle().get_physical_device_features2(
                selected_device.selected_physical_device,
                &mut features2,
            );
            instance.ash_handle().get_physical_device_properties2(
                selected_device.selected_physical_device,
                &mut properties2,
            );
            device_create_info_builder = device_create_info_builder.push_next(&mut features2);

            // make sure synchronization2 features are enabled: it is required for the framework to work
            assert!(get_synchronization2_features.synchronization2 != 0);

            // make sure dynamic rendering is enbaled
            assert!(get_dynamic_rendering_features.dynamic_rendering != 0);

            // nVidia cannot build the AS on the host
            //assert!(accel_structure_features.acceleration_structure_host_commands != 0);

            let mut raytracing_info: Option<RaytracingInfo> = Option::None;

            let device_create_info = device_create_info_builder;
            let device = instance.ash_handle().create_device(
                selected_device.selected_physical_device.to_owned(),
                &device_create_info,
                instance.get_alloc_callbacks(),
            )?;

            // open requested swapchain extensions (or implied ones)

            let debug_utils_ext = instance.get_debug_ext_extension().map(|_ext| {
                Arc::new(ash::ext::debug_utils::Device::new(
                    instance.ash_handle(),
                    &device,
                ))
            });

            let swapchain_ext = match device_extensions
                .iter()
                .any(|ext| ext.as_str() == ash::khr::swapchain::NAME.to_str().unwrap_or(""))
            {
                true => Option::Some(ash::khr::swapchain::Device::new(
                    instance.ash_handle(),
                    &device,
                )),
                false => Option::None,
            };

            let deferred_host_operation_ext: Option<ash::khr::deferred_host_operations::Device> =
                match ray_tracing_enabled {
                    true => {
                        raytracing_info = Some(RaytracingInfo::from(
                            &accel_structure_properties,
                            &ray_tracing_pipeline_properties,
                        ));

                        Option::Some(ash::khr::deferred_host_operations::Device::new(
                            instance.ash_handle(),
                            &device,
                        ))
                    }
                    false => Option::None,
                };

            let raytracing_pipeline_ext: Option<ash::khr::ray_tracing_pipeline::Device> =
                match ray_tracing_enabled {
                    true => {
                        raytracing_info = Some(RaytracingInfo::from(
                            &accel_structure_properties,
                            &ray_tracing_pipeline_properties,
                        ));

                        Option::Some(ash::khr::ray_tracing_pipeline::Device::new(
                            instance.ash_handle(),
                            &device,
                        ))
                    }
                    false => Option::None,
                };

            let acceleration_structure_ext: Option<ash::khr::acceleration_structure::Device> =
                match acceleration_structure_enabled {
                    true => Option::Some(ash::khr::acceleration_structure::Device::new(
                        instance.ash_handle(),
                        &device,
                    )),
                    false => Option::None,
                };

            let raytracing_maintenance_ext: Option<ash::khr::ray_tracing_maintenance1::Device> =
                match device_extensions.iter().any(|ext| {
                    ext.as_str()
                        == ash::khr::ray_tracing_maintenance1::NAME
                            .to_str()
                            .unwrap_or("")
                }) {
                    true => Option::Some(ash::khr::ray_tracing_maintenance1::Device::new(
                        instance.ash_handle(),
                        &device,
                    )),
                    false => Option::None,
                };

            let mut obj_name_bytes = vec![];

            if let Some(ext) = debug_utils_ext.clone() {
                if let Some(name) = debug_name {
                    for name_ch in name.as_bytes().iter() {
                        obj_name_bytes.push(*name_ch);
                    }
                    obj_name_bytes.push(0x00);

                    let object_name =
                        std::ffi::CStr::from_bytes_with_nul_unchecked(obj_name_bytes.as_slice());
                    // set device name for debugging
                    let dbg_info = ash::vk::DebugUtilsObjectNameInfoEXT::default()
                        .object_handle(device.handle())
                        .object_name(object_name);

                    if let Err(err) = ext.set_debug_utils_object_name(&dbg_info) {
                        #[cfg(debug_assertions)]
                        {
                            println!("Error setting the Debug name for the newly created Device, will use handle. Error: {}", err)
                        }
                    }
                }
            }

            #[cfg(not(feature = "better_mutex"))]
            let required_family_collection = Mutex::new(selected_device.required_family_collection);

            #[cfg(feature = "better_mutex")]
            let required_family_collection =
                const_mutex(selected_device.required_family_collection);

            Ok(Arc::new(Self {
                //_name_bytes: obj_name_bytes,
                required_family_collection,
                device,
                extensions: DeviceExtensions {
                    swapchain_khr_ext: swapchain_ext,
                    deferred_host_operation_khr_ext: deferred_host_operation_ext,
                    raytracing_pipeline_khr_ext: raytracing_pipeline_ext,
                    raytracing_maintenance_khr_ext: raytracing_maintenance_ext,
                    acceleration_structure_khr_ext: acceleration_structure_ext,
                    debug_utils_khr_ext: debug_utils_ext,
                },
                instance,
                supported_extension_names: selected_device.supported_extension_names,
                physical_device: selected_device.selected_physical_device,
                ray_tracing_info: raytracing_info,
                swapchain_exists: AtomicBool::new(false),
            }))
        }
    }

    pub(crate) fn move_out_queue_family(
        &self,
        index: usize,
    ) -> VulkanResult<(u32, ConcreteQueueFamilyDescriptor)> {
        #[cfg(feature = "better_mutex")]
        let mut collection = self.required_family_collection.lock();

        #[cfg(not(feature = "better_mutex"))]
        let mut collection = match self.required_family_collection.lock() {
            Ok(lock) => lock,
            Err(err) => {
                return Err(VulkanError::Framework(FrameworkError::MutexError(format!(
                    "{err}"
                ))))
            }
        };

        if collection.len() <= index {
            return Err(VulkanError::Framework(
                FrameworkError::TooManyQueueFamilies(index, collection.len()),
            ));
        }

        collection[index]
            .to_owned()
            .map(|cose| {
                collection[index] = None;
                Ok(cose)
            })
            .ok_or(VulkanError::Framework(
                FrameworkError::QueueFamilyAlreadyCreated(index),
            ))?
    }
}

use crate::memory_heap::{ConcreteMemoryHeapDescriptor, MemoryType};
use crate::prelude::{FrameworkError, VulkanError, VulkanResult};
use crate::{instance::*, queue_family::*};

use ash;

use std::os::raw::c_char;
use std::sync::Mutex;
use std::vec::Vec;

use std::sync::Arc;

pub trait DeviceOwned {
    fn get_parent_device(&self) -> Arc<Device>;
}

struct DeviceExtensions {
    swapchain_khr_ext: Option<ash::extensions::khr::Swapchain>,
    raytracing_pipeline_khr_ext: Option<ash::extensions::khr::RayTracingPipeline>,
    raytracing_maintenance_khr_ext: Option<ash::extensions::khr::RayTracingMaintenance1>,
    acceleration_structure_khr_ext: Option<ash::extensions::khr::AccelerationStructure>,
}

struct DeviceData {
    selected_physical_device: ash::vk::PhysicalDevice,
    selected_device_features: ash::vk::PhysicalDeviceFeatures,
    selected_queues: Vec<ash::vk::DeviceQueueCreateInfo>,
    required_family_collection: Vec<Option<(u32, ConcreteQueueFamilyDescriptor)>>,
    enabled_extensions: Vec<Vec<c_char>>,
    enabled_layers: Vec<Vec<c_char>>,
}

pub struct RaytracingInfo {
    shader_group_handle_size: u32,
    max_ray_dispatch_invocation_count: u32,
    max_ray_hit_attribute_size: u32,
    max_ray_recursion_depth: u32,
    max_shader_group_stride: u32,
    shader_group_base_alignment: u32,
    shader_group_handle_alignment: u32,
}

impl RaytracingInfo {
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
    extensions: DeviceExtensions,
    device: ash::Device,
    physical_device: ash::vk::PhysicalDevice,
    ray_tracing_info: Option<RaytracingInfo>,
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
        let alloc_callbacks = self.instance.get_alloc_callbacks();
        unsafe { self.device.destroy_device(alloc_callbacks) }
    }
}

impl Device {
    pub fn native_handle(&self) -> u64 {
        ash::vk::Handle::as_raw(self.device.handle())
    }

    pub(crate) fn ash_ext_swapchain_khr(&self) -> &Option<ash::extensions::khr::Swapchain> {
        &self.extensions.swapchain_khr_ext
    }

    pub(crate) fn ash_ext_raytracing_maintenance1_khr(
        &self,
    ) -> &Option<ash::extensions::khr::RayTracingMaintenance1> {
        &self.extensions.raytracing_maintenance_khr_ext
    }

    pub(crate) fn ash_ext_raytracing_pipeline_khr(
        &self,
    ) -> &Option<ash::extensions::khr::RayTracingPipeline> {
        &self.extensions.raytracing_pipeline_khr_ext
    }

    pub(crate) fn ash_ext_acceleration_structure_khr(
        &self,
    ) -> &Option<ash::extensions::khr::AccelerationStructure> {
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

    pub fn wait_idle(&self) -> VulkanResult<()> {
        match unsafe { self.device.device_wait_idle() } {
            Ok(_) => Ok(()),
            Err(err) => Err(VulkanError::Vulkan(
                err.as_raw(),
                Some(format!("Error waiting for device to be idle: {}", err)),
            )),
        }
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
        surface_extension: Option<&ash::extensions::khr::Surface>,
        queue_family: &ash::vk::QueueFamilyProperties,
        family_index: u32,
        max_queues: u32,
    ) -> Option<u16>
    where
        I: Iterator<Item = &'a QueueFamilySupportedOperationType>,
    {
        if max_queues < queue_family.queue_count {
            return None;
        }

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
        device_layers: &[String],
        debug_name: Option<&str>,
    ) -> VulkanResult<Arc<Self>> {
        // queue cannot be capable of nothing...
        if queue_descriptors.is_empty() {
            return Err(VulkanError::Framework(FrameworkError::UserInput(Some(
                "Error in queue search: no queue descriptor(s) have been specified".to_string(),
            ))));
        }

        unsafe {
            match instance.ash_handle().enumerate_physical_devices() {
                Err(err) => Err(VulkanError::Vulkan(
                    err.as_raw(),
                    Some(format!("Error enumerating physical devices: {}", err)),
                )),
                Ok(physical_devices) => {
                    let mut best_physical_device_score: i128 = -1;
                    let mut selected_physical_device: Option<DeviceData> = None;

                    if physical_devices.is_empty() {
                        return Err(VulkanError::Framework(
                            FrameworkError::NoSuitableDeviceFound,
                        ));
                    }

                    'suitable_device_search: for phy_device in physical_devices.iter() {
                        let enabled_extensions = device_extensions
                            .iter()
                            .map(|ext_name| {
                                let mut ext_bytes = ext_name.to_owned().into_bytes();
                                ext_bytes.push(b"\0"[0]);

                                ext_bytes
                                    .iter()
                                    .map(|b| *b as c_char)
                                    .collect::<Vec<c_char>>()
                            })
                            .collect::<Vec<Vec<c_char>>>();

                        let enabled_layers = device_layers
                            .iter()
                            .map(|ext_name| {
                                let mut ext_bytes = ext_name.to_owned().into_bytes();
                                ext_bytes.push(b"\0"[0]);

                                ext_bytes
                                    .iter()
                                    .map(|b| *b as c_char)
                                    .collect::<Vec<c_char>>()
                            })
                            .collect::<Vec<Vec<c_char>>>();

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

                        match phy_device_extensions {
                            Ok(supported_extensions) => {
                                let supproted_extensions_map = supported_extensions
                                    .iter()
                                    .map(|f| f.extension_name.to_vec())
                                    .collect::<Vec<Vec<i8>>>();

                                let supported_extensions_strings: Vec<&str> =
                                    supproted_extensions_map
                                        .iter()
                                        .map(|ext| match ext.iter().position(|n| *n == 0) {
                                            Some(ext_len) => std::str::from_utf8_unchecked(
                                                std::ptr::slice_from_raw_parts(
                                                    ext.as_ptr() as *const u8,
                                                    ext_len.min(ext.len()),
                                                )
                                                .as_ref()
                                                .unwrap(),
                                            ),
                                            None => todo!(),
                                        })
                                        .collect::<Vec<&str>>();

                                println!(
                                    "Device {} supports the following extensions: ",
                                    phy_device_name
                                );
                                for ext_str in supported_extensions_strings.iter() {
                                    println!("    {}", ext_str);
                                }

                                for requested_extension in device_extensions.iter() {
                                    if !supported_extensions_strings
                                        .iter()
                                        .any(|supported_ext| requested_extension == supported_ext)
                                    {
                                        println!("Requested extension {} is not supported by physical device {}. This device won't be selected.", requested_extension, phy_device_name);
                                        continue 'suitable_device_search;
                                    } else {
                                        println!("Requested extension {} is supported by physical device {}!", requested_extension, phy_device_name);
                                    }
                                }
                            }
                            Err(err) => {
                                println!("Error enumerating device extensions for device {}: {}. Will skip this device.", phy_device_name, err);

                                continue 'suitable_device_search;
                            }
                        }

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

                        let mut available_queue_families: Vec<(
                            usize,
                            &ash::vk::QueueFamilyProperties,
                        )> = queue_family_properties.iter().enumerate().collect();

                        for current_requested_queue_family_descriptor in queue_descriptors.iter() {
                            // this is the currently selected queue family (queue_family, score)
                            let mut selected_queue_family: Option<(usize, u16)> = None;

                            // the following for loop will search for the best fit for requested capabilities
                            /*'suitable_queue_family_search:*/
                            for (family_index, current_descriptor) in
                                available_queue_families.clone()
                            {
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
                                    let queue_create_info =
                                        ash::vk::DeviceQueueCreateInfo::builder()
                                            .queue_family_index(family_index as u32)
                                            .queue_priorities(
                                                current_requested_queue_family_descriptor
                                                    .get_queue_priorities(),
                                            )
                                            .build();

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
                                    continue 'suitable_device_search;
                                }
                            }
                        }

                        let currently_selected_device_data = DeviceData {
                            selected_physical_device: *phy_device,
                            selected_device_features: phy_device_features,
                            selected_queues,
                            required_family_collection,
                            enabled_extensions,
                            enabled_layers,
                        };

                        println!("Found suitable device: {}", phy_device_name);

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

                    match selected_physical_device {
                        Some(selected_device) => {
                            let layers_ptr = selected_device
                                .enabled_layers
                                .iter()
                                .map(|str| str.as_ptr())
                                .collect::<Vec<*const c_char>>();
                            let extensions_ptr = selected_device
                                .enabled_extensions
                                .iter()
                                .map(|str| str.as_ptr())
                                .collect::<Vec<*const c_char>>();

                            let mut device_create_info_builder =
                                ash::vk::DeviceCreateInfo::builder()
                                    .queue_create_infos(selected_device.selected_queues.as_slice())
                                    .enabled_layer_names(layers_ptr.as_slice())
                                    .enabled_extension_names(extensions_ptr.as_slice());

                            let acceleration_structure_enabled =
                                device_extensions.iter().any(|ext| {
                                    ext.as_str()
                                        == ash::extensions::khr::AccelerationStructure::name()
                                            .to_str()
                                            .unwrap_or("")
                                });

                            let ray_tracing_enabled = device_extensions.iter().any(|ext| {
                                ext.as_str()
                                    == ash::extensions::khr::RayTracingPipeline::name()
                                        .to_str()
                                        .unwrap_or("")
                            });

                            let mut features2 = ash::vk::PhysicalDeviceFeatures2::default();
                            let mut accel_structure_features =
                                ash::vk::PhysicalDeviceAccelerationStructureFeaturesKHR::default();
                            let mut ray_tracing_pipeline_features =
                                ash::vk::PhysicalDeviceRayTracingPipelineFeaturesKHR::default();
                            let mut get_device_address_features =
                                ash::vk::PhysicalDeviceBufferDeviceAddressFeatures::default();

                            let mut properties2 = ash::vk::PhysicalDeviceProperties2::default();
                            let mut ray_tracing_pipeline_properties =
                                ash::vk::PhysicalDeviceRayTracingPipelinePropertiesKHR::default();

                            // Enable raytracing if required extensions have been requested
                            if instance.instance_vulkan_version() != InstanceAPIVersion::Version1_0
                            {
                                if acceleration_structure_enabled {
                                    if ray_tracing_enabled {
                                        properties2.p_next = &mut ray_tracing_pipeline_properties as *mut ash::vk::PhysicalDeviceRayTracingPipelinePropertiesKHR as *mut std::ffi::c_void;
                                        accel_structure_features.p_next = &mut ray_tracing_pipeline_features as *mut ash::vk::PhysicalDeviceRayTracingPipelineFeaturesKHR as *mut std::ffi::c_void;

                                        if instance.instance_vulkan_version()
                                            != InstanceAPIVersion::Version1_1
                                        {
                                            ray_tracing_pipeline_features.p_next = &mut get_device_address_features as *mut ash::vk::PhysicalDeviceBufferDeviceAddressFeatures as *mut std::ffi::c_void;
                                        }
                                    }

                                    features2.p_next = &mut accel_structure_features as *mut ash::vk::PhysicalDeviceAccelerationStructureFeaturesKHR as *mut std::ffi::c_void;
                                }

                                if instance.instance_vulkan_version()
                                    == InstanceAPIVersion::Version1_0
                                {
                                    device_create_info_builder = device_create_info_builder
                                        .enabled_features(
                                            &selected_device.selected_device_features,
                                        );
                                } else {
                                    instance.ash_handle().get_physical_device_features2(
                                        selected_device.selected_physical_device,
                                        &mut features2,
                                    );
                                    instance.ash_handle().get_physical_device_properties2(
                                        selected_device.selected_physical_device,
                                        &mut properties2,
                                    );
                                    device_create_info_builder =
                                        device_create_info_builder.push_next(&mut features2);
                                }
                            } else {
                                device_create_info_builder = device_create_info_builder
                                    .enabled_features(&selected_device.selected_device_features);
                            }

                            let mut raytracing_info: Option<RaytracingInfo> = Option::None;

                            let device_create_info = device_create_info_builder.build();
                            return match instance.ash_handle().create_device(
                                selected_device.selected_physical_device.to_owned(),
                                &device_create_info,
                                instance.get_alloc_callbacks(),
                            ) {
                                Ok(device) => {
                                    // open requested swapchain extensions (or implied ones)
                                    let swapchain_ext: Option<ash::extensions::khr::Swapchain> =
                                        match device_extensions.iter().any(|ext| {
                                            ext.as_str()
                                                == ash::extensions::khr::Swapchain::name()
                                                    .to_str()
                                                    .unwrap_or("")
                                        }) {
                                            true => {
                                                Option::Some(ash::extensions::khr::Swapchain::new(
                                                    instance.ash_handle(),
                                                    &device,
                                                ))
                                            }
                                            false => Option::None,
                                        };

                                    let raytracing_pipeline_ext: Option<
                                        ash::extensions::khr::RayTracingPipeline,
                                    > = match ray_tracing_enabled {
                                        true => {
                                            //ray_tracing_pipeline_properties.
                                            println!(
                                                "    RayTracing shader_group_handle_size: {}",
                                                ray_tracing_pipeline_properties
                                                    .shader_group_handle_size
                                            );
                                            println!("    RayTracing max_ray_dispatch_invocation_count: {}", ray_tracing_pipeline_properties.max_ray_dispatch_invocation_count);
                                            println!(
                                                "    RayTracing max_ray_hit_attribute_size: {}",
                                                ray_tracing_pipeline_properties
                                                    .max_ray_hit_attribute_size
                                            );
                                            println!(
                                                "    RayTracing max_ray_recursion_depth: {}",
                                                ray_tracing_pipeline_properties
                                                    .max_ray_recursion_depth
                                            );
                                            println!(
                                                "    RayTracing max_shader_group_stride: {}",
                                                ray_tracing_pipeline_properties
                                                    .max_shader_group_stride
                                            );
                                            println!(
                                                "    RayTracing shader_group_base_alignment: {}",
                                                ray_tracing_pipeline_properties
                                                    .shader_group_base_alignment
                                            );
                                            println!(
                                                "    RayTracing shader_group_handle_alignment: {}",
                                                ray_tracing_pipeline_properties
                                                    .shader_group_handle_alignment
                                            );

                                            raytracing_info = Some(RaytracingInfo {
                                                shader_group_handle_size:
                                                    ray_tracing_pipeline_properties
                                                        .shader_group_handle_size,
                                                max_ray_dispatch_invocation_count:
                                                    ray_tracing_pipeline_properties
                                                        .max_ray_dispatch_invocation_count,
                                                max_ray_hit_attribute_size:
                                                    ray_tracing_pipeline_properties
                                                        .max_ray_hit_attribute_size,
                                                max_ray_recursion_depth:
                                                    ray_tracing_pipeline_properties
                                                        .max_ray_recursion_depth,
                                                max_shader_group_stride:
                                                    ray_tracing_pipeline_properties
                                                        .max_shader_group_stride,
                                                shader_group_base_alignment:
                                                    ray_tracing_pipeline_properties
                                                        .shader_group_base_alignment,
                                                shader_group_handle_alignment:
                                                    ray_tracing_pipeline_properties
                                                        .shader_group_handle_alignment,
                                            });

                                            Option::Some(
                                                ash::extensions::khr::RayTracingPipeline::new(
                                                    instance.ash_handle(),
                                                    &device,
                                                ),
                                            )
                                        }
                                        false => Option::None,
                                    };

                                    let acceleration_structure_ext: Option<
                                        ash::extensions::khr::AccelerationStructure,
                                    > = match acceleration_structure_enabled {
                                        true => Option::Some(
                                            ash::extensions::khr::AccelerationStructure::new(
                                                instance.ash_handle(),
                                                &device,
                                            ),
                                        ),
                                        false => Option::None,
                                    };

                                    let raytracing_maintenance_ext: Option<
                                        ash::extensions::khr::RayTracingMaintenance1,
                                    > = match device_extensions.iter().any(|ext| {
                                        ext.as_str()
                                            == ash::extensions::khr::RayTracingMaintenance1::name()
                                                .to_str()
                                                .unwrap_or("")
                                    }) {
                                        true => Option::Some(
                                            ash::extensions::khr::RayTracingMaintenance1::new(
                                                instance.ash_handle(),
                                                &device,
                                            ),
                                        ),
                                        false => Option::None,
                                    };

                                    let mut obj_name_bytes = vec![];

                                    if let Some(ext) = instance.get_debug_ext_extension() {
                                        if let Some(name) = debug_name {
                                            for name_ch in name.as_bytes().iter() {
                                                obj_name_bytes.push(*name_ch);
                                            }
                                            obj_name_bytes.push(0x00);

                                            let object_name =
                                                std::ffi::CStr::from_bytes_with_nul_unchecked(
                                                    obj_name_bytes.as_slice(),
                                                );
                                            // set device name for debugging
                                            let dbg_info =
                                                ash::vk::DebugUtilsObjectNameInfoEXT::builder()
                                                    .object_type(ash::vk::ObjectType::DEVICE)
                                                    .object_handle(ash::vk::Handle::as_raw(
                                                        device.handle(),
                                                    ))
                                                    .object_name(object_name)
                                                    .build();

                                            if let Err(err) = ext.set_debug_utils_object_name(
                                                device.handle(),
                                                &dbg_info,
                                            ) {
                                                #[cfg(debug_assertions)]
                                                {
                                                    println!("Error setting the Debug name for the newly created Device, will use handle. Error: {}", err)
                                                }
                                            }
                                        }
                                    }

                                    Ok(Arc::new(Self {
                                        //_name_bytes: obj_name_bytes,
                                        required_family_collection: Mutex::new(
                                            selected_device.required_family_collection,
                                        ),
                                        device,
                                        extensions: DeviceExtensions {
                                            swapchain_khr_ext: swapchain_ext,
                                            raytracing_pipeline_khr_ext: raytracing_pipeline_ext,
                                            raytracing_maintenance_khr_ext:
                                                raytracing_maintenance_ext,
                                            acceleration_structure_khr_ext:
                                                acceleration_structure_ext,
                                        },
                                        instance,
                                        physical_device: selected_device.selected_physical_device,
                                        ray_tracing_info: raytracing_info,
                                    }))
                                }
                                Err(err) => Err(VulkanError::Vulkan(
                                    err.as_raw(),
                                    Some(format!("Error creating the logical device: {}", err)),
                                )),
                            };
                        }
                        None => Err(VulkanError::Framework(
                            FrameworkError::NoSuitableDeviceFound,
                        )),
                    }
                }
            }
        }
    }

    pub(crate) fn move_out_queue_family(
        &self,
        index: usize,
    ) -> VulkanResult<(u32, ConcreteQueueFamilyDescriptor)> {
        match self.required_family_collection.lock() {
            Ok(mut collection) => match collection.len() > index {
                true => match collection[index].to_owned() {
                    Some(cose) => {
                        collection[index] = None;
                        Ok(cose)
                    }
                    None => Err(VulkanError::Framework(
                            FrameworkError::UserInput(Some(format!("The queue family with index {} has already been created once and there can only be one QueueFamily for requested queue capabilies.", index))),
                        ))
                },
                false =>
                    Err(VulkanError::Framework(
                        FrameworkError::UserInput(Some(format!("A queue family with index {} does not exists, at device creation time only {} queue families were requested.", index, collection.len()))),
                    ))
            },
            Err(err) =>
                Err(VulkanError::Framework(
                    FrameworkError::Unknown(Some(format!("Error acquiring internal mutex: {}", err))),
                ))
        }
    }

    pub(crate) fn search_adequate_heap(
        &self,
        current_requested_memory_heap_descriptor: &ConcreteMemoryHeapDescriptor,
    ) -> Option<(u32, u32, u32)> {
        let device_memory_properties = unsafe {
            self.get_parent_instance()
                .ash_handle()
                .get_physical_device_memory_properties(self.physical_device.to_owned())
        };

        let requested_size = current_requested_memory_heap_descriptor.memory_minimum_size();

        'suitable_heap_search: for (memory_type_index, heap_descriptor) in
            device_memory_properties.memory_types.iter().enumerate()
        {
            // discard heaps that are too small
            let available_size =
                device_memory_properties.memory_heaps[heap_descriptor.heap_index as usize].size;
            if requested_size > available_size {
                continue 'suitable_heap_search;
            }

            match current_requested_memory_heap_descriptor.memory_type() {
                MemoryType::DeviceLocal(host_visibility) => {
                    // if I want device-local memory just ignore heaps that are not device-local
                    if !heap_descriptor
                        .property_flags
                        .contains(ash::vk::MemoryPropertyFlags::DEVICE_LOCAL)
                    {
                        continue 'suitable_heap_search;
                    }

                    match host_visibility {
                        Some(visibility_model) => {
                            // a visibility model is specified, exclude heaps that are not suitable as not memory-mappable
                            if !heap_descriptor
                                .property_flags
                                .contains(ash::vk::MemoryPropertyFlags::HOST_VISIBLE)
                            {
                                continue 'suitable_heap_search;
                            }

                            match visibility_model.cached() {
                                true => {
                                    if !heap_descriptor
                                        .property_flags
                                        .contains(ash::vk::MemoryPropertyFlags::HOST_CACHED)
                                    {
                                        continue 'suitable_heap_search;
                                    }
                                }
                                false => {
                                    if heap_descriptor
                                        .property_flags
                                        .contains(ash::vk::MemoryPropertyFlags::HOST_CACHED)
                                    {
                                        continue 'suitable_heap_search;
                                    }
                                }
                            }
                        }
                        None => {
                            // a visibility model is NOT specified, the user wants memory that is not memory-mappable
                            if heap_descriptor
                                .property_flags
                                .contains(ash::vk::MemoryPropertyFlags::HOST_VISIBLE)
                            {
                                continue 'suitable_heap_search;
                            }

                            // Only avaialble on Vulkan 1.1
                            /*
                            if (instance.instance_vulkan_version()
                                != InstanceAPIVersion::Version1_0)
                                && (!heap_descriptor.property_flags.contains(
                                    ash::vk::MemoryPropertyFlags::PROTECTED,
                                ))
                            {
                                continue 'suitable_heap_search;
                            }
                            */
                        }
                    }
                }
                MemoryType::HostLocal(_coherence_model) => {
                    // if I want host-local memory just ignore heaps that are not host-local
                    if heap_descriptor
                        .property_flags
                        .contains(ash::vk::MemoryPropertyFlags::DEVICE_LOCAL)
                    {
                        continue 'suitable_heap_search;
                    }
                }
            }

            // If I'm here the previous search has find that the current heap is suitable...
            return Some((
                heap_descriptor.heap_index,
                memory_type_index as u32,
                heap_descriptor.property_flags.as_raw(),
            ));
        }
        None
    }
}

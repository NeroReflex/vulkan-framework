use crate::instance;
use crate::instance::InstanceOwned;
use crate::queue_family;

use ash;
use ash::extensions;
use ash::vk::Handle;

use crate::result::VkError;

use std::os::raw::c_char;
use std::vec::Vec;

use std::rc::{Rc, Weak};

pub(crate) trait DeviceOwned {
    fn get_parent_device(&self) -> Weak<crate::device::Device>;
}

struct DeviceData {
    selected_physical_device: ash::vk::PhysicalDevice,
    selected_device_features: ash::vk::PhysicalDeviceFeatures,
    selected_queues: Vec<ash::vk::DeviceQueueCreateInfo>,
    required_family_collection: Vec<(u32, queue_family::ConcreteQueueFamilyDescriptor)>,
    enabled_extensions: Vec<Vec<c_char>>,
    enabled_layers: Vec<Vec<c_char>>,
    validation_layers: bool,
}

pub struct Device {
    data: Box<DeviceData>,
    instance: Weak<crate::instance::Instance>,
    device: ash::Device,
}

impl InstanceOwned for Device {
    fn get_parent_instance(&self) -> Weak<crate::instance::Instance> {
        self.instance.clone()
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        match self.get_parent_instance().upgrade() {
            Some(instance) => {
                let alloc_callbacks = instance.get_alloc_callbacks();
                unsafe { self.device.destroy_device(alloc_callbacks) }
            }
            None => {
                assert!(true == false)
            }
        }
    }
}

impl Device {
    pub fn native_handle(&self) -> &ash::Device {
        &self.device
    }

    pub fn are_validation_layers_enabled(&self) -> bool {
        self.data.validation_layers
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
    fn corresponds(
        operations: &[queue_family::QueueFamilySupportedOperationType],
        instance: &ash::Instance,
        device: &ash::vk::PhysicalDevice,
        surface_extension: Option<&ash::extensions::khr::Surface>,
        queue_family: &ash::vk::QueueFamilyProperties,
        family_index: u32,
        max_queues: u32,
    ) -> Option<u16> {
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

            /*match queue_family.queue_flags.contains(ash::vk::QueueFlags::) {
                true => score += 1,
                false => score += 0
            };*/

            match feature {
                queue_family::QueueFamilySupportedOperationType::Transfer => {
                    if !queue_family
                        .queue_flags
                        .contains(ash::vk::QueueFlags::TRANSFER)
                    {
                        return None;
                    }

                    score -= 1;
                }
                queue_family::QueueFamilySupportedOperationType::Present(surface) => {
                    unsafe {
                        match surface_extension {
                            Some(ext) => {
                                match ext.get_physical_device_surface_support(
                                    device.to_owned(),
                                    family_index,
                                    *surface,
                                ) {
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
                                return None;
                            }
                        }
                    }

                    score -= 1;
                }
                queue_family::QueueFamilySupportedOperationType::Graphics => {
                    if !queue_family
                        .queue_flags
                        .contains(ash::vk::QueueFlags::GRAPHICS)
                    {
                        return None;
                    }

                    score -= 1;
                }
                queue_family::QueueFamilySupportedOperationType::Compute => {
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

    pub fn new(
        instance_weak_ptr: Weak<crate::instance::Instance>,
        queue_descriptors: &[queue_family::ConcreteQueueFamilyDescriptor],
        device_extensions: &[String],
        device_layers: &[String],
    ) -> Result<Rc<Device>, VkError> {
        // queue cannot be capable of nothing...
        if queue_descriptors.is_empty() {
            return Err(VkError {});
        }

        match instance_weak_ptr.upgrade() {
            Some(instance) => {
                unsafe {
                    match instance.native_handle().enumerate_physical_devices() {
                        Err(_err) => {
                            return Err(VkError {});
                        }
                        Ok(physical_devices) => {
                            let mut best_physical_device_score: i128 = -1;
                            let mut selected_physical_device: Option<DeviceData> = None;

                            if physical_devices.is_empty() {
                                return Err(VkError {});
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

                                let mut enabled_layers = device_layers
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

                                /*
                                Previous implementations of Vulkan made a distinction between instance and device specific validation layers, but this is no longer the case.
                                That means that the enabledLayerCount and ppEnabledLayerNames fields of VkDeviceCreateInfo are ignored by up-to-date implementations.
                                However, it is still a good idea to set them anyway to be compatible with older implementations:
                                */
                                if instance.is_debugging_enabled() {
                                    enabled_layers.push(
                                        b"VK_LAYER_KHRONOS_validation\0"
                                            .iter()
                                            .map(|c| *c as c_char)
                                            .collect::<Vec<c_char>>(),
                                    );
                                }

                                let phy_device_features = instance
                                    .native_handle()
                                    .get_physical_device_features(phy_device.to_owned());
                                let phy_device_properties = instance
                                    .native_handle()
                                    .get_physical_device_properties(phy_device.to_owned());

                                let msbytes = match phy_device_properties.device_type.as_raw() {
                                    DISCRETE_GPU => 0xC000000000000000u64,
                                    INTEGRATED_GPU => 0x8000000000000000u64,
                                    VK_PHYSICAL_DEVICE_TYPE_CPU => 0x4000000000000000u64,
                                    _ => 0x0000000000000000u64,
                                };

                                let current_score = msbytes | 0 as u64;

                                // get queues properties (used to check if all requested queues are available on this device)
                                let queue_family_properties = instance
                                    .native_handle()
                                    .get_physical_device_queue_family_properties(
                                        phy_device.to_owned(),
                                    );

                                // Check if all requested queues are supported
                                let mut selected_queues: Vec<ash::vk::DeviceQueueCreateInfo> =
                                    vec![];
                                let mut required_family_collection: Vec<(
                                    u32,
                                    queue_family::ConcreteQueueFamilyDescriptor,
                                )> = vec![];

                                let mut available_queue_families: Vec<(
                                    usize,
                                    &ash::vk::QueueFamilyProperties,
                                )> = queue_family_properties.iter().enumerate().collect();

                                for current_requested_queue_family_descriptor in
                                    queue_descriptors.iter()
                                {
                                    // this is the currently selected queue family (queue_family, score)
                                    let mut selected_queue_family: Option<(usize, u16)> = None;

                                    // the following for loop will search for the best fit for requested capabilities
                                    'suitable_queue_family_search: for (
                                        family_index,
                                        current_descriptor,
                                    ) in
                                        available_queue_families.clone()
                                    {
                                        match Self::corresponds(
                                            current_requested_queue_family_descriptor
                                                .get_supported_operations(),
                                            instance.native_handle(),
                                            phy_device,
                                            instance.get_surface_khr_extension(),
                                            current_descriptor,
                                            family_index as u32,
                                            current_requested_queue_family_descriptor.max_queues()
                                                as u32,
                                        ) {
                                            Some(score) => {
                                                // Found a suitable queue family.
                                                // Use this queue family if it's a better fit than the previous one
                                                match selected_queue_family {
                                                    Some((_, best_fit_queue_score)) => {
                                                        if best_fit_queue_score > score {
                                                            selected_queue_family =
                                                                Some((family_index, score))
                                                        }
                                                    }
                                                    None => {
                                                        selected_queue_family =
                                                            Some((family_index, score))
                                                    }
                                                }

                                                // Stop the search. This changes the algorithm from a "best fit" to a "first fit"
                                                //break 'suitable_queue_family_search;
                                            }
                                            None => {}
                                        }
                                    }

                                    // if any of the queue is not supported continue the search for a suitable device
                                    // otherwise remove the current best fit from the queue of available queue_families to avoid choosing it two times
                                    match selected_queue_family {
                                        Some((family_index, _)) => {
                                            let mut queue_create_info =
                                                ash::vk::DeviceQueueCreateInfo::default();
                                            queue_create_info.queue_family_index =
                                                family_index as u32;
                                            queue_create_info.queue_count =
                                                current_requested_queue_family_descriptor
                                                    .max_queues();
                                            queue_create_info.p_queue_priorities =
                                                current_requested_queue_family_descriptor
                                                    .get_queue_priorities()
                                                    .as_ptr();

                                            selected_queues.push(queue_create_info);
                                            required_family_collection.push((
                                                family_index as u32,
                                                current_requested_queue_family_descriptor.clone(),
                                            ));

                                            available_queue_families = available_queue_families.iter().map(|(queue_family_index, queue_family_properties)| -> Option<(usize, &ash::vk::QueueFamilyProperties)> {
                                                if *queue_family_index != family_index {
                                                    return Some((*queue_family_index, queue_family_properties))
                                                }

                                                None
                                            }).into_iter().flatten().collect();
                                        }
                                        None => {
                                            continue 'suitable_device_search;
                                        }
                                    }
                                }

                                /*
                                if (deviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU) { // Exclude inadequate devices
                                    currentScore = 0;
                                }
                                */

                                let currently_selected_device_data = DeviceData {
                                    selected_physical_device: phy_device.clone(),
                                    selected_device_features: phy_device_features,
                                    selected_queues: selected_queues,
                                    required_family_collection: required_family_collection,
                                    enabled_extensions: enabled_extensions,
                                    enabled_layers: enabled_layers,
                                    validation_layers: instance.is_debugging_enabled(),
                                };

                                match selected_physical_device {
                                    Some(currently_selected_device) => {
                                        if best_physical_device_score < current_score as i128 {
                                            best_physical_device_score = current_score as i128;
                                            selected_physical_device =
                                                Some(currently_selected_device_data);
                                        } else {
                                            selected_physical_device =
                                                Some(currently_selected_device);
                                        }
                                    }
                                    None => {
                                        best_physical_device_score = current_score as i128;
                                        selected_physical_device =
                                            Some(currently_selected_device_data);
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

                                    let mut device_create_info =
                                        ash::vk::DeviceCreateInfo::default();

                                    device_create_info.queue_create_info_count =
                                        selected_device.selected_queues.len() as u32;
                                    device_create_info.p_queue_create_infos =
                                        selected_device.selected_queues.as_slice().as_ptr();

                                    device_create_info.p_enabled_features =
                                        &selected_device.selected_device_features;

                                    device_create_info.enabled_layer_count =
                                        layers_ptr.len() as u32;
                                    device_create_info.pp_enabled_layer_names =
                                        layers_ptr.as_slice().as_ptr();

                                    device_create_info.enabled_extension_count =
                                        extensions_ptr.len() as u32;
                                    device_create_info.pp_enabled_extension_names =
                                        extensions_ptr.as_slice().as_ptr();

                                    return match instance.native_handle().create_device(
                                        selected_device.selected_physical_device.to_owned(),
                                        &device_create_info,
                                        instance.get_alloc_callbacks(),
                                    ) {
                                        Ok(device) => Ok(Rc::new(Self {
                                            data: Box::new(selected_device),
                                            device: device,
                                            instance: instance_weak_ptr,
                                        })),
                                        Err(_err) => Err(VkError {}),
                                    };
                                }
                                None => {
                                    return Err(VkError {});
                                }
                            }
                        }
                    }
                }
            }
            None => {
                assert!(true == false);

                return Err(VkError {});
            }
        }
    }
}

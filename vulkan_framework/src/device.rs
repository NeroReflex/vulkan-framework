use crate::memory_heap::{ConcreteMemoryHeapDescriptor, MemoryType};
use crate::prelude::{VulkanError, VulkanResult};
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
}

struct DeviceData {
    selected_physical_device: ash::vk::PhysicalDevice,
    selected_device_features: ash::vk::PhysicalDeviceFeatures,
    selected_queues: Vec<ash::vk::DeviceQueueCreateInfo>,
    required_family_collection: Vec<Option<(u32, ConcreteQueueFamilyDescriptor)>>,
    enabled_extensions: Vec<Vec<c_char>>,
    enabled_layers: Vec<Vec<c_char>>,
}

pub struct Device {
    //_name_bytes: Vec<u8>,
    required_family_collection: Mutex<Vec<Option<(u32, ConcreteQueueFamilyDescriptor)>>>,
    instance: Arc<Instance>,
    extensions: DeviceExtensions,
    device: ash::Device,
    physical_device: ash::vk::PhysicalDevice,
}

impl PartialEq for Device {
    fn eq(&self, other: &Self) -> bool {
        ash::vk::Handle::as_raw(self.device.handle())
            == ash::vk::Handle::as_raw(other.device.handle())
    }

    fn ne(&self, other: &Self) -> bool {
        !self.eq(other)
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

    pub fn ash_handle(&self) -> &ash::Device {
        &self.device
    }

    pub fn wait_idle(&self) -> VulkanResult<()> {
        match unsafe { self.device.device_wait_idle() } {
            Ok(_) => Ok(()),
            Err(err) => {
                #[cfg(debug_assertions)]
                {
                    panic!("Error waiting for device to be idle: {}", err)
                }

                Err(VulkanError::Unspecified)
            }
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
                                panic!("SurfaceKHR extension not available, have you forgotten to specify it on instance creation?")
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
            return Err(VulkanError::Unspecified);
        }

        unsafe {
            match instance.ash_handle().enumerate_physical_devices() {
                Err(_err) => Err(VulkanError::Unspecified),
                Ok(physical_devices) => {
                    let mut best_physical_device_score: i128 = -1;
                    let mut selected_physical_device: Option<DeviceData> = None;

                    if physical_devices.is_empty() {
                        return Err(VulkanError::Unspecified);
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
                                match Self::corresponds(
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
                                                selected_queue_family = Some((family_index, score))
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
                                    queue_create_info.queue_family_index = family_index as u32;
                                    queue_create_info.queue_count =
                                        current_requested_queue_family_descriptor.max_queues()
                                            as u32;
                                    queue_create_info.p_queue_priorities =
                                        current_requested_queue_family_descriptor
                                            .get_queue_priorities()
                                            .as_ptr();

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

                            let mut device_create_info = ash::vk::DeviceCreateInfo::default();

                            device_create_info.queue_create_info_count =
                                selected_device.selected_queues.len() as u32;
                            device_create_info.p_queue_create_infos =
                                selected_device.selected_queues.as_slice().as_ptr();

                            device_create_info.p_enabled_features =
                                &selected_device.selected_device_features;

                            device_create_info.enabled_layer_count = layers_ptr.len() as u32;
                            device_create_info.pp_enabled_layer_names =
                                layers_ptr.as_slice().as_ptr();

                            device_create_info.enabled_extension_count =
                                extensions_ptr.len() as u32;
                            device_create_info.pp_enabled_extension_names =
                                extensions_ptr.as_slice().as_ptr();

                            return match instance.ash_handle().create_device(
                                selected_device.selected_physical_device.to_owned(),
                                &device_create_info,
                                instance.get_alloc_callbacks(),
                            ) {
                                Ok(device) => {
                                    // open requested swapchain extensions (or implied ones)
                                    let swapchain_ext: Option<ash::extensions::khr::Swapchain> =
                                        match device_extensions.iter().any(|ext| {
                                            *ext == String::from(
                                                ash::extensions::khr::Swapchain::name()
                                                    .to_str()
                                                    .unwrap_or(""),
                                            )
                                        }) {
                                            true => {
                                                Option::Some(ash::extensions::khr::Swapchain::new(
                                                    instance.ash_handle(),
                                                    &device,
                                                ))
                                            }
                                            false => Option::None,
                                        };

                                    let mut obj_name_bytes = vec![];

                                    match instance.get_debug_ext_extension() {
                                        Some(ext) => {
                                            match debug_name {
                                                Some(name) => {
                                                    for name_ch in name.as_bytes().iter() {
                                                        obj_name_bytes.push(*name_ch);
                                                    }
                                                    obj_name_bytes.push(0x00);

                                                    let object_name = std::ffi::CStr::from_bytes_with_nul_unchecked(obj_name_bytes.as_slice());
                                                    // set device name for debugging
                                                    let dbg_info = ash::vk::DebugUtilsObjectNameInfoEXT::builder()
                                                        .object_type(ash::vk::ObjectType::DEVICE)
                                                        .object_handle(ash::vk::Handle::as_raw(device.handle()))
                                                        .object_name(object_name)
                                                        .build();

                                                    match ext.set_debug_utils_object_name(
                                                        device.handle(),
                                                        &dbg_info,
                                                    ) {
                                                        Ok(_) => {
                                                            #[cfg(debug_assertions)]
                                                            {
                                                                println!("Device Debug object name changed");
                                                            }
                                                        }
                                                        Err(err) => {
                                                            #[cfg(debug_assertions)]
                                                            {
                                                                panic!("Error setting the Debug name for the newly created Device, will use handle. Error: {}", err)
                                                            }
                                                        }
                                                    }
                                                }
                                                None => {}
                                            };
                                        }
                                        None => {}
                                    }

                                    Ok(Arc::new(Self {
                                        //_name_bytes: obj_name_bytes,
                                        required_family_collection: Mutex::new(
                                            selected_device.required_family_collection,
                                        ),
                                        device,
                                        extensions: DeviceExtensions {
                                            swapchain_khr_ext: swapchain_ext,
                                        },
                                        instance,
                                        physical_device: selected_device.selected_physical_device,
                                    }))
                                }
                                Err(_err) => Err(VulkanError::Unspecified),
                            };
                        }
                        None => Err(VulkanError::Unspecified),
                    }
                }
            }
        }
    }

    pub(crate) fn move_out_queue_family(
        &self,
        index: usize,
    ) -> Option<(u32, ConcreteQueueFamilyDescriptor)> {
        match self.required_family_collection.lock() {
            Ok(mut collection) => match collection.len() > index {
                true => match collection[index].to_owned() {
                    Some(cose) => {
                        collection[index] = None;
                        Some(cose)
                    }
                    None => {
                        #[cfg(debug_assertions)]
                        {
                            panic!("The queue family with index {} has already been created once and there can only be one QueueFamily for requested queue capabilies.", index)
                        }

                        Option::None
                    }
                },
                false => {
                    #[cfg(debug_assertions)]
                    {
                        panic!("A queue family with index {} does not exists, at device creation time only {} queue families were requested.", index, collection.len())
                    }

                    Option::None
                }
            },
            Err(err) => {
                #[cfg(debug_assertions)]
                {
                    panic!("Error acquiring internal mutex: {}", err)
                }

                Option::None
            }
        }
    }

    pub(crate) fn search_adequate_heap(
        &self,
        current_requested_memory_heap_descriptor: &ConcreteMemoryHeapDescriptor,
    ) -> Option<(u32, u32)> {
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
                        Some(_visibility_model) => {
                            // a visibility model is specified, exclude heaps that are not suitable as not memory-mappable
                            if !heap_descriptor
                                .property_flags
                                .contains(ash::vk::MemoryPropertyFlags::HOST_VISIBLE)
                            {
                                continue 'suitable_heap_search;
                            }
                        }
                        None => {
                            // a visibility model is NOT specified, the user wants memory that is not memory-mappable, so protected memory <3
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
            return Some((heap_descriptor.heap_index, memory_type_index as u32));
        }
        None
    }
}

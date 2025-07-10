use crate::prelude::{FrameworkError, VulkanError, VulkanResult};

use std::os::raw::c_char;
use std::string::String;
use std::vec::Vec;

use std::sync::Arc;

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum InstanceAPIVersion {
    Version1_0,
    Version1_1,
    Version1_2,
    Version1_3,
}

pub trait InstanceOwned {
    fn get_parent_instance(&self) -> Arc<Instance>;
}

pub(crate) struct InstanceData {
    application_name: Vec<c_char>,
    engine_name: Vec<c_char>,
    enabled_layers: Vec<Vec<c_char>>,
    enabled_extensions: Vec<Vec<c_char>>,
}

struct InstanceExtensions {
    surface_khr_ext: Option<ash::extensions::khr::Surface>,
    debug_ext_ext: Option<ash::extensions::ext::DebugUtils>,
}

pub struct Instance {
    //alloc_callbacks: Option<ash::vk::AllocationCallbacks>,
    entry: ash::Entry,
    instance: ash::Instance,
    extensions: InstanceExtensions,
    version: InstanceAPIVersion,
}

impl Drop for Instance {
    fn drop(&mut self) {
        let alloc_callbacks = self.get_alloc_callbacks();

        unsafe { self.instance.destroy_instance(alloc_callbacks) }
    }
}

impl Instance {
    #[inline]
    pub fn instance_vulkan_version(&self) -> InstanceAPIVersion {
        self.version
    }

    #[inline]
    pub(crate) fn get_debug_ext_extension(&self) -> Option<&ash::extensions::ext::DebugUtils> {
        match self.extensions.debug_ext_ext.as_ref() {
            Some(debug_ext_ext) => Some(debug_ext_ext),
            None => None,
        }
    }

    #[inline]
    pub(crate) fn get_surface_khr_extension(&self) -> Option<&ash::extensions::khr::Surface> {
        match self.extensions.surface_khr_ext.as_ref() {
            Some(ext) => Some(ext),
            None => None,
        }
    }

    #[inline]
    pub fn get_alloc_callbacks(&self) -> Option<&ash::vk::AllocationCallbacks> {
        // TODO: implement in such a way that Instance remains Send + Sync

        /*match self.data.alloc_callbacks.as_ref() {
            Some(callbacks) => Some(callbacks),
            None => None,
        }*/

        None
    }

    #[inline]
    pub fn native_handle(&self) -> u64 {
        ash::vk::Handle::as_raw(self.instance.handle())
    }

    #[inline]
    pub(crate) fn ash_handle(&self) -> &ash::Instance {
        &self.instance
    }

    /**
     * Creates a new vulkan instance with required instance layers and extensions.
     *
     * For each requested extension that is also supported by the framework an handle will be created and the user
     * should be using it in case low-level operations are to be performed with native handles.
     *
     * *Hint*: for development build you should specify the "VK_EXT_debug_utils" instance extensions alongside with "VK_LAYER_KHRONOS_validation":
     * it will help you a lot to have validation layers giving you names of erroring stuff instead of raw handles or no errors and stuff going nut at runtime.
     *
     * Supported extensions are:
     *   - VK_EXT_debug_utils
     *   - VK_KHR_surface
     *
     */
    pub fn new(
        instance_layers: &[String],
        instance_extensions: &[String],
        engine_name: &String,
        app_name: &String,
        api_version: &InstanceAPIVersion,
    ) -> VulkanResult<Arc<Self>> {
        // TODO: allow the user to provide its own in a way that Instance remains Send + Sync
        let alloc_callbacks: Option<ash::vk::AllocationCallbacks> = None;

        let mut app_bytes = app_name.to_owned().into_bytes();
        app_bytes.push(b"\0"[0]);

        let mut engine_bytes = engine_name.to_owned().into_bytes();
        engine_bytes.push(b"\0"[0]);

        let data = Box::<InstanceData>::new(InstanceData {
            application_name: app_bytes
                .iter()
                .map(|b| *b as c_char)
                .collect::<Vec<c_char>>(),
            engine_name: engine_bytes
                .iter()
                .map(|b| *b as c_char)
                .collect::<Vec<c_char>>(),
            enabled_layers: instance_layers
                .iter()
                .map(|ext_name| {
                    let mut ext_bytes = ext_name.to_owned().into_bytes();
                    ext_bytes.push(b"\0"[0]);

                    ext_bytes
                        .iter()
                        .map(|b| *b as c_char)
                        .collect::<Vec<c_char>>()
                })
                .collect(),
            enabled_extensions: instance_extensions
                .iter()
                .map(|ext_name| {
                    let mut ext_bytes = ext_name.to_owned().into_bytes();
                    ext_bytes.push(b"\0"[0]);

                    ext_bytes
                        .iter()
                        .map(|b| *b as c_char)
                        .collect::<Vec<c_char>>()
                })
                .collect(),
        });

        let layers_ptr = data
            .enabled_layers
            .iter()
            .map(|str| str.as_ptr())
            .collect::<Vec<*const c_char>>();
        let extensions_ptr = data
            .enabled_extensions
            .iter()
            .map(|str| str.as_ptr())
            .collect::<Vec<*const c_char>>();

        unsafe {
            let app_info = ash::vk::ApplicationInfo {
                s_type: ash::vk::StructureType::APPLICATION_INFO,
                p_next: std::ptr::null(),
                p_application_name: data.as_ref().application_name.as_ptr(),
                application_version: 0,
                p_engine_name: data.as_ref().engine_name.as_ptr(),
                engine_version: 0,
                api_version: match api_version {
                    InstanceAPIVersion::Version1_0 => ash::vk::make_api_version(0, 1, 0, 0),
                    InstanceAPIVersion::Version1_1 => ash::vk::make_api_version(0, 1, 1, 0),
                    InstanceAPIVersion::Version1_2 => ash::vk::make_api_version(0, 1, 2, 0),
                    InstanceAPIVersion::Version1_3 => ash::vk::make_api_version(0, 1, 3, 0),
                },
            };

            let create_info = ash::vk::InstanceCreateInfo {
                s_type: ash::vk::StructureType::INSTANCE_CREATE_INFO,
                p_next: std::ptr::null(),
                flags: ash::vk::InstanceCreateFlags::default(),
                p_application_info: &app_info as *const ash::vk::ApplicationInfo,
                pp_enabled_layer_names: layers_ptr.as_ptr(),
                enabled_layer_count: layers_ptr.len() as u32,
                pp_enabled_extension_names: extensions_ptr.as_ptr(),
                enabled_extension_count: extensions_ptr.len() as u32,
            };

            if let Ok(entry) = ash::Entry::load() {
                let Ok(instance) = entry.create_instance(
                    &create_info,
                    match alloc_callbacks.as_ref() {
                        Some(callbacks) => Some(callbacks),
                        None => None,
                    },
                ) else {
                    return Err(VulkanError::Framework(
                        FrameworkError::CannotCreateVulkanInstance,
                    ));
                };

                // also enable debugging extension for debug build
                let debug_ext = match instance_extensions.iter().any(|ext| {
                    ext.as_str()
                        == ash::extensions::ext::DebugUtils::name()
                            .to_str()
                            .unwrap_or("")
                }) {
                    true => Option::Some(ash::extensions::ext::DebugUtils::new(&entry, &instance)),
                    false => Option::None,
                };

                // if requested enable the swapchain required extension(s)
                let surface_ext = match instance_extensions.iter().any(|ext| {
                    ext.as_str() == ash::extensions::khr::Surface::name().to_str().unwrap_or("")
                }) {
                    true => Option::Some(ash::extensions::khr::Surface::new(&entry, &instance)),
                    false => Option::None,
                };

                return Ok(Arc::new(Self {
                    //data: data,
                    //alloc_callbacks,
                    entry,
                    instance,
                    extensions: InstanceExtensions {
                        surface_khr_ext: surface_ext,
                        debug_ext_ext: debug_ext,
                    },
                    version: *api_version,
                }));
            }

            Err(VulkanError::Framework(FrameworkError::CannotLoadVulkan))
        }
    }
}

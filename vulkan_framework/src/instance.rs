use ash::prelude::VkResult;

use crate::result::VkError;

use std::os::raw::c_char;
use std::rc::{Rc, Weak};
use std::string::String;
use std::vec::Vec;

pub enum InstanceAPIVersion {
    Version1_0,
    Version1_1,
    Version1_2,
    Version1_3,
}

pub(crate) trait InstanceOwned {
    fn get_parent_instance(&self) -> Weak<Instance>;
}

pub(crate) struct InstanceData {
    application_name: Vec<c_char>,
    engine_name: Vec<c_char>,
    enabled_layers: Vec<Vec<c_char>>,
    enabled_extensions: Vec<Vec<c_char>>,
    validation_layers: bool,
    alloc_callbacks: Option<ash::vk::AllocationCallbacks>,
}

pub struct Instance {
    data: Box<InstanceData>,
    entry: ash::Entry,
    instance: ash::Instance,
}

impl Drop for Instance {
    fn drop(&mut self) {
        let alloc_callbacks = self.get_alloc_callbacks();
        //println!("> Dropping {}", self.name);
        unsafe {
            self.instance.destroy_instance(alloc_callbacks);
        }
    }
}

impl Instance {
    pub fn get_alloc_callbacks(&self) -> Option<&ash::vk::AllocationCallbacks> {
        match self.data.alloc_callbacks.as_ref() {
            Some(callbacks) => Some(callbacks),
            None => None,
        }
    }

    pub fn native_handle(&self) -> &ash::Instance {
        &self.instance
    }

    pub fn are_validation_layers_enabled(&self) -> bool {
        self.data.validation_layers
    }

    pub fn new(
        instance_extensions: &[String],
        engine_name: &String,
        app_name: &String,
        api_version: &InstanceAPIVersion,
        enable_validation_layers: bool,
    ) -> Result<Rc<Instance>, VkError> {
        let mut app_bytes = app_name.to_owned().into_bytes();
        app_bytes.push(b"\0"[0]);

        let mut engine_bytes = engine_name.to_owned().into_bytes();
        engine_bytes.push(b"\0"[0]);

        let validation_layer_name_bytes = b"VK_LAYER_KHRONOS_validation\0"
            .iter()
            .map(|c| *c as c_char)
            .collect::<Vec<c_char>>();

        let data = Box::<InstanceData>::new(InstanceData {
            application_name: app_bytes
                .iter()
                .map(|b| *b as c_char)
                .collect::<Vec<c_char>>(),
            engine_name: engine_bytes
                .iter()
                .map(|b| *b as c_char)
                .collect::<Vec<c_char>>(),
            enabled_layers: match enable_validation_layers {
                true => vec![validation_layer_name_bytes
                    .iter()
                    .map(|b| *b as c_char)
                    .collect::<Vec<c_char>>()],
                false => vec![],
            },
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
            validation_layers: enable_validation_layers,
            alloc_callbacks: None,
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
                if let Ok(instance) = entry.create_instance(
                    &create_info,
                    match data.alloc_callbacks.as_ref() {
                        Some(callbacks) => Some(callbacks),
                        None => None,
                    },
                ) {
                    return Ok(Rc::new(Instance {
                        data: data,
                        entry: entry,
                        instance: instance,
                    }));
                }

                return Err(VkError {});
            }

            return Err(VkError {});
        }
    }
}

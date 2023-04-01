use crate::result::VkError;

use std::os::raw::c_char;
use std::vec::Vec;
use std::string::String;

pub enum InstanceAPIVersion {
    Version1_0,
    Version1_1,
    Version1_2,

}
struct InstanceData {
    application_name: Vec<c_char>,
    engine_name: Vec<c_char>,
    enabled_layers: Vec<Vec<c_char>>,
    enabled_extensions: Vec<Vec<c_char>>,
}

pub struct Instance {
    data: Box<InstanceData>,
    entry: ash::Entry,
    instance: ash::Instance,
}

impl Drop for Instance {
    fn drop(&mut self) {
        //println!("> Dropping {}", self.name);
    }
}

impl Instance {
    pub fn new(instance_extensions: &[String], engine_name: &String, app_name: &String, api_version: &InstanceAPIVersion) -> Result<Instance, VkError> {
        let mut app_bytes = app_name.to_owned().into_bytes();
        app_bytes.push(b"\0"[0]);

        let mut engine_bytes = engine_name.to_owned().into_bytes();
        engine_bytes.push(b"\0"[0]);

        let data = Box::<InstanceData>::new(
            InstanceData {
                application_name: app_bytes.iter().map(|b| *b as c_char).collect::<Vec<c_char>>(),
                engine_name: engine_bytes.iter().map(|b| *b as c_char).collect::<Vec<c_char>>(),
                enabled_layers: vec!(),
                enabled_extensions: instance_extensions.iter().map(|ext_name| {
                    let mut ext_bytes = ext_name.to_owned().into_bytes();
                    ext_bytes.push(b"\0"[0]);

                    ext_bytes.iter().map(|b| *b as c_char).collect::<Vec<c_char>>()}
                ).collect()
            }
        );

        let layers_ptr = data.enabled_layers.iter().map(|str| str.as_ptr()).collect::<Vec<*const c_char>>();
        let extensions_ptr = data.enabled_extensions.iter().map(|str| str.as_ptr()).collect::<Vec<*const c_char>>();

        unsafe {
            let app_info = ash::vk::ApplicationInfo {
                s_type: ash::vk::StructureType::APPLICATION_INFO,
                p_next: ::std::ptr::null(),
                p_application_name: data.as_ref().application_name.as_ptr(),
                application_version: 0,
                p_engine_name: data.as_ref().engine_name.as_ptr(),
                engine_version: 0,
                api_version: ash::vk::make_api_version(0, 1, 0, 0)
            };

            let create_info = ash::vk::InstanceCreateInfo
                {
                    s_type: ash::vk::StructureType::INSTANCE_CREATE_INFO,
                    p_next: ::std::ptr::null(),
                    flags: ash::vk::InstanceCreateFlags::default(),
                    p_application_info: &app_info as *const ash::vk::ApplicationInfo,
                    pp_enabled_layer_names: layers_ptr.as_ptr(),
                    enabled_layer_count: layers_ptr.len() as u32,
                    pp_enabled_extension_names: extensions_ptr.as_ptr(),
                    enabled_extension_count: extensions_ptr.len() as u32,
                }
            ;

            if let Ok(entry) = ash::Entry::load() {
                if let Ok(instance) = entry.create_instance(&create_info, None) {
                    return Ok(
                        Instance {
                            data: data,
                            entry: entry,
                            instance: instance
                        }
                    );
                }

                return Err(
                    VkError {}
                )
                
            }

            return Err(
                VkError {}
            )
            
        }

    }


}
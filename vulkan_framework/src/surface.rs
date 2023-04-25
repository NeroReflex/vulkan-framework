use ash;

use crate::instance::{Instance, InstanceOwned};
use crate::prelude::*;

use std::sync::Arc;

pub struct Surface {
    instance: Arc<Instance>,
    surface: ash::vk::SurfaceKHR,
}

impl InstanceOwned for Surface {
    fn get_parent_instance(&self) -> Arc<Instance> {
        self.instance.clone()
    }
}

impl Drop for Surface {
    fn drop(&mut self) {
        match self.instance.get_surface_khr_extension() {
            Some(surface_khr_ext) => unsafe {
                surface_khr_ext.destroy_surface(self.surface, self.instance.get_alloc_callbacks());
            },
            None => {
                #[cfg(debug_assertions)]
                {
                    panic!("Surface KHR extension has not been loaded, have you forgotten to specify surface support at instance creation? You have to call manually call vkDestroySurfaceKHR before destroy the Instance.");
                }
            }
        }
    }
}

impl Surface {
    pub fn new(instance: Arc<Instance>, surface: ash::vk::SurfaceKHR) -> VulkanResult<Arc<Self>> {
        Self::from_raw(instance, ash::vk::Handle::as_raw(surface))
    }

    pub fn from_raw(instance: Arc<Instance>, raw_surface_khr: u64) -> VulkanResult<Arc<Self>> {
        Ok(Arc::new(Self {
            instance,
            surface: ash::vk::Handle::from_raw(raw_surface_khr),
        }))
    }

    pub fn native_handle(&self) -> u64 {
        ash::vk::Handle::as_raw(self.surface)
    }

    pub(crate) fn ash_handle(&self) -> &ash::vk::SurfaceKHR {
        &self.surface
    }
}

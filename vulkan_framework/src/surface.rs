use ash;

use crate::instance::Instance;
use crate::prelude::*;

pub struct Surface<'surface> {
    instance: &'surface Instance<'surface>,
    surface: ash::vk::SurfaceKHR,
}

impl<'surface> Drop for Surface<'surface> {
    fn drop(&mut self) {
        match self.instance.get_surface_khr_extension() {
            Some(surface_khr_ext) => unsafe {
                surface_khr_ext.destroy_surface(self.surface, self.instance.get_alloc_callbacks());
            },
            None => {
                #[cfg(debug_assertions)]
                {
                    println!("Surface KHR extension has not been loaded, have you forgotten to specify surface support at instance creation? You have to call manually call vkDestroySurfaceKHR before destroy the Instance.");
                    assert_eq!(true, false)
                }
            }
        }
    }
}

impl<'surface> Surface<'surface> {
    pub fn new(instance: &'surface Instance, surface: ash::vk::SurfaceKHR) -> VulkanResult<Self> {
        Self::from_raw(instance, ash::vk::Handle::as_raw(surface) as u64)
    }

    pub fn from_raw(instance: &'surface Instance, raw_surface_khr: u64) -> VulkanResult<Self> {
        Ok(Self {
            instance: instance,
            surface: ash::vk::Handle::from_raw(raw_surface_khr),
        })
    }

    pub fn native_handle(&self) -> u64 {
        ash::vk::Handle::as_raw(self.surface)
    }

    pub(crate) fn ash_handle(&self) -> &ash::vk::SurfaceKHR {
        &self.surface
    }
}

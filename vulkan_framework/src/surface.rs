use ash;

use std::sync::{Arc, Weak};

use crate::instance::Instance;

pub struct Surface {
    instance: Weak<Instance>,
    surface: ash::vk::SurfaceKHR,
}

impl Drop for Surface {
    fn drop(&mut self) {
        match self.instance.upgrade() {
            Some(framework_instance) => match framework_instance.get_surface_khr_extension() {
                Some(surface_khr_ext) => unsafe {
                    surface_khr_ext
                        .destroy_surface(self.surface, framework_instance.get_alloc_callbacks());
                },
                None => {}
            },
            None => {}
        }
    }
}

impl Surface {
    pub fn new(instance: Arc<Instance>, surface: ash::vk::SurfaceKHR) -> Arc<Self> {
        Arc::new(Self {
            instance: Arc::downgrade(&instance),
            surface: surface,
        })
    }

    pub fn from_raw(instance: Arc<Instance>, raw_surface_khr: u64) -> Arc<Self> {
        Arc::new(Self {
            instance: Arc::downgrade(&instance),
            surface: ash::vk::Handle::from_raw(raw_surface_khr),
        })
    }

    pub fn native_handle(&self) -> &ash::vk::SurfaceKHR {
        &self.surface
    }
}

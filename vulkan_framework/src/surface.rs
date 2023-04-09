use ash;

use std::sync::{Arc, Weak, Mutex};

use crate::instance::Instance;
use crate::prelude::*;

pub struct Surface {
    instance: Weak<Mutex<Instance>>,
    surface: ash::vk::SurfaceKHR,
}

impl Drop for Surface {
    fn drop(&mut self) {
        match self.instance.upgrade() {
            Some(instance_mutex) => {
                match instance_mutex.lock() {
                    Ok(instance) => {
                        match instance.get_surface_khr_extension() {
                            Some(surface_khr_ext) => unsafe {
                                surface_khr_ext
                                    .destroy_surface(self.surface, instance.get_alloc_callbacks());
                            },
                            None => {
                                #[cfg(debug_assertions)]
                                {
                                    println!("Surface KHR extension has not been loaded, have you forgotten to specify surface support at instance creation? You have to call manually call vkDestroySurfaceKHR before destroy the Instance.");
                                    assert_eq!(true, false)
                                }
                            }
                        }
                    },
                    Err(err) => {
                        #[cfg(debug_assertions)]
                        {
                            println!("Error acquiring Instance mutex: {}", err);
                            assert_eq!(true, false)
                        }
                    }
                }
                
            },
            None => {
                #[cfg(debug_assertions)]
                {
                    println!("Instance has already been deleted, if you had validation layers enabled you would have known.");
                    assert_eq!(true, false)
                }
            }
        }
    }
}

impl Surface {
    pub fn new(instance: Weak<Mutex<Instance>>, surface: ash::vk::SurfaceKHR) -> VulkanResult<Arc<Mutex<Self>>> {
        Self::from_raw(instance, ash::vk::Handle::as_raw(surface) as u64)
    }

    pub fn from_raw(instance: Weak<Mutex<Instance>>, raw_surface_khr: u64) -> VulkanResult<Arc<Mutex<Self>>> {
        match instance.upgrade() {
            Some(_) => Ok(Arc::new(Mutex::new(Self {
                instance: instance,
                surface: ash::vk::Handle::from_raw(raw_surface_khr),
            }))),
            None => {
                #[cfg(debug_assertions)]
                {
                    println!("The provided Instance does not exists (anymore?). Have you dropped the last Arc<> holding it before calling this function?");
                    assert_eq!(true, false);
                }

                Err(VulkanError::new())
            }
        }
    }

    pub fn native_handle(&self) -> &ash::vk::SurfaceKHR {
        &self.surface
    }
}

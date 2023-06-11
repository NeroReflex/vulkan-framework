use std::sync::Arc;

use crate::{device::{Device, DeviceOwned}, instance::InstanceOwned, prelude::{VulkanResult, VulkanError}};


#[derive(Copy, Clone, PartialEq, Eq)]
pub enum Filtering {
    Nearest,
    Linear,
    Cubic
}

impl Filtering {
    pub(crate) fn ash_flags(&self) -> ash::vk::Filter {
        match self {
            Filtering::Cubic => ash::vk::Filter::CUBIC_IMG,
            Filtering::Linear => ash::vk::Filter::LINEAR,
            Filtering::Nearest => ash::vk::Filter::NEAREST,
        }
    } 
}

#[derive(Copy, Clone, PartialEq, Eq)]
pub enum MipmapMode {
    ModeNearest,
    ModeLinear
}

impl MipmapMode {
    pub(crate) fn ash_flags(&self) -> ash::vk::SamplerMipmapMode {
        match self {
            MipmapMode::ModeNearest => ash::vk::SamplerMipmapMode::NEAREST,
            MipmapMode::ModeLinear => ash::vk::SamplerMipmapMode::LINEAR,
        }
    } 
}

pub struct Sampler {
    device: Arc<Device>,
    sampler: ash::vk::Sampler,
    mag_filter: Filtering,
    min_filter: Filtering,
    mipmap_mode: MipmapMode,
    max_anisotropy: f32,
}

impl DeviceOwned for Sampler {
    fn get_parent_device(&self) -> Arc<Device> {
        self.device.clone()
    }
}

impl Drop for Sampler {
    fn drop(&mut self) {
        unsafe {
            self.device.ash_handle().destroy_sampler(self.sampler, self.device.get_parent_instance().get_alloc_callbacks())
        }
    }
}

impl Sampler {
    pub fn max_anisotropy(&self) -> f32 {
        self.max_anisotropy
    }

    pub fn is_anisotropic_enabled(&self) -> bool {
        self.max_anisotropy > 1.0
    }

    pub(crate) fn ash_native(&self) -> ash::vk::Sampler {
        self.sampler
    }

    pub fn native_handle(&self) -> u64 {
        ash::vk::Handle::as_raw(self.sampler)
    }

    pub fn new(
        device: Arc<Device>,
        mag_filter: Filtering,
        min_filter: Filtering,
        mipmap_mode: MipmapMode,
        max_anisotropy: f32,
    ) -> VulkanResult<Arc<Self>> {
        let create_info = ash::vk::SamplerCreateInfo::builder()
            .border_color(ash::vk::BorderColor::INT_OPAQUE_BLACK)
            .anisotropy_enable(max_anisotropy > 1.0)
            .max_anisotropy(max_anisotropy)
            .unnormalized_coordinates(false)
            .address_mode_u(ash::vk::SamplerAddressMode::REPEAT)
            .address_mode_v(ash::vk::SamplerAddressMode::REPEAT)
            .address_mode_w(ash::vk::SamplerAddressMode::REPEAT)
            .mipmap_mode(mipmap_mode.ash_flags())
            .mag_filter(mag_filter.ash_flags())
            .min_filter(min_filter.ash_flags())
            .build();

        match unsafe {
            device.ash_handle().create_sampler(&create_info, device.get_parent_instance().get_alloc_callbacks())
        } {
            Ok(sampler) => {
                Ok(
                    Arc::new(
                        Self {
                            device,
                            sampler,
                            mag_filter,
                            min_filter,
                            mipmap_mode,
                            max_anisotropy,
                        }
                    )
                )
            },
            Err(err) => {
                Err(VulkanError::Vulkan(err.as_raw()))
            }
        }
    }
}
use std::sync::Arc;

use crate::{
    device::{Device, DeviceOwned},
    instance::InstanceOwned,
    prelude::{FrameworkError, VulkanError, VulkanResult},
    shader_layout_binding::BindingDescriptor,
};

pub struct DescriptorSetLayout {
    device: Arc<Device>,
    layout: ash::vk::DescriptorSetLayout,
    descriptors: Vec<Arc<BindingDescriptor>>,
}

pub trait DescriptorSetLayoutDependant {
    fn get_parent_descriptor_set_layout(&self) -> Arc<DescriptorSetLayout>;
}

impl Drop for DescriptorSetLayout {
    fn drop(&mut self) {
        unsafe {
            self.get_parent_device()
                .ash_handle()
                .destroy_descriptor_set_layout(
                    self.layout,
                    self.get_parent_device()
                        .get_parent_instance()
                        .get_alloc_callbacks(),
                )
        }
    }
}

impl DeviceOwned for DescriptorSetLayout {
    fn get_parent_device(&self) -> Arc<Device> {
        self.device.clone()
    }
}

impl DescriptorSetLayout {
    pub fn native_handle(&self) -> u64 {
        ash::vk::Handle::as_raw(self.layout)
    }

    pub(crate) fn ash_handle(&self) -> ash::vk::DescriptorSetLayout {
        self.layout
    }

    pub fn binding_range(&self) -> (u32, u32) {
        let mut min_idx: u32 = 1000;
        let mut max_idx: u32 = 0;

        for range in self.descriptors.iter() {
            let (start, end) = range.binding_range();
            min_idx = start.min(min_idx);
            max_idx = end.max(max_idx);
        }

        (min_idx, max_idx)
    }

    pub fn descriptors(&self) -> Vec<Arc<BindingDescriptor>> {
        self.descriptors.clone()
    }

    /*pub fn from_shaders(shaders: &[Arc<dyn ShaderTrait>]) -> VulkanResult<Arc<Self>> {
        let mut bindings: Vec<Arc<BindingDescriptor>> = Vec::new();

        let mut maybe_device: Option<Arc<Device>> = Option::None;

        for shader in shaders.iter() {
            match &maybe_device {
                Option::None => {
                    maybe_device = Option::Some(shader.get_parent_device());
                }
                Option::Some(dev) => {
                    if dev != &shader.get_parent_device() {
                        return Err(VulkanError::_);
                    }
                }
            }

            let mut current_shader_collection = shader.get_parent_binding_descriptors();

            bindings.append(&mut current_shader_collection);
        }

        let dev = match maybe_device {
            Option::Some(device) => device,
            Option::None => return Err(VulkanError::_),
        };

        Self::new(dev, bindings.as_slice())
    }*/

    pub fn new(
        device: Arc<Device>,
        descriptors: &[Arc<BindingDescriptor>],
    ) -> VulkanResult<Arc<Self>> {
        if descriptors.is_empty() {
            return Err(VulkanError::Framework(FrameworkError::Unknown(Some(
                "Error creating the descriptor set layout: no binding descriptors specified"
                    .to_string(),
            ))));
        }

        // a collection of VkDescriptorSetLayoutBinding
        let bindings: Vec<ash::vk::DescriptorSetLayoutBinding> =
            descriptors.iter().map(|d| d.ash_handle()).collect();

        let create_info = ash::vk::DescriptorSetLayoutCreateInfo::default()
            .bindings(bindings.as_slice());

        match unsafe {
            device.ash_handle().create_descriptor_set_layout(
                &create_info,
                device.get_parent_instance().get_alloc_callbacks(),
            )
        } {
            Ok(layout) => Ok(Arc::new(Self {
                device,
                layout,
                descriptors: descriptors.to_vec(),
            })),
            Err(err) => Err(VulkanError::Vulkan(
                err.as_raw(),
                Some(format!("Error creating the descriptor set layout: {}", err)),
            )),
        }
    }
}

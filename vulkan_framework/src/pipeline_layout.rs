use std::sync::Arc;
use std::vec::Vec;

use smallvec::SmallVec;

use crate::{
    descriptor_set_layout::DescriptorSetLayout,
    device::{Device, DeviceOwned},
    instance::InstanceOwned,
    prelude::{VulkanError, VulkanResult},
    push_constant_range::PushConstanRange,
};

pub struct PipelineLayout {
    device: Arc<Device>,
    layout_bindings: SmallVec<[Arc<DescriptorSetLayout>; 8]>,
    push_constant_ranges: SmallVec<[Arc<PushConstanRange>; 8]>,
    pipeline_layout: ash::vk::PipelineLayout,
}

pub trait PipelineLayoutDependant {
    fn get_parent_pipeline_layout(&self) -> Arc<PipelineLayout>;
}

impl Drop for PipelineLayout {
    fn drop(&mut self) {
        unsafe {
            self.device.ash_handle().destroy_pipeline_layout(
                self.pipeline_layout,
                self.device.get_parent_instance().get_alloc_callbacks(),
            )
        }
    }
}

impl DeviceOwned for PipelineLayout {
    fn get_parent_device(&self) -> Arc<Device> {
        self.device.clone()
    }
}

impl PipelineLayout {
    pub fn native_handle(&self) -> u64 {
        ash::vk::Handle::as_raw(self.pipeline_layout)
    }

    pub(crate) fn ash_handle(&self) -> ash::vk::PipelineLayout {
        self.pipeline_layout
    }

    pub fn new(
        device: Arc<Device>,
        binding_descriptors: &[Arc<DescriptorSetLayout>],
        constant_ranges: &[Arc<PushConstanRange>],
        debug_name: Option<&str>,
    ) -> VulkanResult<Arc<Self>> {
        let set_layouts = binding_descriptors
            .iter()
            .map(|layout_binding| {
                // TODO: make sure all of these are from the same device
                //assert_eq!(layout_binding.get_parent_device(), device);

                layout_binding.ash_handle()
            })
            .collect::<Vec<ash::vk::DescriptorSetLayout>>();

        let ranges = constant_ranges
            .iter()
            .map(|r| r.ash_handle())
            .collect::<Vec<ash::vk::PushConstantRange>>();

        let create_info = ash::vk::PipelineLayoutCreateInfo::builder()
            .set_layouts(set_layouts.as_slice())
            .push_constant_ranges(ranges.as_slice())
            .build();

        match unsafe {
            device.ash_handle().create_pipeline_layout(
                &create_info,
                device.get_parent_instance().get_alloc_callbacks(),
            )
        } {
            Ok(pipeline_layout) => {
                let mut obj_name_bytes = vec![];
                if let Some(ext) = device.get_parent_instance().get_debug_ext_extension() {
                    if let Some(name) = debug_name {
                        for name_ch in name.as_bytes().iter() {
                            obj_name_bytes.push(*name_ch);
                        }
                        obj_name_bytes.push(0x00);

                        unsafe {
                            let object_name = std::ffi::CStr::from_bytes_with_nul_unchecked(
                                obj_name_bytes.as_slice(),
                            );
                            // set device name for debugging
                            let dbg_info = ash::vk::DebugUtilsObjectNameInfoEXT::builder()
                                .object_type(ash::vk::ObjectType::PIPELINE_LAYOUT)
                                .object_handle(ash::vk::Handle::as_raw(pipeline_layout))
                                .object_name(object_name)
                                .build();

                            match ext.set_debug_utils_object_name(
                                device.ash_handle().handle(),
                                &dbg_info,
                            ) {
                                Ok(_) => {
                                    #[cfg(debug_assertions)]
                                    {
                                        println!("Pipeline Layout Debug object name changed");
                                    }
                                }
                                Err(err) => {
                                    #[cfg(debug_assertions)]
                                    {
                                        panic!("Error setting the Debug name for the newly created Pipeline Layout, will use handle. Error: {}", err)
                                    }
                                }
                            }
                        }
                    }
                }

                Ok(Arc::new(Self {
                    device,
                    pipeline_layout,
                    layout_bindings: binding_descriptors.iter().map(|e| e.clone()).collect(),
                    push_constant_ranges: constant_ranges.iter().map(|e| e.clone()).collect(),
                }))
            }
            Err(err) => {
                #[cfg(debug_assertions)]
                {
                    panic!("Error creating the pipeline layout: {}", err)
                }

                Err(VulkanError::Unspecified)
            }
        }
    }
}

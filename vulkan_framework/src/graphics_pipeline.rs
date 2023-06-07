use std::ffi::CStr;
use std::sync::Arc;

use ash::vk::Offset2D;

use crate::device::{Device, DeviceOwned};
use crate::fragment_shader::FragmentShader;
use crate::graphics_pipeline;
use crate::image::{Image1DTrait, Image2DDimensions, Image2DTrait, ImageMultisampling};
use crate::instance::InstanceOwned;
use crate::renderpass::RenderPass;
use crate::vertex_shader::VertexShader;

use crate::pipeline_layout::{PipelineLayout, PipelineLayoutDependant};
use crate::prelude::{VulkanError, VulkanResult};
use crate::shader_trait::PrivateShaderTrait;

#[derive(Copy, Clone, PartialEq, Eq)]
pub enum AttributeType {
    Float,
    Vec1,
    Vec2,
    Vec3,
    Vec4,

    Uint,
    Uvec1,
    Uvec2,
    Uvec3,
    Uvec4,

    Sint,
    Ivec1,
    Ivec2,
    Ivec3,
    Ivec4,
}

impl AttributeType {
    pub(crate) fn ash_format(&self) -> ash::vk::Format {
        match self {
            AttributeType::Float => ash::vk::Format::R32_SFLOAT,
            AttributeType::Vec1 => ash::vk::Format::R32_SFLOAT,
            AttributeType::Vec2 => ash::vk::Format::R32G32_SFLOAT,
            AttributeType::Vec3 => ash::vk::Format::R32G32B32_SFLOAT,
            AttributeType::Vec4 => ash::vk::Format::R32G32B32A32_SFLOAT,

            AttributeType::Uint => ash::vk::Format::R32_UINT,
            AttributeType::Uvec1 => ash::vk::Format::R32_UINT,
            AttributeType::Uvec2 => ash::vk::Format::R32G32_UINT,
            AttributeType::Uvec3 => ash::vk::Format::R32G32B32_UINT,
            AttributeType::Uvec4 => ash::vk::Format::R32G32B32A32_UINT,

            AttributeType::Sint => ash::vk::Format::R32_SINT,
            AttributeType::Ivec1 => ash::vk::Format::R32_SINT,
            AttributeType::Ivec2 => ash::vk::Format::R32G32_SINT,
            AttributeType::Ivec3 => ash::vk::Format::R32G32B32_SINT,
            AttributeType::Ivec4 => ash::vk::Format::R32G32B32A32_SINT,
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq)]
pub struct VertexInputAttribute {
    location: u32,
    offset: u32,
    attr_type: AttributeType,
}

impl VertexInputAttribute {
    /*pub fn ash_description(&self) -> ash::vk::VertexInputAttributeDescription {
        ash::vk::VertexInputAttributeDescription::builder()
            .binding(self.)
            .build()
    }*/

    pub(crate) fn ash_format(&self) -> ash::vk::Format {
        self.attr_type.ash_format()
    }

    pub fn attr_type(&self) -> AttributeType {
        self.attr_type
    }

    pub fn location(&self) -> u32 {
        self.location
    }

    pub fn offset(&self) -> u32 {
        self.offset
    }

    pub fn new(location: u32, offset: u32, attr_type: AttributeType) -> Self {
        Self {
            location,
            offset,
            attr_type,
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq)]
pub enum VertexInputRate {
    PerInstance,
    PerVertex,
}

impl VertexInputRate {
    pub(crate) fn ash_input_rate(&self) -> ash::vk::VertexInputRate {
        match self {
            VertexInputRate::PerVertex => ash::vk::VertexInputRate::VERTEX,
            VertexInputRate::PerInstance => ash::vk::VertexInputRate::INSTANCE,
        }
    }
}

pub struct VertexInputBinding {
    input_rate: VertexInputRate,
    stride: u32,
    attributes: smallvec::SmallVec<[VertexInputAttribute; 16]>,
}

impl VertexInputBinding {
    pub fn input_rate(&self) -> VertexInputRate {
        self.input_rate
    }

    pub fn stride(&self) -> u32 {
        self.stride
    }

    pub fn attributes<'a>(&'a self) -> impl Iterator<Item = &'a VertexInputAttribute> {
        self.attributes.iter()
    }

    pub fn new(
        input_rate: VertexInputRate,
        stride: u32,
        attributes: &[VertexInputAttribute],
    ) -> Self {
        Self {
            input_rate,
            stride,
            attributes: attributes.iter().map(|ia| ia.clone()).collect(),
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq)]
pub enum CullMode {
    None,
}

impl CullMode {
    pub(crate) fn ash_flags(&self) -> ash::vk::CullModeFlags {
        match self {
            CullMode::None => ash::vk::CullModeFlags::NONE,
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq)]
pub enum FrontFace {
    Clockwise,
    CounterClockwise,
}

#[derive(Copy, Clone, PartialEq, Eq)]
pub enum PolygonMode {
    Fill,
}

#[derive(Copy, Clone, PartialEq)]
pub struct Rasterizer {
    cull_mode: CullMode,
    polygon_mode: PolygonMode,
    front_face: FrontFace,
    enable_depth_bias: bool,
    depth_bias_sonstant_factor: f32,
    depth_bias_clamp: Option<f32>,
    depth_bias_slope_factor: f32,
}

impl Rasterizer {
    pub fn polygon_mode(&self) -> PolygonMode {
        self.polygon_mode
    }

    pub fn front_face(&self) -> FrontFace {
        self.front_face
    }

    pub fn cull_mode(&self) -> CullMode {
        self.cull_mode
    }

    pub fn enable_depth_bias(&self) -> bool {
        self.enable_depth_bias
    }

    pub fn depth_bias_sonstant_factor(&self) -> f32 {
        self.depth_bias_sonstant_factor
    }

    pub fn depth_bias_clamp(&self) -> Option<f32> {
        self.depth_bias_clamp
    }

    pub fn depth_bias_slope_factor(&self) -> f32 {
        self.depth_bias_slope_factor
    }

    pub fn new(
        polygon_mode: PolygonMode,
        front_face: FrontFace,
        cull_mode: CullMode,
        enable_depth_bias: bool,
        depth_bias_sonstant_factor: f32,
        depth_bias_clamp: Option<f32>,
        depth_bias_slope_factor: f32,
    ) -> Self {
        Self {
            polygon_mode,
            front_face,
            cull_mode,
            enable_depth_bias,
            depth_bias_sonstant_factor,
            depth_bias_clamp,
            depth_bias_slope_factor,
        }
    }
}

pub struct GraphicsPipeline {
    device: Arc<Device>,
    renderpass: Arc<RenderPass>,
    subpass_index: u32,
    pipeline_layout: Arc<PipelineLayout>,
    pipeline: ash::vk::Pipeline,
}

impl PipelineLayoutDependant for GraphicsPipeline {
    fn get_parent_pipeline_layout(&self) -> Arc<PipelineLayout> {
        self.pipeline_layout.clone()
    }
}

impl DeviceOwned for GraphicsPipeline {
    fn get_parent_device(&self) -> Arc<Device> {
        self.device.clone()
    }
}

impl Drop for GraphicsPipeline {
    fn drop(&mut self) {
        unsafe {
            self.device.ash_handle().destroy_pipeline(
                self.pipeline,
                self.device.get_parent_instance().get_alloc_callbacks(),
            )
        }
    }
}

/*impl DescriptorSetLayoutsDependant for ComputePipeline {
    fn get_descriptor_set_layouts(&self) -> Arc<DescriptorSetLayout> {
        self.descriptor_set_layouts.clone()
    }
}*/

impl GraphicsPipeline {
    pub fn subpass_index(&self) -> u32 {
        self.subpass_index
    }

    pub fn native_handle(&self) -> u64 {
        ash::vk::Handle::as_raw(self.pipeline)
    }

    pub(crate) fn ash_handle(&self) -> ash::vk::Pipeline {
        self.pipeline
    }

    pub fn new(
        renderpass: Arc<RenderPass>,
        subpass_index: u32,
        multisampling: ImageMultisampling,
        dimensions: Image2DDimensions,
        pipeline_layout: Arc<PipelineLayout>,
        bindings: &[VertexInputBinding],
        rasterizer: Rasterizer,
        vertex_shader: (Arc<VertexShader>, Option<String>),
        fragment_shader: (Arc<FragmentShader>, Option<String>),
        debug_name: Option<&str>,
    ) -> VulkanResult<Arc<Self>> {
        // TODO: assert the device is the same between renderpass and pipeline layout

        let device = pipeline_layout.get_parent_device();

        let mut vertex_binding_descriptions: smallvec::SmallVec<
            [ash::vk::VertexInputBindingDescription; 16],
        > = smallvec::smallvec![];
        let mut vertex_attribute_descriptions: smallvec::SmallVec<
            [ash::vk::VertexInputAttributeDescription; 16],
        > = smallvec::smallvec![];

        for (binding_index, binding) in bindings.iter().enumerate() {
            vertex_binding_descriptions.push(
                ash::vk::VertexInputBindingDescription::builder()
                    .binding(binding_index as u32)
                    .input_rate(binding.input_rate().ash_input_rate())
                    .stride(binding.stride())
                    .build(),
            );

            for attribute_on_binding in binding.attributes() {
                vertex_attribute_descriptions.push(
                    ash::vk::VertexInputAttributeDescription::builder()
                        .binding(binding_index as u32)
                        .offset(attribute_on_binding.offset())
                        .location(attribute_on_binding.location())
                        .format(attribute_on_binding.ash_format())
                        .build(),
                )
            }
        }

        let (vertex_shader_module, vertex_shader_entry_name_opt) = vertex_shader;
        let vertex_shader_entry_name: &CStr = match vertex_shader_entry_name_opt {
            Option::Some(_n) => {
                todo!()
            }
            Option::None => {
                unsafe { CStr::from_bytes_with_nul_unchecked(&[109u8, 97u8, 105u8, 110u8, 0u8]) }
                // main
            }
        };

        let (fragment_shader_module, fragment_shader_entry_name_opt) = fragment_shader;
        let fragment_shader_entry_name: &CStr = match fragment_shader_entry_name_opt {
            Option::Some(_n) => {
                todo!()
            }
            Option::None => {
                unsafe { CStr::from_bytes_with_nul_unchecked(&[109u8, 97u8, 105u8, 110u8, 0u8]) }
                // main
            }
        };

        let pipeline_shader_stage_create_info: smallvec::SmallVec<
            [ash::vk::PipelineShaderStageCreateInfo; 8],
        > = smallvec::smallvec![
            ash::vk::PipelineShaderStageCreateInfo::builder()
                .module(vertex_shader_module.ash_handle())
                .name(vertex_shader_entry_name)
                .stage(ash::vk::ShaderStageFlags::VERTEX)
                .build(),
            ash::vk::PipelineShaderStageCreateInfo::builder()
                .module(fragment_shader_module.ash_handle())
                .name(fragment_shader_entry_name)
                .stage(ash::vk::ShaderStageFlags::FRAGMENT)
                .build()
        ];

        let vertex_input_state_create_info = ash::vk::PipelineVertexInputStateCreateInfo::builder()
            .vertex_attribute_descriptions(vertex_attribute_descriptions.as_slice())
            .vertex_binding_descriptions(vertex_binding_descriptions.as_slice())
            .build();

        let multisampling_create_info = ash::vk::PipelineMultisampleStateCreateInfo::builder()
            .rasterization_samples(multisampling.ash_samples())
            .build();

        let viewport = ash::vk::Viewport::builder()
            .x(0.0)
            .y(0.0)
            .width(dimensions.width() as f32)
            .height(dimensions.height() as f32)
            .build();

        let scissor = ash::vk::Rect2D::builder()
            .extent(
                ash::vk::Extent2D::builder()
                    .width(dimensions.width())
                    .height(dimensions.height())
                    .build(),
            )
            .offset(ash::vk::Offset2D::builder().build())
            .build();

        let viewport_state_create_info = ash::vk::PipelineViewportStateCreateInfo::builder()
            .viewports(&[viewport])
            .scissors(&[scissor])
            .build();

        let rasterization_state_create_info =
            ash::vk::PipelineRasterizationStateCreateInfo::builder()
                .cull_mode(rasterizer.cull_mode().ash_flags())
                .build();

        let input_assembly_create_info = ash::vk::PipelineInputAssemblyStateCreateInfo::builder()
            .topology(ash::vk::PrimitiveTopology::TRIANGLE_LIST)
            .primitive_restart_enable(false)
            .build();

        let create_info = ash::vk::GraphicsPipelineCreateInfo::builder()
            .layout(pipeline_layout.ash_handle())
            .render_pass(renderpass.ash_handle())
            .subpass(subpass_index)
            .vertex_input_state(&vertex_input_state_create_info)
            .multisample_state(&multisampling_create_info)
            .viewport_state(&viewport_state_create_info)
            .rasterization_state(&rasterization_state_create_info)
            .stages(pipeline_shader_stage_create_info.as_slice())
            .input_assembly_state(&input_assembly_create_info)
            .build();

        match unsafe {
            device.ash_handle().create_graphics_pipelines(
                ash::vk::PipelineCache::null(),
                &[create_info],
                device.get_parent_instance().get_alloc_callbacks(),
            )
        } {
            Ok(pipelines) => {
                let pipeline = pipelines[0];

                if pipelines.len() != 1 {
                    #[cfg(debug_assertions)]
                    {
                        panic!("Error creating the compute pipeline: expected 1 pipeline to be created, instead {} were created.", pipelines.len())
                    }

                    return Err(VulkanError::Unspecified);
                }

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
                                .object_type(ash::vk::ObjectType::PIPELINE)
                                .object_handle(ash::vk::Handle::as_raw(pipeline))
                                .object_name(object_name)
                                .build();

                            match ext.set_debug_utils_object_name(
                                device.ash_handle().handle(),
                                &dbg_info,
                            ) {
                                Ok(_) => {
                                    #[cfg(debug_assertions)]
                                    {
                                        println!("Pipeline Debug object name changed");
                                    }
                                }
                                Err(err) => {
                                    #[cfg(debug_assertions)]
                                    {
                                        panic!("Error setting the Debug name for the newly created Pipeline, will use handle. Error: {}", err)
                                    }
                                }
                            }
                        }
                    }
                }

                Ok(Arc::new(Self {
                    device,
                    renderpass,
                    subpass_index,
                    pipeline,
                    pipeline_layout,
                }))
            }
            Err(err) => Err(VulkanError::Unspecified),
        }
    }
}

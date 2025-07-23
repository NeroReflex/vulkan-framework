#[derive(Copy, Clone, PartialEq, Eq)]
pub enum PipelineStageAccelerationStructureKHR {
    Build,
    Copy,
}

impl From<PipelineStageAccelerationStructureKHR> for ash::vk::PipelineStageFlags2 {
    fn from(val: PipelineStageAccelerationStructureKHR) -> Self {
        type AshFlags = ash::vk::PipelineStageFlags2;

        match val {
            PipelineStageAccelerationStructureKHR::Build => {
                AshFlags::ACCELERATION_STRUCTURE_BUILD_KHR
            }
            PipelineStageAccelerationStructureKHR::Copy => {
                AshFlags::ACCELERATION_STRUCTURE_COPY_KHR
            }
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq)]
pub enum PipelineStageRayTracingPipelineKHR {
    RayTracingShader,
}

impl From<PipelineStageRayTracingPipelineKHR> for ash::vk::PipelineStageFlags2 {
    fn from(val: PipelineStageRayTracingPipelineKHR) -> Self {
        type AshFlags = ash::vk::PipelineStageFlags2;

        match val {
            PipelineStageRayTracingPipelineKHR::RayTracingShader => {
                AshFlags::RAY_TRACING_SHADER_KHR
            }
        }
    }
}

/// Represents VkPipelineStageFlagBits2 with VK_KHR_synchronization2 that has been promoted to core
/// See https://registry.khronos.org/vulkan/specs/latest/man/html/VkPipelineStageFlagBits2.html
#[derive(Copy, Clone, PartialEq, Eq)]
pub enum PipelineStage {
    None,
    Clear,
    TopOfPipe,
    DrawIndirect,
    VertexInput,
    VertexShader,
    TessellationControlShader,
    TessellationEvaluationShader,
    GeometryShader,
    FragmentShader,
    EarlyFragmentTests,
    LateFragmentTests,
    ColorAttachmentOutput,
    ComputeShader,
    Transfer,
    BottomOfPipe,
    Host,
    AllGraphics,
    AllCommands,
    RayTracingPipelineKHR(PipelineStageRayTracingPipelineKHR),
    AccelerationStructureKHR(PipelineStageAccelerationStructureKHR),
}

impl From<PipelineStage> for ash::vk::PipelineStageFlags {
    fn from(val: PipelineStage) -> Self {
        type AshFlags = ash::vk::PipelineStageFlags;
        match val {
            PipelineStage::TopOfPipe => AshFlags::TOP_OF_PIPE,
            PipelineStage::DrawIndirect => AshFlags::DRAW_INDIRECT,
            PipelineStage::VertexInput => AshFlags::VERTEX_INPUT,
            PipelineStage::VertexShader => AshFlags::VERTEX_SHADER,
            PipelineStage::TessellationControlShader => AshFlags::TESSELLATION_CONTROL_SHADER,
            PipelineStage::TessellationEvaluationShader => AshFlags::TESSELLATION_EVALUATION_SHADER,
            PipelineStage::GeometryShader => AshFlags::GEOMETRY_SHADER,
            PipelineStage::FragmentShader => AshFlags::FRAGMENT_SHADER,
            PipelineStage::EarlyFragmentTests => AshFlags::EARLY_FRAGMENT_TESTS,
            PipelineStage::LateFragmentTests => AshFlags::LATE_FRAGMENT_TESTS,
            PipelineStage::ColorAttachmentOutput => AshFlags::COLOR_ATTACHMENT_OUTPUT,
            PipelineStage::ComputeShader => AshFlags::COMPUTE_SHADER,
            PipelineStage::Transfer => AshFlags::TRANSFER,
            PipelineStage::BottomOfPipe => AshFlags::BOTTOM_OF_PIPE,
            PipelineStage::Host => AshFlags::HOST,
            PipelineStage::AllGraphics => AshFlags::ALL_GRAPHICS,
            PipelineStage::AllCommands => AshFlags::ALL_COMMANDS,
            _ => panic!("Unsupported"),
        }
    }
}

impl From<PipelineStage> for ash::vk::PipelineStageFlags2 {
    fn from(val: PipelineStage) -> Self {
        type AshFlags = ash::vk::PipelineStageFlags2;
        match val {
            PipelineStage::None => AshFlags::NONE,
            PipelineStage::Clear => AshFlags::CLEAR,
            PipelineStage::TopOfPipe => AshFlags::TOP_OF_PIPE,
            PipelineStage::DrawIndirect => AshFlags::DRAW_INDIRECT,
            PipelineStage::VertexInput => AshFlags::VERTEX_INPUT,
            PipelineStage::VertexShader => AshFlags::VERTEX_SHADER,
            PipelineStage::TessellationControlShader => AshFlags::TESSELLATION_CONTROL_SHADER,
            PipelineStage::TessellationEvaluationShader => AshFlags::TESSELLATION_EVALUATION_SHADER,
            PipelineStage::GeometryShader => AshFlags::GEOMETRY_SHADER,
            PipelineStage::FragmentShader => AshFlags::FRAGMENT_SHADER,
            PipelineStage::EarlyFragmentTests => AshFlags::EARLY_FRAGMENT_TESTS,
            PipelineStage::LateFragmentTests => AshFlags::LATE_FRAGMENT_TESTS,
            PipelineStage::ColorAttachmentOutput => AshFlags::COLOR_ATTACHMENT_OUTPUT,
            PipelineStage::ComputeShader => AshFlags::COMPUTE_SHADER,
            PipelineStage::Transfer => AshFlags::TRANSFER,
            PipelineStage::BottomOfPipe => AshFlags::BOTTOM_OF_PIPE,
            PipelineStage::Host => AshFlags::HOST,
            PipelineStage::AllGraphics => AshFlags::ALL_GRAPHICS,
            PipelineStage::AllCommands => AshFlags::ALL_COMMANDS,
            PipelineStage::RayTracingPipelineKHR(value) => value.into(),
            PipelineStage::AccelerationStructureKHR(value) => value.into(),
        }
    }
}

#[derive(Default, Copy, Clone, PartialEq, Eq)]
#[repr(transparent)]
pub struct PipelineStages(ash::vk::PipelineStageFlags2);

impl From<&[PipelineStage]> for PipelineStages {
    fn from(value: &[PipelineStage]) -> Self {
        let mut flags = ash::vk::PipelineStageFlags2::default();
        for v in value.iter() {
            flags |= v.to_owned().into()
        }

        Self(flags)
    }
}

impl From<PipelineStages> for ash::vk::PipelineStageFlags {
    fn from(val: PipelineStages) -> Self {
        // good luck with that!
        ash::vk::PipelineStageFlags::from_raw(u32::try_from(val.0.as_raw()).unwrap())
    }
}

impl From<PipelineStages> for ash::vk::PipelineStageFlags2 {
    fn from(val: PipelineStages) -> Self {
        val.0
    }
}

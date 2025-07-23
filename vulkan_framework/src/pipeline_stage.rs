#[derive(Copy, Clone, PartialEq, Eq)]
pub enum PipelineStageAccelerationStructureKHR {
    Build,
    Copy,
}

impl Into<ash::vk::PipelineStageFlags2> for PipelineStageAccelerationStructureKHR {
    fn into(self) -> ash::vk::PipelineStageFlags2 {
        type AshFlags = ash::vk::PipelineStageFlags2;

        match self {
            Self::Build => AshFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
            Self::Copy => AshFlags::ACCELERATION_STRUCTURE_COPY_KHR,
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq)]
pub enum PipelineStageRayTracingPipelineKHR {
    RayTracingShader,
}

impl Into<ash::vk::PipelineStageFlags2> for PipelineStageRayTracingPipelineKHR {
    fn into(self) -> ash::vk::PipelineStageFlags2 {
        type AshFlags = ash::vk::PipelineStageFlags2;

        match self {
            Self::RayTracingShader => AshFlags::RAY_TRACING_SHADER_KHR,
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

impl Into<ash::vk::PipelineStageFlags> for PipelineStage {
    fn into(self) -> ash::vk::PipelineStageFlags {
        type AshFlags = ash::vk::PipelineStageFlags;
        match self {
            Self::TopOfPipe => AshFlags::TOP_OF_PIPE,
            Self::DrawIndirect => AshFlags::DRAW_INDIRECT,
            Self::VertexInput => AshFlags::VERTEX_INPUT,
            Self::VertexShader => AshFlags::VERTEX_SHADER,
            Self::TessellationControlShader => AshFlags::TESSELLATION_CONTROL_SHADER,
            Self::TessellationEvaluationShader => AshFlags::TESSELLATION_EVALUATION_SHADER,
            Self::GeometryShader => AshFlags::GEOMETRY_SHADER,
            Self::FragmentShader => AshFlags::FRAGMENT_SHADER,
            Self::EarlyFragmentTests => AshFlags::EARLY_FRAGMENT_TESTS,
            Self::LateFragmentTests => AshFlags::LATE_FRAGMENT_TESTS,
            Self::ColorAttachmentOutput => AshFlags::COLOR_ATTACHMENT_OUTPUT,
            Self::ComputeShader => AshFlags::COMPUTE_SHADER,
            Self::Transfer => AshFlags::TRANSFER,
            Self::BottomOfPipe => AshFlags::BOTTOM_OF_PIPE,
            Self::Host => AshFlags::HOST,
            Self::AllGraphics => AshFlags::ALL_GRAPHICS,
            Self::AllCommands => AshFlags::ALL_COMMANDS,
            _ => panic!("Unsupported"),
        }
    }
}

impl Into<ash::vk::PipelineStageFlags2> for PipelineStage {
    fn into(self) -> ash::vk::PipelineStageFlags2 {
        type AshFlags = ash::vk::PipelineStageFlags2;
        match self {
            Self::None => AshFlags::NONE,
            Self::Clear => AshFlags::CLEAR,
            Self::TopOfPipe => AshFlags::TOP_OF_PIPE,
            Self::DrawIndirect => AshFlags::DRAW_INDIRECT,
            Self::VertexInput => AshFlags::VERTEX_INPUT,
            Self::VertexShader => AshFlags::VERTEX_SHADER,
            Self::TessellationControlShader => AshFlags::TESSELLATION_CONTROL_SHADER,
            Self::TessellationEvaluationShader => AshFlags::TESSELLATION_EVALUATION_SHADER,
            Self::GeometryShader => AshFlags::GEOMETRY_SHADER,
            Self::FragmentShader => AshFlags::FRAGMENT_SHADER,
            Self::EarlyFragmentTests => AshFlags::EARLY_FRAGMENT_TESTS,
            Self::LateFragmentTests => AshFlags::LATE_FRAGMENT_TESTS,
            Self::ColorAttachmentOutput => AshFlags::COLOR_ATTACHMENT_OUTPUT,
            Self::ComputeShader => AshFlags::COMPUTE_SHADER,
            Self::Transfer => AshFlags::TRANSFER,
            Self::BottomOfPipe => AshFlags::BOTTOM_OF_PIPE,
            Self::Host => AshFlags::HOST,
            Self::AllGraphics => AshFlags::ALL_GRAPHICS,
            Self::AllCommands => AshFlags::ALL_COMMANDS,
            Self::RayTracingPipelineKHR(value) => value.into(),
            Self::AccelerationStructureKHR(value) => value.into(),
        }
    }
}

#[derive(Default, Copy, Clone, PartialEq, Eq)]
#[repr(transparent)]
pub struct PipelineStages(ash::vk::PipelineStageFlags2);

impl From<&[PipelineStage]> for PipelineStages {
    fn from(value: &[PipelineStage]) -> Self {
        let mut flags = ash::vk::PipelineStageFlags2::default();
        for v in value.into_iter() {
            flags |= v.to_owned().into()
        }

        Self(flags)
    }
}

impl Into<ash::vk::PipelineStageFlags> for PipelineStages {
    fn into(self) -> ash::vk::PipelineStageFlags {
        // good luck with that!
        ash::vk::PipelineStageFlags::from_raw(u32::try_from(self.0.as_raw()).unwrap())
    }
}

impl Into<ash::vk::PipelineStageFlags2> for PipelineStages {
    fn into(self) -> ash::vk::PipelineStageFlags2 {
        self.0
    }
}

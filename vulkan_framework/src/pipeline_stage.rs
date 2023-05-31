#[repr(u32)]
#[derive(PartialEq, Copy, Clone)]
pub enum PipelineStageAccelerationStructureKHR {
    AccelerationStructureBuild = 0x02000000u32,
}

pub struct PipelineStagesAccelerationStructureKHR {
    acceleration_structure_build: bool,
}

impl PipelineStagesAccelerationStructureKHR {
    pub fn empty() -> Self {
        Self {
            acceleration_structure_build: false,
        }
    }

    pub fn from(flags: &[PipelineStageAccelerationStructureKHR]) -> Self {
        Self {
            acceleration_structure_build: flags
                .contains(&PipelineStageAccelerationStructureKHR::AccelerationStructureBuild),
        }
    }

    pub fn new(acceleration_structure_build: bool) -> Self {
        Self {
            acceleration_structure_build,
        }
    }

    pub(crate) fn ash_flags(&self) -> ash::vk::PipelineStageFlags {
        match self.acceleration_structure_build {
            true => ash::vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
            false => ash::vk::PipelineStageFlags::empty(),
        }
    }
}

#[repr(u32)]
#[derive(PartialEq, Copy, Clone)]
pub enum PipelineStageRayTracingPipelineKHR {
    RayTracingShader = 0x00200000u32,
}

pub struct PipelineStagesRayTracingPipelineKHR {
    ray_tracing_shader: bool,
}

impl PipelineStagesRayTracingPipelineKHR {
    pub fn empty() -> Self {
        Self {
            ray_tracing_shader: false,
        }
    }

    pub fn from(flags: &[PipelineStageRayTracingPipelineKHR]) -> Self {
        Self {
            ray_tracing_shader: flags
                .contains(&PipelineStageRayTracingPipelineKHR::RayTracingShader),
        }
    }

    pub fn new(ray_tracing_shader: bool) -> Self {
        Self { ray_tracing_shader }
    }

    pub(crate) fn ash_flags(&self) -> ash::vk::PipelineStageFlags {
        match self.ray_tracing_shader {
            true => ash::vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
            false => ash::vk::PipelineStageFlags::empty(),
        }
    }
}

#[repr(u32)]
#[derive(PartialEq, Copy, Clone)]
pub enum PipelineStageSynchronization2KHR {
    None = 0x00000000u32,
}

pub struct PipelineStagesSynchronization2KHR {
    synchronization_2: bool,
}

impl PipelineStagesSynchronization2KHR {
    pub fn empty() -> Self {
        Self {
            synchronization_2: false,
        }
    }

    pub fn from(flags: &[PipelineStageSynchronization2KHR]) -> Self {
        Self {
            synchronization_2: flags.contains(&PipelineStageSynchronization2KHR::None),
        }
    }

    pub fn new(synchronization_2: bool) -> Self {
        Self { synchronization_2 }
    }

    pub(crate) fn ash_flags(&self) -> ash::vk::PipelineStageFlags {
        match self.synchronization_2 {
            true => ash::vk::PipelineStageFlags::NONE,
            false => ash::vk::PipelineStageFlags::empty(),
        }
    }
}

#[repr(u32)]
#[derive(PartialEq, Copy, Clone)]
pub enum PipelineStage {
    TopOfPipe = 0x00000001u32,
    DrawIndirect = 0x00000002u32,
    VertexInput = 0x00000004u32,
    VertexShader = 0x00000008,
    TessellationControlShader = 0x00000010,
    TessellationEvaluationShader = 0x00000020,
    GeometryShader = 0x00000040,
    FragmentShader = 0x00000080,
    EarlyFragmentTests = 0x00000100,
    LateFragmentTests = 0x00000200,
    ColorAttachmentOutput = 0x00000400,
    ComputeShader = 0x00000800,
    Transfer = 0x00001000,
    BottomOfPipe = 0x00002000,
    Host = 0x00004000,
    AllGraphics = 0x00008000,
    AllCommands = 0x00010000,
    Other(u32),
}

pub struct PipelineStages {
    top_of_pipe: bool,
    draw_indirect: bool,
    vertex_input: bool,
    vertex_shader: bool,
    tessellation_control_shader: bool,
    tessellation_evaluation_shader: bool,
    geometry_shader: bool,
    fragment_shader: bool,
    early_fragment_tests: bool,
    late_fragment_tests: bool,
    color_attachment_output: bool,
    compute_shader: bool,
    transfer: bool,
    bottom_of_pipe: bool,
    host: bool,
    all_graphics: bool,
    all_commands: bool,
    synchronization_2: PipelineStagesSynchronization2KHR,
    acceleration_structure: PipelineStagesAccelerationStructureKHR,
    ray_tracing: PipelineStagesRayTracingPipelineKHR,
}

impl PipelineStages {
    pub fn from(
        stages: &[PipelineStage],
        stages_synchronization_2: Option<&[PipelineStageSynchronization2KHR]>,
        stages_acceleration_structure: Option<&[PipelineStageAccelerationStructureKHR]>,
        stages_ray_tracing: Option<&[PipelineStageRayTracingPipelineKHR]>,
    ) -> Self {
        Self {
            top_of_pipe: stages.contains(&PipelineStage::TopOfPipe),
            draw_indirect: stages.contains(&PipelineStage::DrawIndirect),
            vertex_input: stages.contains(&PipelineStage::VertexInput),
            vertex_shader: stages.contains(&PipelineStage::VertexShader),
            tessellation_control_shader: stages.contains(&PipelineStage::TessellationControlShader),
            tessellation_evaluation_shader: stages
                .contains(&PipelineStage::TessellationEvaluationShader),
            geometry_shader: stages.contains(&PipelineStage::GeometryShader),
            fragment_shader: stages.contains(&PipelineStage::GeometryShader),
            early_fragment_tests: stages.contains(&PipelineStage::EarlyFragmentTests),
            late_fragment_tests: stages.contains(&PipelineStage::LateFragmentTests),
            color_attachment_output: stages.contains(&PipelineStage::ColorAttachmentOutput),
            compute_shader: stages.contains(&PipelineStage::ComputeShader),
            transfer: stages.contains(&PipelineStage::Transfer),
            bottom_of_pipe: stages.contains(&PipelineStage::BottomOfPipe),
            host: stages.contains(&PipelineStage::Host),
            all_graphics: stages.contains(&PipelineStage::AllGraphics),
            all_commands: stages.contains(&PipelineStage::AllCommands),
            synchronization_2: match stages_synchronization_2 {
                Some(flags) => PipelineStagesSynchronization2KHR::from(flags),
                None => PipelineStagesSynchronization2KHR::empty(),
            },
            acceleration_structure: match stages_acceleration_structure {
                Some(flags) => PipelineStagesAccelerationStructureKHR::from(flags),
                None => PipelineStagesAccelerationStructureKHR::empty(),
            },
            ray_tracing: match stages_ray_tracing {
                Some(flags) => PipelineStagesRayTracingPipelineKHR::from(flags),
                None => PipelineStagesRayTracingPipelineKHR::empty(),
            },
        }
    }

    pub fn new(
        top_of_pipe: bool,
        draw_indirect: bool,
        vertex_input: bool,
        vertex_shader: bool,
        tessellation_control_shader: bool,
        tessellation_evaluation_shader: bool,
        geometry_shader: bool,
        fragment_shader: bool,
        early_fragment_tests: bool,
        late_fragment_tests: bool,
        color_attachment_output: bool,
        compute_shader: bool,
        transfer: bool,
        bottom_of_pipe: bool,
        host: bool,
        all_graphics: bool,
        all_commands: bool,
        synchronization_2: Option<PipelineStagesSynchronization2KHR>,
        acceleration_structure: Option<PipelineStagesAccelerationStructureKHR>,
        ray_tracing: Option<PipelineStagesRayTracingPipelineKHR>,
    ) -> Self {
        Self {
            top_of_pipe,
            draw_indirect,
            vertex_input,
            vertex_shader,
            tessellation_control_shader,
            tessellation_evaluation_shader,
            geometry_shader,
            fragment_shader,
            early_fragment_tests,
            late_fragment_tests,
            color_attachment_output,
            compute_shader,
            transfer,
            bottom_of_pipe,
            host,
            all_graphics,
            all_commands,
            synchronization_2: match synchronization_2 {
                Option::Some(s) => s,
                Option::None => PipelineStagesSynchronization2KHR::empty(),
            },
            acceleration_structure: match acceleration_structure {
                Option::Some(s) => s,
                Option::None => PipelineStagesAccelerationStructureKHR::empty(),
            },
            ray_tracing: match ray_tracing {
                Option::Some(rt) => rt,
                Option::None => PipelineStagesRayTracingPipelineKHR::empty(),
            },
        }
    }

    pub(crate) fn ash_flags(&self) -> ash::vk::PipelineStageFlags {
        (match self.top_of_pipe {
            true => ash::vk::PipelineStageFlags::TOP_OF_PIPE,
            false => ash::vk::PipelineStageFlags::empty(),
        }) | (match self.draw_indirect {
            true => ash::vk::PipelineStageFlags::DRAW_INDIRECT,
            false => ash::vk::PipelineStageFlags::empty(),
        }) | (match self.vertex_input {
            true => ash::vk::PipelineStageFlags::VERTEX_INPUT,
            false => ash::vk::PipelineStageFlags::empty(),
        }) | (match self.vertex_shader {
            true => ash::vk::PipelineStageFlags::VERTEX_SHADER,
            false => ash::vk::PipelineStageFlags::empty(),
        }) | (match self.tessellation_control_shader {
            true => ash::vk::PipelineStageFlags::TESSELLATION_CONTROL_SHADER,
            false => ash::vk::PipelineStageFlags::empty(),
        }) | (match self.tessellation_evaluation_shader {
            true => ash::vk::PipelineStageFlags::TESSELLATION_EVALUATION_SHADER,
            false => ash::vk::PipelineStageFlags::empty(),
        }) | (match self.geometry_shader {
            true => ash::vk::PipelineStageFlags::GEOMETRY_SHADER,
            false => ash::vk::PipelineStageFlags::empty(),
        }) | (match self.fragment_shader {
            true => ash::vk::PipelineStageFlags::FRAGMENT_SHADER,
            false => ash::vk::PipelineStageFlags::empty(),
        }) | (match self.early_fragment_tests {
            true => ash::vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
            false => ash::vk::PipelineStageFlags::empty(),
        }) | (match self.late_fragment_tests {
            true => ash::vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
            false => ash::vk::PipelineStageFlags::empty(),
        }) | (match self.color_attachment_output {
            true => ash::vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            false => ash::vk::PipelineStageFlags::empty(),
        }) | (match self.compute_shader {
            true => ash::vk::PipelineStageFlags::COMPUTE_SHADER,
            false => ash::vk::PipelineStageFlags::empty(),
        }) | (match self.transfer {
            true => ash::vk::PipelineStageFlags::TRANSFER,
            false => ash::vk::PipelineStageFlags::empty(),
        }) | (match self.bottom_of_pipe {
            true => ash::vk::PipelineStageFlags::BOTTOM_OF_PIPE,
            false => ash::vk::PipelineStageFlags::empty(),
        }) | (match self.host {
            true => ash::vk::PipelineStageFlags::HOST,
            false => ash::vk::PipelineStageFlags::empty(),
        }) | (match self.host {
            true => ash::vk::PipelineStageFlags::HOST,
            false => ash::vk::PipelineStageFlags::empty(),
        }) | (match self.all_graphics {
            true => ash::vk::PipelineStageFlags::ALL_GRAPHICS,
            false => ash::vk::PipelineStageFlags::empty(),
        }) | (match self.all_commands {
            true => ash::vk::PipelineStageFlags::ALL_COMMANDS,
            false => ash::vk::PipelineStageFlags::empty(),
        }) | self.acceleration_structure.ash_flags()
            | self.ray_tracing.ash_flags()
            | self.synchronization_2.ash_flags()
    }
}

use std::{sync::{
    Arc, Mutex,
}, hash::Hash, collections::HashSet};

use ash::vk::Handle;

use crate::{
    command_pool::{CommandPool, CommandPoolOwned},
    device::DeviceOwned,
    instance::InstanceOwned,
    pipeline_layout::PipelineLayout,
    prelude::{VulkanError, VulkanResult}, image::{Image, ImageTrait, ImageLayout}, queue_family::QueueFamily,
};
use crate::{
    compute_pipeline::ComputePipeline, descriptor_set::DescriptorSet, device::Device,
    queue_family::QueueFamilyOwned,
};

enum CommandBufferReferencedResource {
    ComputePipeline(Arc<ComputePipeline>),
    DescriptorSet(Arc<DescriptorSet>),
    PipelineLayout(Arc<PipelineLayout>),
}

impl Eq for CommandBufferReferencedResource {}

impl CommandBufferReferencedResource {
    pub fn hash(&self) -> u128 {
        match self {
            Self::ComputePipeline(l0) => (0b0000u128 << 124u128)  | (l0.native_handle() as u128),
            Self::DescriptorSet(l0) => (0b0001u128 << 124u128)  | (l0.native_handle() as u128),
            Self::PipelineLayout(l0) => (0b0010u128 << 124u128)  | (l0.native_handle() as u128),
        }
    }
}

impl PartialEq for CommandBufferReferencedResource {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::ComputePipeline(l0), Self::ComputePipeline(r0)) => l0.native_handle() == r0.native_handle(),
            (Self::DescriptorSet(l0), Self::DescriptorSet(r0)) => l0.native_handle() == r0.native_handle(),
            (Self::PipelineLayout(l0), Self::PipelineLayout(r0)) => l0.native_handle() == r0.native_handle(),
            _ => false,
        }
    }
}

impl Hash for CommandBufferReferencedResource {
    fn hash<H: /*~const*/ std::hash::Hasher>(&self, state: &mut H) {
        state.write_u128(self.hash())
    }
}

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
            acceleration_structure_build: false
        }
    }

    pub fn from(flags: &[PipelineStageAccelerationStructureKHR]) -> Self {
        Self {
            acceleration_structure_build: flags.contains(&PipelineStageAccelerationStructureKHR::AccelerationStructureBuild)
        }
    }

    pub fn new(
        acceleration_structure_build: bool,
    ) -> Self {
        Self {
            acceleration_structure_build
        }
    }

    pub(crate) fn ash_flags(&self) -> ash::vk::PipelineStageFlags {
        match self.acceleration_structure_build {
            true => ash::vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
            false => ash::vk::PipelineStageFlags::empty()
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
            ray_tracing_shader: false
        }
    }

    pub fn from(flags: &[PipelineStageRayTracingPipelineKHR]) -> Self {
        Self {
            ray_tracing_shader: flags.contains(&PipelineStageRayTracingPipelineKHR::RayTracingShader)
        }
    }

    pub fn new(
        ray_tracing_shader: bool,
    ) -> Self {
        Self {
            ray_tracing_shader
        }
    }

    pub(crate) fn ash_flags(&self) -> ash::vk::PipelineStageFlags {
        match self.ray_tracing_shader {
            true => ash::vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
            false => ash::vk::PipelineStageFlags::empty()
        }
    }
}

#[repr(u32)]
#[derive(PartialEq, Copy, Clone)]
pub enum PipelineStageSynchronization2KHR {
    None = 0x00000000u32,
}

pub struct PipelineStagesSynchronization2KHR {
    synchronization_2: bool
}

impl PipelineStagesSynchronization2KHR {
    pub fn empty() -> Self {
        Self {
            synchronization_2: false
        }
    }

    pub fn from(flags: &[PipelineStageSynchronization2KHR]) -> Self {
        Self {
            synchronization_2: flags.contains(&PipelineStageSynchronization2KHR::None)
        }
    }

    pub fn new(
        synchronization_2: bool,
    ) -> Self {
        Self {
            synchronization_2
        }
    }

    pub(crate) fn ash_flags(&self) -> ash::vk::PipelineStageFlags {
        match self.synchronization_2 {
            true => ash::vk::PipelineStageFlags::NONE,
            false => ash::vk::PipelineStageFlags::empty()
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
    Other(u32)
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
            tessellation_evaluation_shader: stages.contains(&PipelineStage::TessellationEvaluationShader),
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
                Option::None => PipelineStagesRayTracingPipelineKHR::empty()
            },
        }
    }

    pub(crate) fn ash_flags(&self) -> ash::vk::PipelineStageFlags {
        (match self.top_of_pipe {
            true => ash::vk::PipelineStageFlags::TOP_OF_PIPE,
            false => ash::vk::PipelineStageFlags::empty(),
        }) |
        (match self.draw_indirect {
            true => ash::vk::PipelineStageFlags::DRAW_INDIRECT,
            false => ash::vk::PipelineStageFlags::empty(),
        }) |
        (match self.vertex_input {
            true => ash::vk::PipelineStageFlags::VERTEX_INPUT,
            false => ash::vk::PipelineStageFlags::empty(),
        }) |
        (match self.vertex_shader {
            true => ash::vk::PipelineStageFlags::VERTEX_SHADER,
            false => ash::vk::PipelineStageFlags::empty(),
        }) |
        (match self.tessellation_control_shader {
            true => ash::vk::PipelineStageFlags::TESSELLATION_CONTROL_SHADER,
            false => ash::vk::PipelineStageFlags::empty(),
        }) |
        (match self.tessellation_evaluation_shader {
            true => ash::vk::PipelineStageFlags::TESSELLATION_EVALUATION_SHADER,
            false => ash::vk::PipelineStageFlags::empty(),
        }) |
        (match self.geometry_shader {
            true => ash::vk::PipelineStageFlags::GEOMETRY_SHADER,
            false => ash::vk::PipelineStageFlags::empty(),
        }) |
        (match self.fragment_shader {
            true => ash::vk::PipelineStageFlags::FRAGMENT_SHADER,
            false => ash::vk::PipelineStageFlags::empty(),
        }) |
        (match self.early_fragment_tests {
            true => ash::vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
            false => ash::vk::PipelineStageFlags::empty(),
        }) |
        (match self.late_fragment_tests {
            true => ash::vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
            false => ash::vk::PipelineStageFlags::empty(),
        }) |
        (match self.color_attachment_output {
            true => ash::vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            false => ash::vk::PipelineStageFlags::empty(),
        }) |
        (match self.compute_shader {
            true => ash::vk::PipelineStageFlags::COMPUTE_SHADER,
            false => ash::vk::PipelineStageFlags::empty(),
        }) |
        (match self.transfer {
            true => ash::vk::PipelineStageFlags::TRANSFER,
            false => ash::vk::PipelineStageFlags::empty(),
        }) |
        (match self.bottom_of_pipe {
            true => ash::vk::PipelineStageFlags::BOTTOM_OF_PIPE,
            false => ash::vk::PipelineStageFlags::empty(),
        }) |
        (match self.host {
            true => ash::vk::PipelineStageFlags::HOST,
            false => ash::vk::PipelineStageFlags::empty(),
        }) |
        (match self.host {
            true => ash::vk::PipelineStageFlags::HOST,
            false => ash::vk::PipelineStageFlags::empty(),
        }) |
        (match self.all_graphics {
            true => ash::vk::PipelineStageFlags::ALL_GRAPHICS,
            false => ash::vk::PipelineStageFlags::empty(),
        }) |
        (match self.all_commands {
            true => ash::vk::PipelineStageFlags::ALL_COMMANDS,
            false => ash::vk::PipelineStageFlags::empty(),
        }) |
        self.acceleration_structure.ash_flags() |
        self.ray_tracing.ash_flags() |
        self.synchronization_2.ash_flags()
    }
}

pub struct ImageMemoryBarrier {
    src_stages: PipelineStages,
    dst_stages: PipelineStages,
    image: Arc<dyn ImageTrait>,
    old_layout: ImageLayout,
    new_layout: ImageLayout,
    src_queue_family: Arc<QueueFamily>,
    dst_queue_family: Arc<QueueFamily>,
}

impl ImageMemoryBarrier {
    pub(crate) fn ash_src_access_mask_flags(&self) -> ash::vk::AccessFlags {
        ash::vk::AccessFlags::NONE
    }

    pub(crate) fn ash_dst_access_mask_flags(&self) -> ash::vk::AccessFlags {
        ash::vk::AccessFlags::SHADER_WRITE
    }

    pub(crate) fn ash_src_queue_family(&self) -> u32 {
        self.src_queue_family.get_family_index()
    }

    pub(crate) fn ash_dst_queue_family(&self) -> u32 {
        self.dst_queue_family.get_family_index()
    }

    pub(crate) fn ash_image_handle(&self) -> ash::vk::Image {
        ash::vk::Image::from_raw(self.image.native_handle())
    }

    pub(crate) fn ash_src_flags(&self) -> ash::vk::PipelineStageFlags {
        self.src_stages.ash_flags()
    }

    pub(crate) fn ash_dst_flags(&self) -> ash::vk::PipelineStageFlags {
        self.dst_stages.ash_flags()
    }

    pub fn new(
        src_stages: PipelineStages,
        dst_stages: PipelineStages,
        image: Arc<dyn ImageTrait>,
        old_layout: ImageLayout,
        new_layout: ImageLayout,
        src_queue_family: Arc<QueueFamily>,
        dst_queue_family: Arc<QueueFamily>,
    ) -> Self {
        Self {
            src_stages,
            dst_stages,
            image,
            old_layout,
            new_layout,
            src_queue_family,
            dst_queue_family,
        }
    }
}

pub struct CommandBufferRecorder<'a> {
    device: Arc<Device>, // this field is repeated to speed-up execution, otherwise a ton of Arc<>.clone() will be performed
    command_buffer: &'a dyn CommandBufferCrateTrait,

    used_resources: HashSet<CommandBufferReferencedResource>,
}

impl<'a> CommandBufferRecorder<'a> {
    pub fn image_barrier(&mut self, dependency_info: ImageMemoryBarrier) {
        // TODO: check every resource is from the same device

        let image_memory_barrier = ash::vk::ImageMemoryBarrier::builder()
            .image(dependency_info.ash_image_handle())
            .old_layout(dependency_info.old_layout.ash_layout())
            .new_layout(dependency_info.new_layout.ash_layout())
            .src_queue_family_index(dependency_info.ash_src_queue_family())
            .dst_queue_family_index(dependency_info.ash_dst_queue_family())
            .src_access_mask(dependency_info.ash_src_access_mask_flags())
            .dst_access_mask(dependency_info.ash_dst_access_mask_flags())
            .build();
        

        unsafe {
            self.device.ash_handle().cmd_pipeline_barrier(
                self.command_buffer.ash_handle(),
                dependency_info.ash_src_flags(),
                dependency_info.ash_dst_flags(),
                ash::vk::DependencyFlags::empty(),
                &[],
                &[],
                &[image_memory_barrier],
            );
        }
    }

    pub fn bind_compute_pipeline(&mut self, compute_pipeline: Arc<ComputePipeline>) {
        unsafe {
            self.device.ash_handle().cmd_bind_pipeline(
                self.command_buffer.ash_handle(),
                ash::vk::PipelineBindPoint::COMPUTE,
                compute_pipeline.ash_handle(),
            )
        }

        self.used_resources.insert(CommandBufferReferencedResource::ComputePipeline(compute_pipeline));
            //.register_compute_pipeline_usage(compute_pipeline)
    }

    pub fn bind_descriptor_sets(
        &mut self,
        pipeline_layout: Arc<PipelineLayout>,
        offset: u32,
        descriptor_sets: &[Arc<DescriptorSet>],
    ) {
        let mut sets = Vec::<ash::vk::DescriptorSet>::new();

        for ds in descriptor_sets.iter() {
            self.used_resources.insert(CommandBufferReferencedResource::DescriptorSet(ds.clone()));

            sets.push(ds.ash_handle());
        }

        unsafe {
            self.device.ash_handle().cmd_bind_descriptor_sets(
                self.command_buffer.ash_handle(),
                ash::vk::PipelineBindPoint::COMPUTE,
                pipeline_layout.ash_handle(),
                offset,
                sets.as_slice(),
                &[],
            )
        }

        self.used_resources.insert(CommandBufferReferencedResource::PipelineLayout(pipeline_layout));
    }

    pub fn push_constant_for_compute_shader(
        &mut self,
        pipeline_layout: Arc<PipelineLayout>,
        offset: u32,
        data: &[u8],
    ) {
        unsafe {
            self.device.ash_handle().cmd_push_constants(
                self.command_buffer.ash_handle(),
                pipeline_layout.ash_handle(),
                ash::vk::ShaderStageFlags::COMPUTE,
                offset,
                data,
            )
        }

        self.used_resources.insert(CommandBufferReferencedResource::PipelineLayout(pipeline_layout));
    }

    pub fn dispatch(&mut self, group_count_x: u32, group_count_y: u32, group_count_z: u32) {
        unsafe {
            self.device.ash_handle().cmd_dispatch(
                self.command_buffer.ash_handle(),
                group_count_x,
                group_count_y,
                group_count_z,
            )
        }
    }
}

pub trait CommandBufferTrait: CommandPoolOwned {
    fn native_handle(&self) -> u64;

    fn flag_execution_as_finished(&self);
}

pub(crate) trait CommandBufferCrateTrait: CommandBufferTrait {
    fn ash_handle(&self) -> ash::vk::CommandBuffer;
}

pub struct PrimaryCommandBuffer {
    command_pool: Arc<CommandPool>,
    command_buffer: ash::vk::CommandBuffer,
    resources_in_use: Mutex<HashSet<CommandBufferReferencedResource>>,
}

impl Drop for PrimaryCommandBuffer {
    fn drop(&mut self) {
        // Command buffers will be automatically freed when their command pool is destroyed, so we don't need explicit cleanup.
    }
}

impl CommandPoolOwned for PrimaryCommandBuffer {
    fn get_parent_command_pool(&self) -> Arc<CommandPool> {
        self.command_pool.clone()
    }
}

impl CommandBufferTrait for PrimaryCommandBuffer {
    fn native_handle(&self) -> u64 {
        ash::vk::Handle::as_raw(self.command_buffer)
    }

    fn flag_execution_as_finished(&self) {
        // TODO: if the record was a one-time-submit clear the list of used resources
        //self.recording_status.store(0, Ordering::Release);
    }
}

impl CommandBufferCrateTrait for PrimaryCommandBuffer {
    fn ash_handle(&self) -> ash::vk::CommandBuffer {
        self.command_buffer
    }
}

impl PrimaryCommandBuffer {
    pub fn record_commands<F>(&self, commands_writer_fn: F) -> VulkanResult<()>
    where
        F: Fn(&mut CommandBufferRecorder) + Sized,
    {
        let device = self
            .get_parent_command_pool()
            .get_parent_queue_family()
            .get_parent_device();

        match self.resources_in_use.lock() {
            Ok(mut resources_lck) => {
                let begin_info = ash::vk::CommandBufferBeginInfo::builder()
                    .flags(ash::vk::CommandBufferUsageFlags::empty() /*ash::vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT*/)
                    .build();

                match unsafe {
                    device
                        .ash_handle()
                        .begin_command_buffer(self.ash_handle(), &begin_info)
                } {
                    Ok(_) => {
                        let mut recorder = CommandBufferRecorder {
                            device: device.clone(),
                            command_buffer: self,
                            used_resources: HashSet::new(),
                        };

                        commands_writer_fn(&mut recorder);

                        match unsafe { device.ash_handle().end_command_buffer(self.ash_handle()) } {
                            Ok(()) => {
                                *resources_lck = recorder.used_resources;

                                Ok(())
                            }
                            Err(_err) => {
                                #[cfg(debug_assertions)]
                                {
                                    panic!("Error creating the command buffer recorder: the command buffer already is in recording state!")
                                }

                                Err(VulkanError::Unspecified)
                            }
                        }
                    }
                    Err(err) => {
                        // TODO: the command buffer is in the previous state... A good one unless the error is DEVICE_LOST

                        #[cfg(debug_assertions)]
                        {
                            panic!("Error opening the command buffer for writing: {}", err);
                        }

                        Err(VulkanError::Unspecified)
                    }
                }
            },
            Err(err) => {
                #[cfg(debug_assertions)]
                {
                    panic!("Error opening the command buffer for writing: Error acquiring mutex: {}", err);
                }

                Err(VulkanError::Unspecified)
            }
        }
    }

    pub fn new(
        command_pool: Arc<CommandPool>,
        debug_name: Option<&str>,
    ) -> VulkanResult<Arc<Self>> {
        let device = command_pool.get_parent_queue_family().get_parent_device();

        let create_info = ash::vk::CommandBufferAllocateInfo::builder()
            .command_pool(command_pool.ash_handle())
            .level(ash::vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1)
            .build();

        match unsafe { device.ash_handle().allocate_command_buffers(&create_info) } {
            Ok(command_buffers) => {
                let command_buffer = command_buffers[0];

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
                                .object_type(ash::vk::ObjectType::COMMAND_BUFFER)
                                .object_handle(ash::vk::Handle::as_raw(command_buffer))
                                .object_name(object_name)
                                .build();

                            match ext.set_debug_utils_object_name(
                                device.ash_handle().handle(),
                                &dbg_info,
                            ) {
                                Ok(_) => {
                                    #[cfg(debug_assertions)]
                                    {
                                        println!("Command Pool Debug object name changed");
                                    }
                                }
                                Err(err) => {
                                    #[cfg(debug_assertions)]
                                    {
                                        panic!("Error setting the Debug name for the newly created Command Pool, will use handle. Error: {}", err)
                                    }
                                }
                            }
                        }
                    }
                }

                Ok(Arc::new(Self {
                    command_buffer,
                    command_pool,
                    resources_in_use: Mutex::new(HashSet::new()),
                }))
            }
            Err(err) => {
                #[cfg(debug_assertions)]
                {
                    panic!("Error creating the command buffer: {}", err)
                }

                Err(VulkanError::Unspecified)
            }
        }
    }
}

use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

use crate::{queue_family::QueueFamilyOwned, compute_pipeline::ComputePipeline, device::Device, pipeline_layout::{PipelineLayout, PipelineLayoutDependant}, descriptor_set::DescriptorSet};
use crate::{
    command_pool::{CommandPool, CommandPoolOwned},
    device::DeviceOwned,
    instance::InstanceOwned,
    prelude::{VulkanError, VulkanResult},
};

// TODO: it would be better for performance to use smallvec...
pub struct ResourcesInUseByGPU {
    layouts: Vec<Arc<PipelineLayout>>,
    compute_pipelines: Vec<Arc<ComputePipeline>>,
    descriptor_sets: Vec<Arc<DescriptorSet>>,
}

impl ResourcesInUseByGPU {
    pub fn create() -> Self {
        Self {
            layouts: vec!(),
            compute_pipelines: vec!(),
            descriptor_sets: vec!(),
        }
    }
}

pub struct CommandBufferRecorder<'a> {
    device: Arc<Device>, // this field is repeated to speed-up execution, otherwise a ton of Arc<>.clone() will be performed
    command_buffer: &'a dyn CommandBufferCrateTrait,

    used_resources: ResourcesInUseByGPU
}

impl<'a> CommandBufferRecorder<'a> {
    pub fn native_handle(&self) -> u64 {
        ash::vk::Handle::as_raw(self.command_buffer.ash_handle())
    }

    pub fn use_compute_pipeline(
        &mut self,
        compute_pipeline: Arc<ComputePipeline>,
        descriptor_sets: &[Arc<DescriptorSet>],
    ) {
        unsafe {
            self.device.ash_handle().cmd_bind_pipeline(
                self.command_buffer.ash_handle(), 
                ash::vk::PipelineBindPoint::COMPUTE, 
                compute_pipeline.ash_handle()
            ) 
        }

        let mut sets = Vec::<ash::vk::DescriptorSet>::new();
        let mut dynamic_offsets = Vec::<u32>::new();

        for ds in descriptor_sets.iter() {
            self.used_resources.descriptor_sets.push(ds.clone());
            sets.push(ds.ash_handle());
            dynamic_offsets.push(0)
        }

        unsafe {
            self.device.ash_handle().cmd_bind_descriptor_sets(
                self.command_buffer.ash_handle(),
                ash::vk::PipelineBindPoint::COMPUTE,
                compute_pipeline.get_parent_pipeline_layout().ash_handle(),
                0,
                sets.as_slice(),
                dynamic_offsets.as_slice()
            )
        }

        self.used_resources.compute_pipelines.push(compute_pipeline)
    }

}

pub struct OneTimeSubmittablePrimaryCommandBuffer {
    command_buffer: Arc<PrimaryCommandBuffer>,
    status_registered: AtomicBool,
}

impl Drop for OneTimeSubmittablePrimaryCommandBuffer {
    fn drop(&mut self) {
        match self.command_buffer.recording_status.compare_exchange(
            true,
            false,
            Ordering::Acquire,
            Ordering::Relaxed,
        ) {
            Ok(_) => {}
            Err(_err) => {
                #[cfg(debug_assertions)]
                {
                    panic!("Error removing the command buffer recorder. In release mode this will lead to an unusable command buffer as it won't be able to record any more command.");
                }
            }
        }
    }
}

impl OneTimeSubmittablePrimaryCommandBuffer {
    pub fn submit(&self) -> VulkanResult<()> {

        todo!()
    }

    pub fn new(command_buffer: Arc<PrimaryCommandBuffer>) -> VulkanResult<Arc<Self>> {
        match command_buffer.recording_status.compare_exchange(
            false,
            true,
            Ordering::Acquire,
            Ordering::Relaxed,
        ) {
            Ok(_) => Ok(Arc::new(Self {
                command_buffer,
                status_registered: AtomicBool::new(false),
            })),
            Err(_err) => {
                #[cfg(debug_assertions)]
                {
                    panic!("Error removing the command buffer recorder. In release mode this will lead to an unusable command buffer as it won't be able to record any more command.");
                }
            }
        }
    }

    pub fn begin_commands(&self) -> VulkanResult<CommandBufferRecorder> {
        let device = self
            .command_buffer
            .get_parent_command_pool()
            .get_parent_queue_family()
            .get_parent_device();

        let begin_info = ash::vk::CommandBufferBeginInfo::builder()
            .flags(ash::vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)
            .build();

        match self.status_registered.compare_exchange(
            false,
            true,
            Ordering::Acquire,
            Ordering::Acquire,
        ) {
            Ok(_) => match unsafe {
                device
                            .ash_handle()
                            .begin_command_buffer(self.command_buffer.ash_handle(), &begin_info)
            } {
                Ok(()) => {
                    Ok(
                        CommandBufferRecorder {
                            device,
                            command_buffer: self.command_buffer.as_ref(),
                            used_resources: ResourcesInUseByGPU::create(),
                        }
                    )
                }
                Err(err) => {
                    #[cfg(debug_assertions)]
                    {
                        panic!("Error creating the command buffer recorder: {}", err)
                    }

                    Err(VulkanError::Unspecified)
                }
            },
            Err(_err) => {
                #[cfg(debug_assertions)]
                {
                    panic!("Error creating the command buffer recorder: the command buffer already is in recording state!")
                }

                Err(VulkanError::Unspecified)
            }
        }
    }

    pub fn end_commands<'a>(&self, recorder: CommandBufferRecorder<'a>) -> VulkanResult<()> {
        #[cfg(debug_assertions)]
        {
            assert_eq!(recorder.command_buffer.native_handle(), self.command_buffer.native_handle())
        }

        let device = self
            .command_buffer
            .get_parent_command_pool()
            .get_parent_queue_family()
            .get_parent_device();

        match self.status_registered.compare_exchange(
            true,
            false,
            Ordering::Acquire,
            Ordering::Acquire,
        ) {
            Ok(_) => match unsafe {
                device
                    .ash_handle()
                    .end_command_buffer(self.command_buffer.ash_handle())
            } {
                Ok(()) => Ok(()),
                Err(_err) => {
                    #[cfg(debug_assertions)]
                    {
                        panic!("Error creating the command buffer recorder: the command buffer already is in recording state!")
                    }

                    Err(VulkanError::Unspecified)
                }
            },
            Err(_) => {
                #[cfg(debug_assertions)]
                {
                    panic!("Error creating the command buffer recorder: the command buffer already is in recording state!")
                }

                Err(VulkanError::Unspecified)
            }
        }
    }
}

pub trait CommandBufferTrait: CommandPoolOwned {
    fn native_handle(&self) -> u64;
}

pub(crate) trait CommandBufferCrateTrait: CommandBufferTrait {
    fn ash_handle(&self) -> ash::vk::CommandBuffer;
}

pub struct PrimaryCommandBuffer {
    command_pool: Arc<CommandPool>,
    command_buffer: ash::vk::CommandBuffer,
    recording_status: AtomicBool,
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
}

impl CommandBufferCrateTrait for PrimaryCommandBuffer {
    fn ash_handle(&self) -> ash::vk::CommandBuffer {
        self.command_buffer
    }
}

impl PrimaryCommandBuffer {
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
                match device.get_parent_instance().get_debug_ext_extension() {
                    Some(ext) => {
                        match debug_name {
                            Some(name) => {
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
                            None => {}
                        };
                    }
                    None => {}
                }

                Ok(Arc::new(Self {
                    command_buffer,
                    command_pool,
                    recording_status: AtomicBool::new(false),
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

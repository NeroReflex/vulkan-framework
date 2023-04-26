use std::sync::{
    atomic::{AtomicU8, Ordering},
    Arc,
};

use crate::{
    command_pool::{CommandPool, CommandPoolOwned},
    device::DeviceOwned,
    instance::InstanceOwned,
    prelude::{VulkanError, VulkanResult},
};
use crate::{
    compute_pipeline::ComputePipeline, descriptor_set::DescriptorSet, device::Device,
    pipeline_layout::PipelineLayoutDependant, queue_family::QueueFamilyOwned,
};

// TODO: it would be better for performance to use smallvec...
pub struct ResourcesInUseByGPU {
    //layouts: Vec<Arc<PipelineLayout>>,
    compute_pipelines: Vec<Arc<ComputePipeline>>,
    descriptor_sets: Vec<Arc<DescriptorSet>>,
}

impl ResourcesInUseByGPU {
    pub fn create() -> Self {
        Self {
            //layouts: vec![],
            compute_pipelines: vec![],
            descriptor_sets: vec![],
        }
    }
}

pub struct CommandBufferRecorder<'a> {
    device: Arc<Device>, // this field is repeated to speed-up execution, otherwise a ton of Arc<>.clone() will be performed
    command_buffer: &'a dyn CommandBufferCrateTrait,

    used_resources: ResourcesInUseByGPU,
}

impl<'a> CommandBufferRecorder<'a> {
    pub fn use_compute_pipeline(
        &mut self,
        compute_pipeline: Arc<ComputePipeline>,
        descriptor_sets: &[Arc<DescriptorSet>],
    ) {
        unsafe {
            self.device.ash_handle().cmd_bind_pipeline(
                self.command_buffer.ash_handle(),
                ash::vk::PipelineBindPoint::COMPUTE,
                compute_pipeline.ash_handle(),
            )
        }

        let mut sets = Vec::<ash::vk::DescriptorSet>::new();

        for ds in descriptor_sets.iter() {
            self.used_resources.descriptor_sets.push(ds.clone());
            sets.push(ds.ash_handle());
        }

        unsafe {
            self.device.ash_handle().cmd_bind_descriptor_sets(
                self.command_buffer.ash_handle(),
                ash::vk::PipelineBindPoint::COMPUTE,
                compute_pipeline.get_parent_pipeline_layout().ash_handle(),
                0,
                sets.as_slice(),
                &[],
            )
        }

        self.used_resources.compute_pipelines.push(compute_pipeline)
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
    recording_status: AtomicU8,
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
        // store a zero so that a new set of commands can be registered
        self.recording_status.store(0, Ordering::Release);
    }
}

impl CommandBufferCrateTrait for PrimaryCommandBuffer {
    fn ash_handle(&self) -> ash::vk::CommandBuffer {
        self.command_buffer
    }
}

impl PrimaryCommandBuffer {
    pub fn record_commands<F>(&self, commands_writer_fn: F) -> VulkanResult<ResourcesInUseByGPU>
    where
        F: Fn(&mut CommandBufferRecorder) + Sized,
    {
        match self
            .recording_status
            .compare_exchange(0, 1, Ordering::Acquire, Ordering::Relaxed)
        {
            Ok(_) => {
                let device = self
                    .get_parent_command_pool()
                    .get_parent_queue_family()
                    .get_parent_device();

                let begin_info = ash::vk::CommandBufferBeginInfo::builder()
                    .flags(ash::vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)
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
                            used_resources: ResourcesInUseByGPU::create(),
                        };

                        commands_writer_fn(&mut recorder);

                        match unsafe { device.ash_handle().end_command_buffer(self.ash_handle()) } {
                            Ok(()) => {
                                self.recording_status.store(2, Ordering::Release);

                                Ok(recorder.used_resources)
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
                        self.recording_status.store(0, Ordering::Release);

                        #[cfg(debug_assertions)]
                        {
                            panic!("Error opening the command buffer for writing: {}", err);
                        }

                        Err(VulkanError::Unspecified)
                    }
                }
            }
            Err(_) => {
                #[cfg(debug_assertions)]
                {
                    panic!("Error creating the command buffer recorder: a record operation is already in progress.");
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
                    recording_status: AtomicU8::new(0),
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

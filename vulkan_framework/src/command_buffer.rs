use std::sync::{Arc, atomic::{AtomicBool, Ordering}};

use crate::{command_pool::{CommandPool, CommandPoolOwned, self}, prelude::{VulkanResult, VulkanError}, device::DeviceOwned, instance::InstanceOwned};
use crate::{queue_family::{QueueFamilyOwned}};

pub struct OneTimeSubmittable<'a> {
    command_buffer: &'a dyn CommandBufferCrateTrait
}

pub struct CommandBufferRecorder<'a> {
    command_buffer: &'a dyn CommandBufferCrateTrait
}

impl<'a> Drop for CommandBufferRecorder<'a> {
    fn drop(&mut self) {
        let device = self.command_buffer.get_parent_command_pool().get_parent_queue_family().get_parent_device();

        let _ = unsafe { device.ash_handle().end_command_buffer(self.command_buffer.ash_handle()) };
    }
}

impl<'a> CommandBufferRecorder<'a> {
    pub(crate) fn from_command_buffer_unchecked(command_buffer: &'a dyn CommandBufferCrateTrait) -> Self {
        Self {
            command_buffer
        }
    }
}

pub trait CommandBufferTrait : CommandPoolOwned {
    fn native_handle(&self) -> u64;

    fn register_commands(&self) -> VulkanResult<CommandBufferRecorder>;

    fn end_commands<'a>(&self, recorder: CommandBufferRecorder<'a>) -> VulkanResult<Arc<OneTimeSubmittable>>;
}

pub(crate) trait CommandBufferCrateTrait : CommandBufferTrait {
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

    fn register_commands(&self) -> VulkanResult<CommandBufferRecorder> {
        let device = self.get_parent_command_pool().get_parent_queue_family().get_parent_device();

        match self.recording_status.compare_exchange(false, true, Ordering::Acquire, Ordering::Acquire) {
            Ok(_) => {

                let begin_info = ash::vk::CommandBufferBeginInfo::builder()
                    .flags(ash::vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)
                    .build();

                match unsafe { device.ash_handle().begin_command_buffer(self.command_buffer, &begin_info) } {
                    Ok(()) => {
                        Ok(CommandBufferRecorder::from_command_buffer_unchecked(self))
                    },
                    Err(err) => {
                        #[cfg(debug_assertions)]
                        {
                            println!("Error creating the command buffer recorder: {}", err);
                            assert_eq!(true, false)
                        }

                        Err(VulkanError::Unspecified)
                    }
                }
            },
            Err(_err) => {
                #[cfg(debug_assertions)]
                {
                    println!("Error creating the command buffer recorder: the command buffer already is in recording state!");
                    assert_eq!(true, false)
                }

                Err(VulkanError::Unspecified)
            }
        }
    }

    fn end_commands<'a>(&self, recorder: CommandBufferRecorder<'a>) -> VulkanResult<Arc<OneTimeSubmittable>> {
        let device = self.get_parent_command_pool().get_parent_queue_family().get_parent_device();

        match self.recording_status.compare_exchange(true, false, Ordering::Acquire, Ordering::Acquire) {
            Ok(_) => {
                match unsafe { device.ash_handle().end_command_buffer(self.ash_handle()) } {
                    Ok(()) => {
                        Ok(Arc::new(OneTimeSubmittable { command_buffer: self }))
                    },
                    Err(err) => {
                        #[cfg(debug_assertions)]
                        {
                            println!("Error creating the command buffer recorder: the command buffer already is in recording state!");
                            assert_eq!(true, false)
                        }

                        Err(VulkanError::Unspecified)
                    }
                }
            },
            Err(_) => {
                #[cfg(debug_assertions)]
                {
                    println!("Error creating the command buffer recorder: the command buffer already is in recording state!");
                    assert_eq!(true, false)
                }

                Err(VulkanError::Unspecified)
            }
        }
    }
}

impl CommandBufferCrateTrait for PrimaryCommandBuffer {
    fn ash_handle(&self) -> ash::vk::CommandBuffer {
        self.command_buffer.clone()
    }
}

impl PrimaryCommandBuffer {
    pub fn new(command_pool: Arc<CommandPool>, debug_name: Option<&str>) -> VulkanResult<Arc<Self>> {
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
                                                println!("Error setting the Debug name for the newly created Command Pool, will use handle. Error: {}", err);
                                                assert_eq!(true, false);
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

                Ok(
                    Arc::new(
                        Self {
                            command_buffer,
                            command_pool,
                            recording_status: AtomicBool::new(false)
                        }
                    )
                )
            },
            Err(err) => {
                #[cfg(debug_assertions)]
                {
                    println!("Error creating the command buffer: {}", err);
                    assert_eq!(true, false)
                }

                Err(VulkanError::Unspecified)
            }
        }
    }
}
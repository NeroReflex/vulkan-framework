use std::sync::Arc;

use crate::{prelude::{VulkanResult, VulkanError, FrameworkError}, raytracing_pipeline::RaytracingPipeline, device::DeviceOwned, memory_pool::{MemoryPool, MemoryPoolBacked}, memory_allocator::MemoryAllocator, utils, buffer::{Buffer, ConcreteBufferDescriptor, BufferUsage}, memory_heap::{MemoryHeapOwned, MemoryType, MemoryHostVisibility}};

pub struct RaytracingBindingTables
{
    handle_size: u32,
    handle_size_aligned: u32,
    group_count: u32,
    sbt_size: u32,
    shader_handle_storage: Vec<u8>,
    raytracing_pipeline: Arc<RaytracingPipeline>,
    raygen_buffer: Arc<Buffer>,
    raygen_buffer_addr: u64,
    miss_buffer: Arc<Buffer>,
    miss_buffer_addr: u64,
    closesthit_buffer: Arc<Buffer>,
    closesthit_buffer_addr: u64,
    callable_buffer: Option<Arc<Buffer>>,
    callable_buffer_addr: Option<u64>
}

pub fn required_memory_type() -> MemoryType {
    // TODO: check memory_pool is VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    MemoryType::DeviceLocal(Some(MemoryHostVisibility::new(false)))
}

impl RaytracingBindingTables
{
    pub(crate) fn ash_callable_strided(&self) -> ash::vk::StridedDeviceAddressRegionKHR {
        let mut result = ash::vk::StridedDeviceAddressRegionKHR::builder()
            .stride(self.handle_size_aligned as u64)
            .size(self.handle_size_aligned as u64)
            .device_address(match self.callable_buffer_addr {
                Some(addr) => addr,
                None => 0,
            })
            .build();
        
        if self.callable_buffer.is_none() {
            unsafe {
                result = std::mem::transmute(vec![0; core::mem::size_of::<ash::vk::StridedDeviceAddressRegionKHR>()]);
            }
        }

        result
    }

    pub(crate) fn ash_raygen_strided(&self) -> ash::vk::StridedDeviceAddressRegionKHR {
        ash::vk::StridedDeviceAddressRegionKHR::builder()
            .stride(self.handle_size_aligned as u64)
            .size(self.handle_size_aligned as u64)
            .device_address(self.raygen_buffer_addr)
            .build()
    }

    pub(crate) fn ash_miss_strided(&self) -> ash::vk::StridedDeviceAddressRegionKHR {
        ash::vk::StridedDeviceAddressRegionKHR::builder()
            .stride(self.handle_size_aligned as u64)
            .size(self.handle_size_aligned as u64)
            .device_address(self.miss_buffer_addr)
            .build()
    }

    pub(crate) fn ash_closesthit_strided(&self) -> ash::vk::StridedDeviceAddressRegionKHR {
        ash::vk::StridedDeviceAddressRegionKHR::builder()
            .stride(self.handle_size_aligned as u64)
            .size(self.handle_size_aligned as u64)
            .device_address(self.closesthit_buffer_addr)
            .build()
    }

    /**
     * The given memory pool MUST manage a memory heap that has the same memory type as the one returned from required_memory_type()
     */
    pub fn new(
        raytracing_pipeline: Arc<RaytracingPipeline>,
        memory_pool: Arc<MemoryPool>,
    ) -> VulkanResult<Arc<Self>>
    {
        let required_memory_type = required_memory_type();
        let mem_type = memory_pool.get_parent_memory_heap().memory_type();
        assert!(mem_type == required_memory_type);

        // this feature is required for raytracing to work at all
        if !memory_pool.features().device_addressable() {
            return Err(VulkanError::Framework(FrameworkError::Unknown(Some(format!("Missing feature on MemoryPool: device_addressable need to be set")))))
        }

        let device = raytracing_pipeline.get_parent_device();

        match device.ray_tracing_info() {
            Some(rt_info) => {
                
                match device.ash_ext_raytracing_pipeline_khr() {
                    Some(rt_ext) => {
                        let handle_size = rt_info.shader_group_handle_size();
                        let handle_size_aligned = utils::aligned_size(handle_size, rt_info.shader_group_handle_alignment());
                        let group_count: u32 = raytracing_pipeline.shader_group_size();
                        
                        let sbt_size = group_count * handle_size_aligned;

                        let buffer_descriptor = ConcreteBufferDescriptor::new(
                            BufferUsage::Unmanaged((ash::vk::BufferUsageFlags::SHADER_BINDING_TABLE_KHR | ash::vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS).as_raw()),
                            handle_size as u64
                        );

                        match unsafe {
                            rt_ext.get_ray_tracing_shader_group_handles(
                                raytracing_pipeline.ash_handle(),
                                0,
                                group_count,
                                sbt_size as usize,
                            )
                        } {
                            Ok(shader_handle_storage) => {
                                let raygen_buffer = Buffer::new(
                                    memory_pool.clone(),
                                    buffer_descriptor,
                                    None,
                                    None,
                                ).unwrap();
                                memory_pool.write_raw_data(raygen_buffer.allocation_offset(), &shader_handle_storage[((handle_size_aligned * 0) as usize)..(((handle_size_aligned * 0) + handle_size) as usize)]).unwrap();
                                let raygen_buffer_addr = unsafe {
                                    let info = ash::vk::BufferDeviceAddressInfo::builder()
                                        .buffer(raygen_buffer.ash_handle());

                                    let device_addr = raytracing_pipeline.get_parent_device().ash_handle().get_buffer_device_address(&info);
                                    device_addr
                                };

                                let miss_buffer = Buffer::new(
                                    memory_pool.clone(),
                                    buffer_descriptor,
                                    None,
                                    None,
                                ).unwrap();
                                memory_pool.write_raw_data(miss_buffer.allocation_offset(), &shader_handle_storage[((handle_size_aligned * 1) as usize)..(((handle_size_aligned * 1) + handle_size) as usize)]).unwrap();
                                let miss_buffer_addr = unsafe {
                                    let info = ash::vk::BufferDeviceAddressInfo::builder()
                                        .buffer(miss_buffer.ash_handle());

                                    let device_addr = raytracing_pipeline.get_parent_device().ash_handle().get_buffer_device_address(&info);
                                    device_addr
                                };

                                let closesthit_buffer = Buffer::new(
                                    memory_pool.clone(),
                                    buffer_descriptor,
                                    None,
                                    None,
                                ).unwrap();
                                memory_pool.write_raw_data(closesthit_buffer.allocation_offset(), &shader_handle_storage[((handle_size_aligned * 2) as usize)..(((handle_size_aligned * 2) + handle_size) as usize)]).unwrap();
                                let closesthit_buffer_addr = unsafe {
                                    let info = ash::vk::BufferDeviceAddressInfo::builder()
                                        .buffer(closesthit_buffer.ash_handle());

                                    let device_addr = raytracing_pipeline.get_parent_device().ash_handle().get_buffer_device_address(&info);
                                    device_addr
                                };

                                let callable_buffer = Some(Buffer::new(
                                    memory_pool.clone(),
                                    buffer_descriptor,
                                    None,
                                    None,
                                ).unwrap());

                                let callable_buffer_addr = match &callable_buffer {
                                    Some(cb) => {
                                        memory_pool.write_raw_data(cb.allocation_offset(), &shader_handle_storage[0..1]).unwrap();
                                    
                                        unsafe {
                                            let info = ash::vk::BufferDeviceAddressInfo::builder()
                                                .buffer(cb.ash_handle());
        
                                            let device_addr = raytracing_pipeline.get_parent_device().ash_handle().get_buffer_device_address(&info);
                                            Some(device_addr)
                                        }
                                    },
                                    None => None
                                };

                                Ok(
                                    Arc::new(
                                        Self {
                                            raytracing_pipeline,
                                            handle_size,
                                            handle_size_aligned,
                                            group_count,
                                            sbt_size,
                                            shader_handle_storage,
                                            raygen_buffer,
                                            raygen_buffer_addr,
                                            miss_buffer,
                                            miss_buffer_addr,
                                            closesthit_buffer,
                                            closesthit_buffer_addr,
                                            callable_buffer,
                                            callable_buffer_addr,
                                        }
                                    )
                                )
                            },
                            Err(err) => {
                                Err(VulkanError::Vulkan(err.as_raw(), None))
                            }
                        }
                    },
                    None => Err(VulkanError::MissingExtension(String::from(
                        "VK_KHR_ray_tracing_pipeline",
                    )))
                }
            },
            None => {
                Err(VulkanError::MissingExtension(String::from("VK_KHR_ray_tracing_pipeline")))
            }
        }
    }
}

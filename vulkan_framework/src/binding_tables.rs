use std::sync::Arc;

use crate::{
    buffer::{AllocatedBuffer, Buffer, BufferTrait, BufferUsage, ConcreteBufferDescriptor},
    device::DeviceOwned,
    memory_heap::{MemoryHeapOwned, MemoryHostVisibility, MemoryType},
    memory_pool::{MemoryPool, MemoryPoolBacked},
    prelude::{FrameworkError, VulkanError, VulkanResult},
    raytracing_pipeline::RaytracingPipeline,
    utils,
};

pub struct RaytracingBindingTableCallableBuffer {
    callable_buffer: Arc<dyn BufferTrait>,
    callable_buffer_addr: u64,
}

pub struct RaytracingBindingTables {
    handle_size: u32,
    handle_size_aligned: u32,
    group_count: u32,
    sbt_size: u32,
    shader_handle_storage: Vec<u8>,
    raytracing_pipeline: Arc<RaytracingPipeline>,
    raygen_buffer: Arc<dyn BufferTrait>,
    raygen_buffer_addr: u64,
    miss_buffer: Arc<dyn BufferTrait>,
    miss_buffer_addr: u64,
    closesthit_buffer: Arc<dyn BufferTrait>,
    closesthit_buffer_addr: u64,
    callable: Option<RaytracingBindingTableCallableBuffer>,
}

pub fn required_memory_type() -> MemoryType {
    MemoryType::DeviceLocal(Some(MemoryHostVisibility::visible(false)))
}

impl RaytracingBindingTables {
    pub(crate) fn ash_callable_strided(&self) -> ash::vk::StridedDeviceAddressRegionKHR {
        let mut result = ash::vk::StridedDeviceAddressRegionKHR::default()
            .stride(self.handle_size_aligned as u64)
            .size(self.handle_size_aligned as u64)
            .device_address(match &self.callable {
                Some(cb) => cb.callable_buffer_addr,
                None => 0,
            });

        if self.callable.is_none() {
            unsafe {
                result = std::mem::transmute(vec![
                    0;
                    core::mem::size_of::<
                        ash::vk::StridedDeviceAddressRegionKHR,
                    >()
                ]);
            }
        }

        result
    }

    pub(crate) fn ash_raygen_strided(&self) -> ash::vk::StridedDeviceAddressRegionKHR {
        ash::vk::StridedDeviceAddressRegionKHR::default()
            .stride(self.handle_size_aligned as u64)
            .size(self.handle_size_aligned as u64)
            .device_address(self.raygen_buffer_addr)
    }

    pub(crate) fn ash_miss_strided(&self) -> ash::vk::StridedDeviceAddressRegionKHR {
        ash::vk::StridedDeviceAddressRegionKHR::default()
            .stride(self.handle_size_aligned as u64)
            .size(self.handle_size_aligned as u64)
            .device_address(self.miss_buffer_addr)
    }

    pub(crate) fn ash_closesthit_strided(&self) -> ash::vk::StridedDeviceAddressRegionKHR {
        ash::vk::StridedDeviceAddressRegionKHR::default()
            .stride(self.handle_size_aligned as u64)
            .size(self.handle_size_aligned as u64)
            .device_address(self.closesthit_buffer_addr)
    }

    /**
     * The given memory pool MUST manage a memory heap that has the same memory type as the one returned from required_memory_type()
     */
    pub fn new(
        raytracing_pipeline: Arc<RaytracingPipeline>,
        memory_pool: Arc<MemoryPool>,
    ) -> VulkanResult<Arc<Self>> {
        let required_memory_type = required_memory_type();
        let mem_type = memory_pool.get_parent_memory_heap().memory_type();
        assert!(mem_type == required_memory_type);

        // this feature is required for raytracing to work at all
        if !memory_pool.features().device_addressable() {
            return Err(VulkanError::Framework(
                FrameworkError::MemoryPoolNotAddressable,
            ));
        }

        let device = raytracing_pipeline.get_parent_device();

        let Some(rt_info) = device.ray_tracing_info() else {
            return Err(VulkanError::MissingExtension(String::from(
                "VK_KHR_ray_tracing_pipeline",
            )));
        };

        let Some(rt_ext) = device.ash_ext_raytracing_pipeline_khr() else {
            return Err(VulkanError::MissingExtension(String::from(
                "VK_KHR_ray_tracing_pipeline",
            )));
        };

        let handle_size = rt_info.shader_group_handle_size();
        let handle_size_aligned =
            utils::aligned_size(handle_size, rt_info.shader_group_handle_alignment());
        let group_count: u32 = raytracing_pipeline.shader_group_size();

        let sbt_size = group_count * handle_size_aligned;

        let buffer_descriptor = ConcreteBufferDescriptor::new(
            BufferUsage::from(
                (ash::vk::BufferUsageFlags::SHADER_BINDING_TABLE_KHR
                    | ash::vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS)
                    .as_raw(),
            ),
            handle_size as u64,
        );

        let shader_handle_storage = unsafe {
            rt_ext.get_ray_tracing_shader_group_handles(
                raytracing_pipeline.ash_handle(),
                0,
                group_count,
                sbt_size as usize,
            )
        }
        .map_err(|err| VulkanError::Vulkan(err.as_raw(), None))?;

        let raygen_buffer = AllocatedBuffer::new(
            memory_pool.clone(),
            Buffer::new(device.clone(), buffer_descriptor, None, None)?,
        )?;
        memory_pool
            .write_raw_data(
                raygen_buffer.allocation_offset(),
                &shader_handle_storage[((handle_size_aligned * 0) as usize)
                    ..(((handle_size_aligned * 0) + handle_size) as usize)],
            )
            .unwrap();
        let raygen_buffer_addr = unsafe {
            let info =
                ash::vk::BufferDeviceAddressInfo::default().buffer(raygen_buffer.ash_handle());

            let device_addr = raytracing_pipeline
                .get_parent_device()
                .ash_handle()
                .get_buffer_device_address(&info);
            device_addr
        };

        let miss_buffer = AllocatedBuffer::new(
            memory_pool.clone(),
            Buffer::new(device.clone(), buffer_descriptor, None, None)?,
        )?;
        memory_pool
            .write_raw_data(
                miss_buffer.allocation_offset(),
                &shader_handle_storage[(handle_size_aligned as usize)
                    ..((handle_size_aligned + handle_size) as usize)],
            )
            .unwrap();
        let miss_buffer_addr = unsafe {
            let info = ash::vk::BufferDeviceAddressInfo::default().buffer(miss_buffer.ash_handle());

            let device_addr = raytracing_pipeline
                .get_parent_device()
                .ash_handle()
                .get_buffer_device_address(&info);
            device_addr
        };

        let closesthit_buffer = AllocatedBuffer::new(
            memory_pool.clone(),
            Buffer::new(device.clone(), buffer_descriptor, None, None)?,
        )?;
        memory_pool
            .write_raw_data(
                closesthit_buffer.allocation_offset(),
                &shader_handle_storage[((handle_size_aligned * 2) as usize)
                    ..(((handle_size_aligned * 2) + handle_size) as usize)],
            )
            .unwrap();
        let closesthit_buffer_addr = unsafe {
            let info =
                ash::vk::BufferDeviceAddressInfo::default().buffer(closesthit_buffer.ash_handle());

            let device_addr = raytracing_pipeline
                .get_parent_device()
                .ash_handle()
                .get_buffer_device_address(&info);
            device_addr
        };

        let callable = Some({
            let callable_buffer = AllocatedBuffer::new(
                memory_pool.clone(),
                Buffer::new(device.clone(), buffer_descriptor, None, None)?,
            )?;

            memory_pool
                .write_raw_data(
                    callable_buffer.allocation_offset(),
                    &shader_handle_storage[0..1],
                )
                .unwrap();

            let info =
                ash::vk::BufferDeviceAddressInfo::default().buffer(callable_buffer.ash_handle());

            let callable_buffer_addr = unsafe {
                raytracing_pipeline
                    .get_parent_device()
                    .ash_handle()
                    .get_buffer_device_address(&info)
            };

            RaytracingBindingTableCallableBuffer {
                callable_buffer,
                callable_buffer_addr,
            }
        });

        Ok(Arc::new(Self {
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
            callable,
        }))
    }
}

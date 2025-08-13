use std::sync::Arc;

use crate::{
    buffer::{Buffer, BufferTrait, BufferUsage, ConcreteBufferDescriptor},
    device::DeviceOwned,
    memory_heap::{MemoryHostVisibility, MemoryType},
    memory_management::{MemoryManagementTags, MemoryManagerTrait},
    memory_pool::{MemoryMap, MemoryPoolBacked, MemoryPoolFeatures},
    prelude::{VulkanError, VulkanResult},
    raytracing_pipeline::RaytracingPipeline,
    utils,
};

pub struct RaytracingBindingTableCallableBuffer {
    _callable_buffer: Arc<dyn BufferTrait>,
    callable_buffer_addr: u64,
}

pub struct RaytracingBindingTables {
    handle_size: u32,
    handle_size_aligned: u32,
    group_count: u32,
    _shader_handle_storage: Vec<u8>,
    raytracing_pipeline: Arc<RaytracingPipeline>,
    _raygen_buffer: Arc<dyn BufferTrait>,
    raygen_buffer_addr: u64,
    _miss_buffer: Arc<dyn BufferTrait>,
    miss_buffer_addr: u64,
    _closesthit_buffer: Arc<dyn BufferTrait>,
    closesthit_buffer_addr: u64,
    callable: Option<RaytracingBindingTableCallableBuffer>,
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
        memory_manager: &mut dyn MemoryManagerTrait,
        allocation_tags: MemoryManagementTags,
    ) -> VulkanResult<Arc<Self>> {
        // memory type should be DeviceLocal for performance reasons, but also be host visible so that I can write to it
        let memory_type = MemoryType::device_local_and_host_visible();

        // memory pool features MUST include device addressable
        let memory_pool_features = MemoryPoolFeatures::new(true);

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
                (group_count * handle_size_aligned) as usize,
            )
        }?;

        let raygen_unallocated_buffer = Buffer::new(device.clone(), buffer_descriptor, None, None)?;
        let miss_unallocated_buffer = Buffer::new(device.clone(), buffer_descriptor, None, None)?;
        let closesthit_unallocated_buffer =
            Buffer::new(device.clone(), buffer_descriptor, None, None)?;
        let callable_unallocated_buffer =
            Buffer::new(device.clone(), buffer_descriptor, None, None)?;

        let allocation_result = memory_manager.allocate_resources(
            &memory_type,
            &memory_pool_features,
            vec![
                raygen_unallocated_buffer.into(),
                miss_unallocated_buffer.into(),
                closesthit_unallocated_buffer.into(),
                callable_unallocated_buffer.into(),
            ],
            allocation_tags,
        )?;

        assert_eq!(allocation_result.len(), 4_usize);

        let raygen_buffer = allocation_result[0].buffer();
        let miss_buffer = allocation_result[1].buffer();
        let closesthit_buffer = allocation_result[2].buffer();
        let callable_buffer = allocation_result[3].buffer();

        {
            let mem_map = MemoryMap::new(raygen_buffer.get_backing_memory_pool())?;
            let mut range =
                mem_map.range::<u8>(raygen_buffer.clone() as Arc<dyn MemoryPoolBacked>)?;
            let gpu_data = range.as_mut_slice();
            let raygen_data = &shader_handle_storage[((handle_size_aligned * 0) as usize)
                ..(((handle_size_aligned * 0) + handle_size) as usize)];
            assert_eq!(raygen_data.len(), gpu_data.len());
            gpu_data.copy_from_slice(raygen_data);
        }

        {
            let mem_map = MemoryMap::new(miss_buffer.get_backing_memory_pool())?;
            let mut range = mem_map.range::<u8>(miss_buffer.clone() as Arc<dyn MemoryPoolBacked>)?;
            let gpu_data = range.as_mut_slice();
            let miss_data = &shader_handle_storage
                [(handle_size_aligned as usize)..((handle_size_aligned + handle_size) as usize)];
            assert_eq!(miss_data.len(), gpu_data.len());

            gpu_data.copy_from_slice(miss_data);
        }

        {
            let mem_map = MemoryMap::new(closesthit_buffer.get_backing_memory_pool())?;
            let mut range =
                mem_map.range::<u8>(closesthit_buffer.clone() as Arc<dyn MemoryPoolBacked>)?;
            let gpu_data = range.as_mut_slice();
            let closesthit_data = &shader_handle_storage[((handle_size_aligned * 2) as usize)
                ..(((handle_size_aligned * 2) + handle_size) as usize)];
            assert_eq!(closesthit_data.len(), gpu_data.len());
            gpu_data.copy_from_slice(closesthit_data);
        }

        if raytracing_pipeline.callable_shader_present() {
            let mem_map = MemoryMap::new(callable_buffer.get_backing_memory_pool())?;
            let mut range =
                mem_map.range::<u8>(callable_buffer.clone() as Arc<dyn MemoryPoolBacked>)?;
            let gpu_data = range.as_mut_slice();
            let callable_buffer_data = &shader_handle_storage[((handle_size_aligned * 3) as usize)
                ..(((handle_size_aligned * 3) + handle_size) as usize)];
            assert_eq!(callable_buffer_data.len(), gpu_data.len());
            gpu_data.copy_from_slice(callable_buffer_data);
        }

        let raygen_buffer_addr = unsafe {
            let info =
                ash::vk::BufferDeviceAddressInfo::default().buffer(raygen_buffer.ash_handle());

            let device_addr = raytracing_pipeline
                .get_parent_device()
                .ash_handle()
                .get_buffer_device_address(&info);
            device_addr
        };

        let miss_buffer_addr = unsafe {
            let info = ash::vk::BufferDeviceAddressInfo::default().buffer(miss_buffer.ash_handle());

            let device_addr = raytracing_pipeline
                .get_parent_device()
                .ash_handle()
                .get_buffer_device_address(&info);
            device_addr
        };

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
            let info =
                ash::vk::BufferDeviceAddressInfo::default().buffer(callable_buffer.ash_handle());

            let callable_buffer_addr = unsafe {
                raytracing_pipeline
                    .get_parent_device()
                    .ash_handle()
                    .get_buffer_device_address(&info)
            };

            RaytracingBindingTableCallableBuffer {
                _callable_buffer: callable_buffer,
                callable_buffer_addr,
            }
        });

        Ok(Arc::new(Self {
            raytracing_pipeline,
            handle_size,
            handle_size_aligned,
            group_count,
            _shader_handle_storage: shader_handle_storage,
            _raygen_buffer: raygen_buffer,
            raygen_buffer_addr,
            _miss_buffer: miss_buffer,
            miss_buffer_addr,
            _closesthit_buffer: closesthit_buffer,
            closesthit_buffer_addr,
            callable,
        }))
    }
}

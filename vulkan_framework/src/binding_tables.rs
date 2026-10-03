use std::sync::Arc;

use crate::{
    buffer::{Buffer, BufferTrait, BufferUsage, ConcreteBufferDescriptor},
    device::{Device, DeviceOwned},
    memory_heap::MemoryType,
    memory_management::{MemoryManagementTags, MemoryManagerTrait},
    memory_pool::{MemoryMap, MemoryPoolBacked, MemoryPoolFeatures},
    prelude::{FrameworkError, VulkanError, VulkanResult},
    raytracing_pipeline::RaytracingPipeline,
};

pub struct RaytracingBindingTableCallableBuffer {
    _callable_buffer: Arc<dyn BufferTrait>,
    callable_buffer_addr: u64,
}

pub struct RaytracingBindingTables {
    _raytracing_pipeline: Arc<RaytracingPipeline>,
    _raygen_buffer: Arc<dyn BufferTrait>,
    raygen_region: ash::vk::StridedDeviceAddressRegionKHR,
    _miss_buffer: Arc<dyn BufferTrait>,
    miss_region: ash::vk::StridedDeviceAddressRegionKHR,
    _closesthit_buffer: Arc<dyn BufferTrait>,
    closesthit_region: ash::vk::StridedDeviceAddressRegionKHR,
    stride: u64,
    callable: Option<RaytracingBindingTableCallableBuffer>,
}

/// Shader handles returned by Vulkan are packed; only the destination records use stride.
#[derive(Clone, Copy)]
pub(crate) struct ShaderBindingTableLayout {
    handle_size: u64,
    stride: u64,
    base_alignment: u64,
}

impl ShaderBindingTableLayout {
    pub(crate) fn new(
        handle_size: u32,
        handle_alignment: u32,
        base_alignment: u32,
        max_stride: u32,
    ) -> VulkanResult<Self> {
        if handle_size == 0
            || !handle_alignment.is_power_of_two()
            || !base_alignment.is_power_of_two()
        {
            return Err(ash::vk::Result::ERROR_INITIALIZATION_FAILED.into());
        }
        let handle_size = u64::from(handle_size);
        let alignment = u64::from(handle_alignment);
        let stride = (handle_size + alignment - 1) & !(alignment - 1);
        if stride > u64::from(max_stride) {
            return Err(ash::vk::Result::ERROR_INITIALIZATION_FAILED.into());
        }
        Ok(Self {
            handle_size,
            stride,
            base_alignment: u64::from(base_alignment),
        })
    }

    pub(crate) fn allocation_size(&self, record_count: u32) -> VulkanResult<u64> {
        if record_count == 0 {
            return Ok(0);
        }
        self.stride
            .checked_mul(u64::from(record_count))
            .and_then(|size| size.checked_add(self.base_alignment - 1))
            .ok_or_else(|| ash::vk::Result::ERROR_INITIALIZATION_FAILED.into())
    }

    pub(crate) fn region(
        &self,
        buffer_address: u64,
        buffer_size: u64,
        record_count: u32,
    ) -> VulkanResult<(u64, ash::vk::StridedDeviceAddressRegionKHR)> {
        if record_count == 0 {
            return Ok((0, ash::vk::StridedDeviceAddressRegionKHR::default()));
        }
        let address = buffer_address
            .checked_add(self.base_alignment - 1)
            .map(|address| address & !(self.base_alignment - 1))
            .filter(|_| buffer_address != 0)
            .ok_or(ash::vk::Result::ERROR_INITIALIZATION_FAILED)?;
        let offset = address - buffer_address;
        let size = self
            .stride
            .checked_mul(u64::from(record_count))
            .ok_or(ash::vk::Result::ERROR_INITIALIZATION_FAILED)?;
        if offset.checked_add(size).is_none_or(|end| end > buffer_size)
            || address.checked_add(size).is_none()
        {
            return Err(ash::vk::Result::ERROR_INITIALIZATION_FAILED.into());
        }
        Ok((
            offset,
            ash::vk::StridedDeviceAddressRegionKHR::default()
                .device_address(address)
                .stride(self.stride)
                .size(size),
        ))
    }

    pub(crate) fn write_handles(
        &self,
        packed_handles: &[u8],
        first_group: u32,
        record_count: u32,
        destination: &mut [u8],
    ) -> VulkanResult<()> {
        let source_start = u64::from(first_group)
            .checked_mul(self.handle_size)
            .ok_or(ash::vk::Result::ERROR_INITIALIZATION_FAILED)?;
        let source_size = u64::from(record_count)
            .checked_mul(self.handle_size)
            .ok_or(ash::vk::Result::ERROR_INITIALIZATION_FAILED)?;
        let source_end = source_start
            .checked_add(source_size)
            .ok_or(ash::vk::Result::ERROR_INITIALIZATION_FAILED)?;
        let destination_size = u64::from(record_count)
            .checked_mul(self.stride)
            .ok_or(ash::vk::Result::ERROR_INITIALIZATION_FAILED)?;
        if source_end > packed_handles.len() as u64 || destination_size > destination.len() as u64 {
            return Err(ash::vk::Result::ERROR_INITIALIZATION_FAILED.into());
        }
        destination.fill(0);
        for record in 0..u64::from(record_count) {
            let source = (source_start + record * self.handle_size) as usize;
            let target = (record * self.stride) as usize;
            destination[target..target + self.handle_size as usize]
                .copy_from_slice(&packed_handles[source..source + self.handle_size as usize]);
        }
        Ok(())
    }
}

impl DeviceOwned for RaytracingBindingTables {
    fn get_parent_device(&self) -> Arc<Device> {
        self._raytracing_pipeline.get_parent_device()
    }
}

impl RaytracingBindingTables {
    pub(crate) fn ash_callable_strided(&self) -> ash::vk::StridedDeviceAddressRegionKHR {
        match &self.callable {
            Some(callable) => ash::vk::StridedDeviceAddressRegionKHR::default()
                .device_address(callable.callable_buffer_addr)
                .stride(self.stride)
                .size(self.stride),
            None => ash::vk::StridedDeviceAddressRegionKHR::default(),
        }
    }

    pub(crate) fn ash_raygen_strided(&self) -> ash::vk::StridedDeviceAddressRegionKHR {
        self.raygen_region
    }

    pub(crate) fn ash_miss_strided(&self) -> ash::vk::StridedDeviceAddressRegionKHR {
        self.miss_region
    }

    pub(crate) fn ash_closesthit_strided(&self) -> ash::vk::StridedDeviceAddressRegionKHR {
        self.closesthit_region
    }

    /// Builds one raygen record, one miss record, all hit-group records in pipeline
    /// group order, and one callable record when present. Each region is aligned
    /// within its backing buffer; an absent callable has an all-zero region.
    pub fn new(
        raytracing_pipeline: Arc<RaytracingPipeline>,
        memory_manager: &mut dyn MemoryManagerTrait,
        allocation_tags: MemoryManagementTags,
    ) -> VulkanResult<Arc<Self>> {
        let device = raytracing_pipeline.get_parent_device();
        if memory_manager.get_parent_device() != device {
            return Err(FrameworkError::ResourceFromIncompatibleDevice.into());
        }

        let rt_info = device
            .ray_tracing_info()
            .as_ref()
            .ok_or_else(|| VulkanError::MissingExtension("VK_KHR_ray_tracing_pipeline".into()))?;
        let rt_ext = device
            .ash_ext_raytracing_pipeline_khr()
            .as_ref()
            .ok_or_else(|| VulkanError::MissingExtension("VK_KHR_ray_tracing_pipeline".into()))?;
        let layout = ShaderBindingTableLayout::new(
            rt_info.shader_group_handle_size(),
            rt_info.shader_group_handle_alignment(),
            rt_info.shader_group_base_alignment(),
            rt_info.max_shader_group_stride(),
        )?;
        let group_count = raytracing_pipeline.shader_group_size();
        let callable_present = raytracing_pipeline.callable_shader_present();
        let hit_count = group_count
            .checked_sub(2 + u32::from(callable_present))
            .filter(|count| *count > 0)
            .ok_or(ash::vk::Result::ERROR_INITIALIZATION_FAILED)?;
        let packed_size = usize::try_from(u64::from(group_count) * layout.handle_size)
            .map_err(|_| ash::vk::Result::ERROR_INITIALIZATION_FAILED)?;
        let shader_handles = unsafe {
            rt_ext.get_ray_tracing_shader_group_handles(
                raytracing_pipeline.ash_handle(),
                0,
                group_count,
                packed_size,
            )
        }?;

        let mut record_counts: smallvec::SmallVec<[u32; 4]> = smallvec::smallvec![1, 1, hit_count];
        let mut first_groups: smallvec::SmallVec<[u32; 4]> = smallvec::smallvec![0, 1, 2];
        if callable_present {
            record_counts.push(1);
            first_groups.push(2 + hit_count);
        }

        let mut buffers = Vec::with_capacity(record_counts.len());
        for &count in &record_counts {
            let descriptor = ConcreteBufferDescriptor::new(
                BufferUsage::from(
                    ash::vk::BufferUsageFlags::SHADER_BINDING_TABLE_KHR
                        | ash::vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                ),
                layout.allocation_size(count)?,
            );
            buffers.push(Buffer::new(device.clone(), descriptor, None, None)?.into());
        }
        let allocations = memory_manager.allocate_resources(
            &MemoryType::device_local_and_host_visible(),
            &MemoryPoolFeatures::new(true),
            buffers,
            allocation_tags,
        )?;
        if allocations.len() != record_counts.len() {
            return Err(ash::vk::Result::ERROR_INITIALIZATION_FAILED.into());
        }

        let mut regions = [ash::vk::StridedDeviceAddressRegionKHR::default(); 4];
        for (index, (allocation, &count)) in allocations.iter().zip(&record_counts).enumerate() {
            let buffer = allocation.buffer();
            if buffer.get_parent_device() != device {
                return Err(FrameworkError::ResourceFromIncompatibleDevice.into());
            }
            if !buffer
                .get_backing_memory_pool()
                .features()
                .device_addressable()
            {
                return Err(ash::vk::Result::ERROR_INITIALIZATION_FAILED.into());
            }
            let address = unsafe {
                device.ash_handle().get_buffer_device_address(
                    &ash::vk::BufferDeviceAddressInfo::default().buffer(buffer.ash_handle()),
                )
            };
            let (offset, region) = layout.region(address, buffer.size(), count)?;
            let end = usize::try_from(offset + region.size)
                .map_err(|_| ash::vk::Result::ERROR_INITIALIZATION_FAILED)?;
            let mapping = MemoryMap::new(buffer.get_backing_memory_pool())?;
            let mut range = mapping.range::<u8>(buffer.clone() as Arc<dyn MemoryPoolBacked>)?;
            layout.write_handles(
                &shader_handles,
                first_groups[index],
                count,
                &mut range.as_mut_slice()[offset as usize..end],
            )?;
            regions[index] = region;
        }

        Ok(Arc::new(Self {
            _raytracing_pipeline: raytracing_pipeline,
            _raygen_buffer: allocations[0].buffer(),
            raygen_region: regions[0],
            _miss_buffer: allocations[1].buffer(),
            miss_region: regions[1],
            _closesthit_buffer: allocations[2].buffer(),
            closesthit_region: regions[2],
            stride: layout.stride,
            callable: if callable_present {
                Some(RaytracingBindingTableCallableBuffer {
                    _callable_buffer: allocations[3].buffer(),
                    callable_buffer_addr: regions[3].device_address,
                })
            } else {
                None
            },
        }))
    }
}

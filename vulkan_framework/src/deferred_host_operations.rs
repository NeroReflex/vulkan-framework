use std::sync::Arc;

use ash::vk::DeferredOperationKHR;

use crate::{
    acceleration_structure::{
        bottom_level::BottomLevelAccelerationStructure, top_level::TopLevelAccelerationStructure,
        AllowedBuildingDevice,
    },
    device::{Device, DeviceOwned},
    prelude::{VulkanError, VulkanResult},
};

pub struct DeferredHostOperationKHR {
    device: Arc<Device>,
    deferred_operation: DeferredOperationKHR,

    tlas: Option<Arc<TopLevelAccelerationStructure>>,
    blas: Option<Arc<BottomLevelAccelerationStructure>>,
}

impl DeviceOwned for DeferredHostOperationKHR {
    fn get_parent_device(&self) -> Arc<Device> {
        self.device.clone()
    }
}

impl Drop for DeferredHostOperationKHR {
    fn drop(&mut self) {
        let Some(deferred_host_op_ext) = self.device.ash_ext_deferred_host_operation_khr() else {
            panic!("Missing VK_KHR_deferred_host_operations extension, but to create this object I HAD to use it. Nasty bug.");
        };

        unsafe { deferred_host_op_ext.deferred_operation_join(self.deferred_operation) }.unwrap();

        unsafe { deferred_host_op_ext.get_deferred_operation_result(self.deferred_operation) }
            .unwrap();

        unsafe { deferred_host_op_ext.destroy_deferred_operation(self.deferred_operation, None) };

        drop(self.tlas.take());
        drop(self.blas.take());
    }
}

impl DeferredHostOperationKHR {
    pub fn build_blas(
        blas: Arc<BottomLevelAccelerationStructure>,
        primitive_offset: u32,
        primitive_count: u32,
        first_vertex: u32,
        transform_offset: u32,
    ) -> VulkanResult<Arc<Self>> {
        let device = blas.index_buffer().buffer().get_parent_device();

        let Some(deferred_host_op_ext) = device.ash_ext_deferred_host_operation_khr() else {
            return Err(VulkanError::MissingExtension(String::from(
                "VK_KHR_deferred_host_operations",
            )));
        };

        let deferred_operation = unsafe { deferred_host_op_ext.create_deferred_operation(None) }?;

        assert!(blas.allowed_building_devices() != AllowedBuildingDevice::DeviceOnly);

        // TODO: assert from same device

        let (geometries, range_infos) = blas
            .ash_build_info(
                primitive_offset,
                primitive_count,
                first_vertex,
                transform_offset,
            )
            .unwrap();

        let ranges_collection: Vec<&[ash::vk::AccelerationStructureBuildRangeInfoKHR]> =
            range_infos.iter().map(|r| r.as_slice()).collect();

        // From vulkan specs: https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkGetAccelerationStructureBuildSizesKHR.html
        // The srcAccelerationStructure, dstAccelerationStructure, and mode members of pBuildInfo are ignored.
        // Any VkDeviceOrHostAddressKHR members of pBuildInfo are ignored by this command,
        // except that the hostAddress member of VkAccelerationStructureGeometryTrianglesDataKHR::transformData
        // will be examined to check if it is NULL.
        let geometry_info = ash::vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .geometries(geometries.as_slice())
            .flags(ash::vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
            .ty(ash::vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
            .mode(ash::vk::BuildAccelerationStructureModeKHR::BUILD)
            .src_acceleration_structure(ash::vk::AccelerationStructureKHR::null())
            .dst_acceleration_structure(blas.ash_handle())
            .scratch_data(blas.device_build_scratch_buffer().addr());

        // check if ray_tracing extension is enabled
        let Some(rt_ext) = device.ash_ext_acceleration_structure_khr() else {
            panic!("Ray tracing pipeline is not enabled!");
        };

        unsafe {
            rt_ext.build_acceleration_structures(
                deferred_operation,
                &[geometry_info],
                ranges_collection.as_slice(),
            )
        }?;

        let tlas = None;
        let blas = Some(blas.clone());
        Ok(Arc::new(Self {
            device,
            deferred_operation,

            tlas,
            blas,
        }))
    }

    pub fn build_tlas(
        tlas: Arc<TopLevelAccelerationStructure>,
        primitive_offset: u32,
        primitive_count: u32,
    ) -> VulkanResult<Arc<Self>> {
        let device = tlas.instance_buffer().buffer().get_parent_device();

        let Some(deferred_host_op_ext) = device.ash_ext_deferred_host_operation_khr() else {
            return Err(VulkanError::MissingExtension(String::from(
                "VK_KHR_deferred_host_operations",
            )));
        };

        let deferred_operation = unsafe { deferred_host_op_ext.create_deferred_operation(None) }?;

        assert!(tlas.allowed_building_devices() != AllowedBuildingDevice::HostOnly);

        let (geometries, range_infos) = tlas
            .ash_build_info(primitive_offset, primitive_count)
            .unwrap();

        assert!(!geometries.is_empty());
        assert!(!range_infos.is_empty());

        let ranges_collection: smallvec::SmallVec<
            [&[ash::vk::AccelerationStructureBuildRangeInfoKHR]; 1],
        > = range_infos.iter().map(|r| r.as_slice()).collect();

        assert!(!ranges_collection.is_empty());

        // See https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkGetAccelerationStructureBuildSizesKHR.html
        let geometry_info = ash::vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .geometries(geometries.as_slice())
            .flags(tlas.build_flags())
            .ty(ash::vk::AccelerationStructureTypeKHR::TOP_LEVEL)
            .mode(ash::vk::BuildAccelerationStructureModeKHR::BUILD)
            .src_acceleration_structure(ash::vk::AccelerationStructureKHR::null())
            .dst_acceleration_structure(tlas.ash_handle())
            .scratch_data(tlas.device_build_scratch_buffer().addr());

        // check if ray_tracing extension is enabled
        let Some(rt_ext) = device.ash_ext_acceleration_structure_khr() else {
            panic!("Ray tracing pipeline is not enabled!");
        };

        assert_eq!(ranges_collection.len(), 1_usize);

        unsafe {
            rt_ext.build_acceleration_structures(
                deferred_operation,
                [geometry_info].as_slice(),
                ranges_collection.as_slice(),
            )
        }?;

        let tlas = Some(tlas.clone());
        let blas = None;
        Ok(Arc::new(Self {
            device,
            deferred_operation,

            tlas,
            blas,
        }))
    }
}

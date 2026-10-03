use std::sync::{Arc, Weak};

use crate::{
    acceleration_structure::{
        bottom_level::{
            BottomLevelAccelerationStructure, BottomLevelAccelerationStructureIndexBuffer,
            BottomLevelAccelerationStructureTransformBuffer,
            BottomLevelAccelerationStructureVertexBuffer, BottomLevelTrianglesGroupDecl,
            BottomLevelVerticesTopologyDecl,
        },
        scratch_buffer::DeviceScratchBuffer,
        top_level::{
            TopLevelAccelerationStructure, TopLevelAccelerationStructureInstanceBuffer,
            TopLevelBLASGroupDecl,
        },
        AllowedBuildingDevice, VertexIndexing,
    },
    buffer::{AllocatedBuffer, Buffer, BufferUsage},
    command_buffer::{CommandBufferRecorder, CommandBufferTrait, PrimaryCommandBuffer},
    command_pool::CommandPool,
    device::{Device, DeviceOwned},
    fence::Fence,
    graphics_pipeline::AttributeType,
    instance::Instance,
    memory_heap::MemoryType,
    memory_management::{
        AllocatedResource, DefaultMemoryManager, MemoryManagementTags, MemoryManagerTrait,
        UnallocatedResource,
    },
    memory_pool::MemoryPoolFeatures,
    prelude::{FrameworkError, VulkanError, VulkanResult},
    queue::Queue,
    queue_family::{ConcreteQueueFamilyDescriptor, QueueFamily, QueueFamilySupportedOperationType},
};

// Track every allocation, including scratch backing buffers, without owning it.
struct TrackingMemoryManager {
    inner: DefaultMemoryManager,
    buffers: Vec<Weak<AllocatedBuffer>>,
}
impl DeviceOwned for TrackingMemoryManager {
    fn get_parent_device(&self) -> Arc<Device> {
        self.inner.get_parent_device()
    }
}
impl MemoryManagerTrait for TrackingMemoryManager {
    fn allocate_resources(
        &mut self,
        memory_type: &MemoryType,
        features: &MemoryPoolFeatures,
        resources: Vec<UnallocatedResource>,
        tags: MemoryManagementTags,
    ) -> VulkanResult<Vec<AllocatedResource>> {
        let allocations = self
            .inner
            .allocate_resources(memory_type, features, resources, tags)?;
        self.buffers.extend(
            allocations
                .iter()
                .map(|allocation| Arc::downgrade(&allocation.buffer())),
        );
        Ok(allocations)
    }
}

struct BuildOwners {
    blas: Arc<BottomLevelAccelerationStructure>,
    tlas: Arc<TopLevelAccelerationStructure>,
}
impl BuildOwners {
    fn record(&self, recorder: &mut CommandBufferRecorder<'_>) {
        // Empty builds exercise all native handles and addresses without requiring
        // initialized geometry or BLAS addresses in TLAS instances.
        recorder.build_blas(self.blas.clone(), 0, 0, 0, 0);
        recorder.build_tlas(self.tlas.clone(), 0, 0);
    }
    fn weak(&self, buffers: &[Weak<AllocatedBuffer>]) -> WeakBuildResources {
        WeakBuildResources {
            blas: Arc::downgrade(&self.blas),
            tlas: Arc::downgrade(&self.tlas),
            blas_scratch: Arc::downgrade(&self.blas.device_build_scratch_buffer()),
            tlas_scratch: Arc::downgrade(&self.tlas.device_build_scratch_buffer()),
            buffers: buffers.to_vec(),
        }
    }
}
struct WeakBuildResources {
    blas: Weak<BottomLevelAccelerationStructure>,
    tlas: Weak<TopLevelAccelerationStructure>,
    blas_scratch: Weak<DeviceScratchBuffer>,
    tlas_scratch: Weak<DeviceScratchBuffer>,
    buffers: Vec<Weak<AllocatedBuffer>>,
}
impl WeakBuildResources {
    fn assert_alive(&self) {
        assert!(self.blas.upgrade().is_some());
        assert!(self.tlas.upgrade().is_some());
        assert!(self.blas_scratch.upgrade().is_some());
        assert!(self.tlas_scratch.upgrade().is_some());
        assert_eq!(
            self.buffers.len(),
            8,
            "four inputs, two destinations and two scratch allocations"
        );
        assert!(self.buffers.iter().all(|buffer| buffer.upgrade().is_some()));
    }
    fn assert_released(&self) {
        assert!(self.blas.upgrade().is_none());
        assert!(self.tlas.upgrade().is_none());
        assert!(self.blas_scratch.upgrade().is_none());
        assert!(self.tlas_scratch.upgrade().is_none());
        assert!(self.buffers.iter().all(|buffer| buffer.upgrade().is_none()));
    }
}

type Fixture = (
    Arc<Device>,
    Arc<QueueFamily>,
    TrackingMemoryManager,
    BuildOwners,
);
fn setup() -> VulkanResult<Option<Fixture>> {
    let instance = Instance::new(&[], &[], &"as_lifetime_tests".into(), &"headless".into())?;
    let descriptor =
        ConcreteQueueFamilyDescriptor::new(&[QueueFamilySupportedOperationType::Compute], &[1.0]);
    let extensions = [
        ash::khr::acceleration_structure::NAME,
        ash::khr::deferred_host_operations::NAME,
    ]
    .map(|name| name.to_str().unwrap().to_owned());
    let device = match Device::new(instance, &[descriptor], &extensions, None) {
        Ok(device) => device,
        Err(VulkanError::Framework(FrameworkError::NoSuitableDeviceFound)) => {
            eprintln!("Skipping AS lifetime test: no device supports acceleration structures");
            return Ok(None);
        }
        Err(err) => return Err(err),
    };
    let family = QueueFamily::new(device.clone(), 0)?;
    let mut memory = TrackingMemoryManager {
        inner: DefaultMemoryManager::new(device.clone()),
        buffers: Vec::new(),
    };
    let vertices = BottomLevelVerticesTopologyDecl::new(3, AttributeType::Vec3, 0);
    let triangles = BottomLevelTrianglesGroupDecl::new(VertexIndexing::UInt32, 1);
    let instances = TopLevelBLASGroupDecl::new();
    let descriptors = [
        BottomLevelAccelerationStructureVertexBuffer::template(&vertices, BufferUsage::default()),
        BottomLevelAccelerationStructureIndexBuffer::template(&triangles, BufferUsage::default()),
        BottomLevelAccelerationStructureTransformBuffer::template(BufferUsage::default()),
        TopLevelAccelerationStructureInstanceBuffer::template(
            &instances,
            1,
            BufferUsage::default(),
        ),
    ];
    let unallocated = descriptors
        .into_iter()
        .map(|descriptor| Buffer::new(device.clone(), descriptor, None, None).map(Into::into))
        .collect::<VulkanResult<Vec<_>>>()?;
    let inputs = memory.allocate_resources(
        &MemoryType::device_local_and_host_visible(),
        &MemoryPoolFeatures::new(true),
        unallocated,
        MemoryManagementTags::default(),
    )?;
    let blas = BottomLevelAccelerationStructure::new(
        &mut memory,
        AllowedBuildingDevice::DeviceOnly,
        BottomLevelAccelerationStructureVertexBuffer::new(vertices, inputs[0].buffer())?,
        BottomLevelAccelerationStructureIndexBuffer::new(triangles, inputs[1].buffer())?,
        BottomLevelAccelerationStructureTransformBuffer::new(inputs[2].buffer())?,
        MemoryManagementTags::default(),
        None,
        None,
        ash::vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE,
    )?;
    let tlas = TopLevelAccelerationStructure::new(
        &mut memory,
        AllowedBuildingDevice::DeviceOnly,
        TopLevelAccelerationStructureInstanceBuffer::new(instances, 1, inputs[3].buffer())?,
        MemoryManagementTags::default(),
        None,
        None,
    )?;
    Ok(Some((device, family, memory, BuildOwners { blas, tlas })))
}

#[test]
fn one_time_as_builds_retain_owners_inputs_storage_and_scratch_through_rollback() -> VulkanResult<()>
{
    let Some((device, family, memory, owners)) = setup()? else {
        return Ok(());
    };
    let resources = owners.weak(&memory.buffers);
    let pool = CommandPool::new(family.clone(), None)?;
    let command_buffer = PrimaryCommandBuffer::new(pool.clone(), None)?;
    command_buffer.record_one_time_submit(|recorder| owners.record(recorder))?;
    assert_eq!(Arc::strong_count(&owners.blas), 2);
    assert_eq!(Arc::strong_count(&owners.tlas), 2);
    drop(owners);
    resources.assert_alive();
    let queue = Queue::new(family, None)?;
    let fence = Fence::new(device, false, None)?;
    let bad_batch: Vec<Arc<dyn CommandBufferTrait>> = vec![
        command_buffer.clone(),
        PrimaryCommandBuffer::new(pool, None)?,
    ];
    assert!(queue.submit(&bad_batch, &[], &[], fence.clone()).is_err());
    resources.assert_alive();
    fence.reset()?;
    let good_batch: Vec<Arc<dyn CommandBufferTrait>> = vec![command_buffer];
    assert!(Queue::submit_marked(&good_batch, || Err(
        ash::vk::Result::ERROR_OUT_OF_HOST_MEMORY.into()
    ))
    .is_err());
    resources.assert_alive();
    let waiter = queue.submit(&good_batch, &[], &[], fence)?;
    resources.assert_alive();
    drop(waiter);
    resources.assert_released();
    Ok(())
}

#[test]
fn reusable_as_builds_retain_resources_after_completion_until_rerecord_or_drop() -> VulkanResult<()>
{
    for rerecord in [false, true] {
        let Some((device, family, memory, owners)) = setup()? else {
            return Ok(());
        };
        let resources = owners.weak(&memory.buffers);
        let pool = CommandPool::new(family.clone(), None)?;
        let command_buffer = PrimaryCommandBuffer::new(pool, None)?;
        command_buffer.record_commands_raw(
            |recorder| owners.record(recorder),
            ash::vk::CommandBufferUsageFlags::empty(),
        )?;
        drop(owners);
        resources.assert_alive();
        let queue = Queue::new(family, None)?;
        drop(queue.submit(
            &[command_buffer.clone()],
            &[],
            &[],
            Fence::new(device, false, None)?,
        )?);
        resources.assert_alive();
        if rerecord {
            command_buffer.record_one_time_submit(|_| {})?;
        } else {
            drop(command_buffer);
        }
        resources.assert_released();
    }
    Ok(())
}

#[test]
fn discarded_as_recordings_release_resources_without_submission() -> VulkanResult<()> {
    let Some((_, family, memory, owners)) = setup()? else {
        return Ok(());
    };
    let resources = owners.weak(&memory.buffers);
    let command_buffer = PrimaryCommandBuffer::new(CommandPool::new(family, None)?, None)?;
    command_buffer.record_one_time_submit(|recorder| {
        owners.record(recorder);
        let stages = [
            crate::pipeline_stage::PipelineStage::AccelerationStructureKHR(
                crate::pipeline_stage::PipelineStageAccelerationStructureKHR::Build,
            ),
        ]
        .as_slice()
        .into();
        let access = ash::vk::AccessFlags2::ACCELERATION_STRUCTURE_WRITE_KHR.into();
        recorder.pipeline_barriers([crate::memory_barriers::MemoryBarrier::new(
            stages, access, stages, access,
        )
        .into()]);
        owners.record(recorder);
    })?;
    assert_eq!(
        Arc::strong_count(&owners.blas),
        2,
        "duplicate references must be deduplicated"
    );
    assert_eq!(Arc::strong_count(&owners.tlas), 2);
    drop(owners);
    resources.assert_alive();
    command_buffer.record_one_time_submit(|_| {})?;
    resources.assert_released();
    Ok(())
}

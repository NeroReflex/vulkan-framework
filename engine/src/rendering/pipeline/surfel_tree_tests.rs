use std::sync::Arc;

use vulkan_framework::{
    buffer::{Buffer, BufferTrait, BufferUseAs, ConcreteBufferDescriptor},
    command_buffer::{CommandBufferRecorder, CommandBufferTrait, PrimaryCommandBuffer},
    command_pool::CommandPool,
    compute_pipeline::ComputePipeline,
    descriptor_pool::{
        DescriptorPool, DescriptorPoolConcreteDescriptor, DescriptorPoolSizesConcreteDescriptor,
    },
    descriptor_set::DescriptorSet,
    descriptor_set_layout::DescriptorSetLayout,
    device::Device,
    fence::Fence,
    instance::Instance,
    memory_barriers::{MemoryAccessAs, MemoryBarrier},
    memory_heap::MemoryType,
    memory_management::{
        DefaultMemoryManager, MemoryManagementTags, MemoryManagerTrait, UnallocatedResource,
    },
    memory_pool::{MemoryMap, MemoryPoolBacked, MemoryPoolFeatures},
    pipeline_layout::PipelineLayout,
    pipeline_stage::PipelineStage,
    prelude::*,
    push_constant_range::PushConstanRange,
    queue::Queue,
    queue_family::{ConcreteQueueFamilyDescriptor, QueueFamily, QueueFamilySupportedOperationType},
    shader_layout_binding::{BindingDescriptor, BindingType, NativeBindingType},
    shader_stage_access::{ShaderStageAccessIn, ShaderStagesAccess},
    shaders::compute_shader::ComputeShader,
};

use inline_spirv::inline_spirv;

use super::global_illumination::{BUILD_BLOCKS, BUILD_RANGE_CAP, BUILD_SPLIT_ROUNDS, BUILD_WORDS};

const LEAF: u32 = 0x8000_0000;
const DEAD: u32 = 1 << 2;
const MISSING: u32 = 0xFFFF_FFFF;
const SURFELS_MISSED: u32 = 0xFFFF_FFFE;
const SURFELS_TOO_CLOSE: u32 = 0xFFFF_FFFD;
const GATHER_MAX: usize = 64;
const SURFEL_FLOATS: usize = 19;

const LEGACY_SPV: &[u32] = inline_spirv!(
    r#"
#version 460
#include "engine/shaders/surfel_reorder/legacy_counting_sort.comp"
"#,
    glsl,
    comp,
    vulkan1_2,
    entry = "main"
);

const PREFIX_SPV: &[u32] = inline_spirv!(
    r#"
#version 460
#include "engine/shaders/surfel_reorder/surfel_prefix.comp"
"#,
    glsl,
    comp,
    vulkan1_2,
    entry = "main"
);

const COMMIT_SPV: &[u32] = inline_spirv!(
    r#"
#version 460
#include "engine/shaders/surfel_reorder/surfel_commit.comp"
"#,
    glsl,
    comp,
    vulkan1_2,
    entry = "main"
);

const SPLIT_SPV: &[u32] = inline_spirv!(
    r#"
#version 460
#include "engine/shaders/surfel_reorder/surfel_bvh_split.comp"
"#,
    glsl,
    comp,
    vulkan1_2,
    entry = "main"
);

const COMPACT_SPV: &[u32] = inline_spirv!(
    r#"
#version 460
#include "engine/shaders/surfel_reorder/surfel_bvh_compact.comp"
"#,
    glsl,
    comp,
    vulkan1_2,
    entry = "main"
);

const AABB_SPV: &[u32] = inline_spirv!(
    r#"
#version 460
#include "engine/shaders/surfel_reorder/bvh_aabb.comp"
"#,
    glsl,
    comp,
    vulkan1_2,
    entry = "main"
);

const QUERY_SPV: &[u32] = inline_spirv!(
    r#"
#version 460
#include "engine/shaders/surfel_reorder/surfel_query_test.comp"
"#,
    glsl,
    comp,
    vulkan1_2,
    entry = "main"
);

#[repr(C)]
#[derive(Clone, Copy)]
struct SurfelGpu {
    instance_id: u32,
    position_x: f32,
    position_y: f32,
    position_z: f32,
    radius: f32,
    normal: u32,
    diffuse: [f32; 3],
    irradiance: [f32; 3],
    direct: [f32; 3],
    contributions: u32,
    frame_contributions: u32,
    flags: u32,
    latest_contribution: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Stats {
    total: i32,
    unordered: i32,
    live: i32,
    high_water: i32,
    reserve: u32,
    discovered: u32,
    pad2: u32,
    pad3: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct BvhNode {
    min_x: f32,
    min_y: f32,
    min_z: f32,
    max_x: f32,
    max_y: f32,
    max_z: f32,
    parent: u32,
    left: u32,
    right: u32,
    flags: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct QueryItem {
    x: f32,
    y: f32,
    z: f32,
    radius: f32,
    count: u32,
    overflow: u32,
    ids: [u32; GATHER_MAX],
    instance_id: u32,
    spawn_result: u32,
}

#[repr(C)]
struct LegacyPush {
    count: u32,
    eye_x: f32,
    eye_y: f32,
    eye_z: f32,
    clip_near: f32,
    clip_far: f32,
}

fn expand_bits(mut v: u32) -> u32 {
    v = v.wrapping_mul(0x0001_0001) & 0xFF00_00FF;
    v = v.wrapping_mul(0x0000_0101) & 0x0F00_F00F;
    v = v.wrapping_mul(0x0000_0011) & 0xC30C_30C3;
    v = v.wrapping_mul(0x0000_0005) & 0x4924_9249;
    v
}

fn morton_unit(p: [f32; 3]) -> u32 {
    let q = p.map(|c| (c * 1024.0).clamp(0.0, 1023.0) as u32);
    expand_bits(q[0]) * 4 + expand_bits(q[1]) * 2 + expand_bits(q[2])
}

fn morton_eye(eye: [f32; 3], p: [f32; 3], near: f32, far: f32) -> u32 {
    let range = far.abs() - near.abs();
    for axis in 0..3 {
        if p[axis] < eye[axis] - range || p[axis] > eye[axis] + range {
            return 0xFFFF_FFFF;
        }
    }
    let n = std::array::from_fn(|axis| (p[axis] - (eye[axis] - range)) / (2.0 * range));
    morton_unit(n)
}

fn cpu_order(points: &[[f32; 3]], eye: [f32; 3], near: f32, far: f32) -> Vec<u32> {
    let codes: Vec<u32> = points.iter().map(|p| morton_eye(eye, *p, near, far)).collect();
    let mut order: Vec<u32> = (0..points.len() as u32).collect();
    order.sort_by(|&a, &b| {
        codes[a as usize]
            .cmp(&codes[b as usize])
            .then(a.cmp(&b))
    });
    order
}

fn storage(device: Arc<Device>, bytes: u64, name: &str) -> VulkanResult<UnallocatedResource> {
    Ok(Buffer::new(
        device,
        ConcreteBufferDescriptor::new([BufferUseAs::StorageBuffer].as_slice().into(), bytes),
        None,
        Some(name),
    )?
    .into())
}

fn bind_storage(set: &DescriptorSet, binding: u32, buffer: Arc<dyn BufferTrait>) -> VulkanResult<()> {
    set.bind_resources(|binder| {
        binder
            .bind_storage_buffers(binding, [(buffer, None, None)].as_slice())
            .unwrap();
    })
}

fn compute_barrier(recorder: &mut CommandBufferRecorder) {
    recorder.pipeline_barriers([MemoryBarrier::new(
        [PipelineStage::ComputeShader].as_slice().into(),
        [MemoryAccessAs::ShaderWrite, MemoryAccessAs::ShaderRead]
            .as_slice()
            .into(),
        [PipelineStage::ComputeShader].as_slice().into(),
        [MemoryAccessAs::ShaderWrite, MemoryAccessAs::ShaderRead]
            .as_slice()
            .into(),
    )
    .into()]);
}

fn push_bytes<T>(value: &T) -> &[u8] {
    unsafe { std::slice::from_raw_parts(value as *const T as *const u8, std::mem::size_of::<T>()) }
}

struct ComputeDevice {
    device: Arc<Device>,
    queue: Arc<Queue>,
    pool: Arc<CommandPool>,
}

fn setup_compute_device() -> VulkanResult<ComputeDevice> {
    let instance = Instance::new(&[], &[], &"surfel_tests".to_string(), &"test".to_string())?;
    let queue_descriptor =
        ConcreteQueueFamilyDescriptor::new(&[QueueFamilySupportedOperationType::Compute], &[1.0]);
    let device = Device::new(instance, &[queue_descriptor], &[], Some("surfel_test_device"))?;
    assert!(
        device.ray_tracing_info().is_none(),
        "the test device must be created without a ray-tracing extension"
    );
    let queue_family = QueueFamily::new(device.clone(), 0)?;
    let pool = CommandPool::new(queue_family.clone(), Some("surfel_test_pool"))?;
    let queue = Queue::new(queue_family, Some("surfel_test_queue"))?;
    Ok(ComputeDevice { device, queue, pool })
}

fn skip_if_no_device() -> Option<ComputeDevice> {
    match setup_compute_device() {
        Ok(device) => Some(device),
        Err(err) => {
            eprintln!("Skipping surfel test, no compute device: {err}");
            None
        }
    }
}

#[test]
fn legacy_counting_sort_matches_cpu() -> VulkanResult<()> {
    let Some(gpu) = skip_if_no_device() else {
        return Ok(());
    };

    let eye = [0.0, 0.0, 0.0];
    let near = 1.0;
    let far = 80.0;
    let mut points = Vec::new();
    for i in 0..96 {
        let x = (i % 8) as f32 * 3.5 - 10.0;
        let y = (i / 8) as f32 * 2.0;
        let z = if i % 5 == 0 { y } else { (i % 3) as f32 };
        points.push([x, y, z]);
    }
    // Duplicate of the first point, so the stable tie break is observable.
    points.push(points[0]);
    // Outside the camera cube. The old code sorts this as 0xFFFFFFFF.
    points.push([10_000.0, 0.0, 0.0]);

    let expected = cpu_order(&points, eye, near, far);

    let point_bytes = (points.len() * std::mem::size_of::<[f32; 3]>()) as u64;
    let order_bytes = (points.len() * std::mem::size_of::<u32>()) as u64;
    let mut memory = DefaultMemoryManager::new(gpu.device.clone());
    let allocated = memory.allocate_resources(
        &MemoryType::host_visible_and_coherent(),
        &MemoryPoolFeatures::new(false),
        vec![
            storage(gpu.device.clone(), point_bytes, "legacy_points")?,
            storage(gpu.device.clone(), order_bytes, "legacy_order")?,
        ],
        MemoryManagementTags::default(),
    )?;
    let point_buffer = allocated[0].buffer();
    let order_buffer = allocated[1].buffer();

    {
        let map = MemoryMap::new(point_buffer.get_backing_memory_pool())?;
        let mut range = map.range::<[f32; 3]>(point_buffer.clone() as Arc<dyn MemoryPoolBacked>)?;
        range.as_mut_slice()[..points.len()].copy_from_slice(&points);
    }

    let binding = |index| {
        BindingDescriptor::new(
            ShaderStagesAccess::compute(),
            BindingType::Native(NativeBindingType::StorageBuffer),
            index,
            1,
        )
    };
    let layout = DescriptorSetLayout::new(gpu.device.clone(), &[binding(0), binding(1)])?;
    let desc_pool = DescriptorPool::new(
        gpu.device.clone(),
        DescriptorPoolConcreteDescriptor::new(
            DescriptorPoolSizesConcreteDescriptor::new(0, 0, 0, 0, 0, 0, 2, 0, 0, None),
            1,
        ),
        Some("legacy_pool"),
    )?;
    let set = DescriptorSet::new(desc_pool, layout.clone())?;
    bind_storage(&set, 0, point_buffer.clone())?;
    bind_storage(&set, 1, order_buffer.clone())?;

    let push_stages: ShaderStagesAccess = [ShaderStageAccessIn::Compute].as_slice().into();
    let pipeline_layout = PipelineLayout::new(
        gpu.device.clone(),
        &[layout],
        &[PushConstanRange::new(0, std::mem::size_of::<LegacyPush>() as u32, push_stages.clone())],
        Some("legacy_layout"),
    )?;
    let pipeline = ComputePipeline::new(
        None,
        pipeline_layout.clone(),
        (
            ComputeShader::new(gpu.device.clone(), LEGACY_SPV)?,
            None,
        ),
        Some("legacy_sort"),
    )?;

    let cmd = PrimaryCommandBuffer::new(gpu.pool.clone(), Some("legacy_cb"))?;
    let params = LegacyPush {
        count: points.len() as u32,
        eye_x: eye[0],
        eye_y: eye[1],
        eye_z: eye[2],
        clip_near: near,
        clip_far: far,
    };
    cmd.record_one_time_submit(|recorder| {
        recorder.bind_compute_pipeline(pipeline.clone());
        recorder.bind_descriptor_sets_for_compute_pipeline(
            pipeline_layout.clone(),
            0,
            &[set.clone()],
        );
        recorder.push_constant(pipeline_layout.clone(), push_stages.clone(), 0, push_bytes(&params));
        recorder.dispatch(1, 1, 1);
    })?;
    let fence = Fence::new(gpu.device.clone(), false, Some("legacy_fence"))?;
    let buffers: Vec<Arc<dyn CommandBufferTrait>> = vec![cmd];
    drop(gpu.queue.submit(buffers.as_slice(), &[], &[], fence)?);

    let map = MemoryMap::new(order_buffer.get_backing_memory_pool())?;
    let order = map.range::<u32>(order_buffer.clone() as Arc<dyn MemoryPoolBacked>)?;
    let got = &order.as_slice()[..points.len()];
    assert_eq!(got, expected.as_slice());
    Ok(())
}

struct TreePipelines {
    prefix: Arc<ComputePipeline>,
    commit: Arc<ComputePipeline>,
    split: Arc<ComputePipeline>,
    compact: Arc<ComputePipeline>,
    aabb: Arc<ComputePipeline>,
    query: Arc<ComputePipeline>,
    layout: Arc<PipelineLayout>,
    prefix_layout: Arc<PipelineLayout>,
    set: Arc<DescriptorSet>,
}

fn tree_pipelines(
    device: Arc<Device>,
    stats: Arc<dyn BufferTrait>,
    surfels: Arc<dyn BufferTrait>,
    nodes: Arc<dyn BufferTrait>,
    discovered: Arc<dyn BufferTrait>,
    scratch: Arc<dyn BufferTrait>,
    queries: Arc<dyn BufferTrait>,
) -> VulkanResult<TreePipelines> {
    let binding = |index| {
        BindingDescriptor::new(
            ShaderStagesAccess::compute(),
            BindingType::Native(NativeBindingType::StorageBuffer),
            index,
            1,
        )
    };
    let set_layout = DescriptorSetLayout::new(
        device.clone(),
        &[
            binding(0),
            binding(1),
            binding(2),
            binding(3),
            binding(6),
            binding(7),
        ],
    )?;
    let pool = DescriptorPool::new(
        device.clone(),
        DescriptorPoolConcreteDescriptor::new(
            DescriptorPoolSizesConcreteDescriptor::new(0, 0, 0, 0, 0, 0, 6, 0, 0, None),
            1,
        ),
        Some("tree_pool"),
    )?;
    let set = DescriptorSet::new(pool, set_layout.clone())?;
    bind_storage(&set, 0, stats)?;
    bind_storage(&set, 1, surfels)?;
    bind_storage(&set, 2, nodes)?;
    bind_storage(&set, 3, discovered)?;
    bind_storage(&set, 6, scratch)?;
    bind_storage(&set, 7, queries)?;

    let layout = PipelineLayout::new(
        device.clone(),
        &[set_layout.clone()],
        &[],
        Some("tree_layout"),
    )?;
    let prefix_stages: ShaderStagesAccess = [ShaderStageAccessIn::Compute].as_slice().into();
    let prefix_layout = PipelineLayout::new(
        device.clone(),
        &[set_layout],
        &[PushConstanRange::new(0, 8, prefix_stages)],
        Some("tree_prefix_layout"),
    )?;
    let make = |words, pipe_layout: Arc<PipelineLayout>, name| {
        ComputePipeline::new(
            None,
            pipe_layout,
            (ComputeShader::new(device.clone(), words)?, None),
            Some(name),
        )
    };
    Ok(TreePipelines {
        prefix: make(PREFIX_SPV, prefix_layout.clone(), "prefix")?,
        commit: make(COMMIT_SPV, layout.clone(), "commit")?,
        split: make(SPLIT_SPV, layout.clone(), "split")?,
        compact: make(COMPACT_SPV, layout.clone(), "compact")?,
        aabb: make(AABB_SPV, layout.clone(), "aabb")?,
        query: make(QUERY_SPV, layout.clone(), "query")?,
        layout,
        prefix_layout,
        set,
    })
}

fn dispatch_prefix(
    recorder: &mut CommandBufferRecorder,
    pipelines: &TreePipelines,
    phase: u32,
    kind: u32,
) {
    recorder.bind_compute_pipeline(pipelines.prefix.clone());
    recorder.bind_descriptor_sets_for_compute_pipeline(
        pipelines.prefix_layout.clone(),
        0,
        &[pipelines.set.clone()],
    );
    let params = [phase, kind];
    recorder.push_constant(
        pipelines.prefix_layout.clone(),
        [ShaderStageAccessIn::Compute].as_slice().into(),
        0,
        push_bytes(&params),
    );
    let groups = if phase == 0 || phase == 2 {
        BUILD_BLOCKS
    } else {
        1
    };
    recorder.dispatch(groups, 1, 1);
}

fn dispatch_build(recorder: &mut CommandBufferRecorder, pipelines: &TreePipelines, with_commit: bool) {
    if with_commit {
        dispatch_prefix(recorder, pipelines, 0, 0);
        compute_barrier(recorder);
        dispatch_prefix(recorder, pipelines, 1, 0);
        compute_barrier(recorder);
        dispatch_prefix(recorder, pipelines, 2, 0);
        compute_barrier(recorder);
        recorder.bind_compute_pipeline(pipelines.commit.clone());
        recorder.bind_descriptor_sets_for_compute_pipeline(
            pipelines.layout.clone(),
            0,
            &[pipelines.set.clone()],
        );
        recorder.dispatch(1, 1, 1);
        compute_barrier(recorder);
    }
    dispatch_prefix(recorder, pipelines, 0, 1);
    compute_barrier(recorder);
    dispatch_prefix(recorder, pipelines, 1, 1);
    compute_barrier(recorder);
    dispatch_prefix(recorder, pipelines, 2, 1);
    compute_barrier(recorder);
    dispatch_prefix(recorder, pipelines, 3, 1);
    compute_barrier(recorder);
    for _ in 0..BUILD_SPLIT_ROUNDS {
        recorder.bind_compute_pipeline(pipelines.split.clone());
        recorder.bind_descriptor_sets_for_compute_pipeline(
            pipelines.layout.clone(),
            0,
            &[pipelines.set.clone()],
        );
        recorder.dispatch(BUILD_RANGE_CAP, 1, 1);
        compute_barrier(recorder);
        recorder.bind_compute_pipeline(pipelines.compact.clone());
        recorder.bind_descriptor_sets_for_compute_pipeline(
            pipelines.layout.clone(),
            0,
            &[pipelines.set.clone()],
        );
        recorder.dispatch(1, 1, 1);
        compute_barrier(recorder);
    }
    for _ in 0..BUILD_SPLIT_ROUNDS {
        recorder.bind_compute_pipeline(pipelines.aabb.clone());
        recorder.bind_descriptor_sets_for_compute_pipeline(
            pipelines.layout.clone(),
            0,
            &[pipelines.set.clone()],
        );
        recorder.dispatch(BUILD_BLOCKS, 1, 1);
        compute_barrier(recorder);
    }
}

fn surfel_at(index: u32, flags: u32) -> SurfelGpu {
    SurfelGpu {
        instance_id: 1,
        position_x: (index % 32) as f32 * 10.0,
        position_y: (index / 32) as f32 * 10.0,
        position_z: 0.0,
        radius: 1.0,
        normal: 0,
        diffuse: [1.0, 1.0, 1.0],
        irradiance: [0.0; 3],
        direct: [0.0; 3],
        contributions: 0,
        frame_contributions: 0,
        flags,
        latest_contribution: 0,
    }
}

fn live_indices(surfels: &[SurfelGpu], high_water: usize) -> Vec<u32> {
    (0..high_water)
        .filter(|&i| surfels[i].flags & DEAD == 0)
        .map(|i| i as u32)
        .collect()
}

fn overlaps(a: [f32; 3], ra: f32, b: [f32; 3], rb: f32) -> bool {
    let d = (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2);
    d <= (ra + rb) * (ra + rb)
}

fn ceil_log2(n: u32) -> u32 {
    if n <= 1 {
        0
    } else {
        32 - (n - 1).leading_zeros()
    }
}

fn assert_tree(nodes: &[BvhNode], surfels: &[SurfelGpu], live: &[u32]) {
    if live.is_empty() {
        assert_eq!(nodes[0].left, nodes[0].parent);
        return;
    }
    if live.len() == 1 {
        assert_eq!(nodes[0].left, live[0] | LEAF);
        assert_eq!(nodes[0].right, live[0] | LEAF);
        return;
    }

    fn walk(nodes: &[BvhNode], index: u32, depth: u32, leaves: &mut Vec<u32>, max_depth: &mut u32) {
        let node = nodes[index as usize];
        for child in [node.left, node.right] {
            if child & LEAF != 0 {
                *max_depth = (*max_depth).max(depth + 1);
                leaves.push(child & !LEAF);
            } else {
                assert_eq!(nodes[child as usize].parent, index);
                walk(nodes, child, depth + 1, leaves, max_depth);
            }
        }
    }

    let mut leaves = Vec::new();
    let mut max_depth = 0;
    assert_eq!(nodes[0].parent, 0);
    walk(nodes, 0, 0, &mut leaves, &mut max_depth);
    leaves.sort_unstable();
    let mut expected = live.to_vec();
    expected.sort_unstable();
    assert_eq!(leaves, expected);
    assert_eq!(max_depth, ceil_log2(live.len() as u32));

    fn aabb_contains(parent: &BvhNode, min: [f32; 3], max: [f32; 3]) -> bool {
        parent.min_x <= min[0] + 1e-4
            && parent.min_y <= min[1] + 1e-4
            && parent.min_z <= min[2] + 1e-4
            && parent.max_x + 1e-4 >= max[0]
            && parent.max_y + 1e-4 >= max[1]
            && parent.max_z + 1e-4 >= max[2]
    }
    for node in nodes.iter().take(live.len() - 1) {
        for child in [node.left, node.right] {
            let (min, max) = if child & LEAF != 0 {
                let id = (child & !LEAF) as usize;
                let s = &surfels[id];
                (
                    [s.position_x - s.radius, s.position_y - s.radius, s.position_z - s.radius],
                    [s.position_x + s.radius, s.position_y + s.radius, s.position_z + s.radius],
                )
            } else {
                let child_node = &nodes[child as usize];
                (
                    [child_node.min_x, child_node.min_y, child_node.min_z],
                    [child_node.max_x, child_node.max_y, child_node.max_z],
                )
            };
            assert!(aabb_contains(node, min, max));
        }
    }
}

fn query_at(x: f32, y: f32, z: f32, radius: f32, instance_id: u32) -> QueryItem {
    QueryItem {
        x,
        y,
        z,
        radius,
        count: 0,
        overflow: 0,
        ids: [0; GATHER_MAX],
        instance_id,
        spawn_result: 0,
    }
}

fn squared(a: [f32; 3], b: [f32; 3]) -> f32 {
    (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)
}

fn cpu_spawn(surfels: &[SurfelGpu], ids: &[u32], point: [f32; 3], radius: f32, instance: u32) -> u32 {
    let mut too_close = false;
    for id in ids {
        let surfel = &surfels[*id as usize];
        let center = [surfel.position_x, surfel.position_y, surfel.position_z];
        let d = squared(point, center);
        if d <= surfel.radius * surfel.radius && surfel.instance_id == instance {
            return *id;
        }
        let reach = radius + surfel.radius;
        if d < reach * reach {
            too_close = true;
        }
    }
    if too_close {
        SURFELS_TOO_CLOSE
    } else {
        SURFELS_MISSED
    }
}

#[test]
fn median_bvh_matches_brute_force_without_moving_surfels() -> VulkanResult<()> {
    let Some(gpu) = skip_if_no_device() else {
        return Ok(());
    };

    const CAP: usize = 2050;
    const NODES: usize = 4096;
    let surfel_bytes = (CAP * std::mem::size_of::<SurfelGpu>()) as u64;
    let node_bytes = (NODES * std::mem::size_of::<BvhNode>()) as u64;
    let query_bytes = (8 * std::mem::size_of::<QueryItem>()) as u64;
    let discovered_bytes = 8182 * 4;

    let mut memory = DefaultMemoryManager::new(gpu.device.clone());
    let allocated = memory.allocate_resources(
        &MemoryType::host_visible_and_coherent(),
        &MemoryPoolFeatures::new(false),
        vec![
            storage(gpu.device.clone(), std::mem::size_of::<Stats>() as u64, "stats")?,
            storage(gpu.device.clone(), surfel_bytes, "surfels")?,
            storage(gpu.device.clone(), node_bytes, "nodes")?,
            storage(gpu.device.clone(), discovered_bytes, "discovered")?,
            storage(gpu.device.clone(), (BUILD_WORDS as u64) * 4, "scratch")?,
            storage(gpu.device.clone(), query_bytes, "queries")?,
        ],
        MemoryManagementTags::default(),
    )?;
    let stats_b = allocated[0].buffer();
    let surfels_b = allocated[1].buffer();
    let nodes_b = allocated[2].buffer();
    let discovered_b = allocated[3].buffer();
    let scratch_b = allocated[4].buffer();
    let queries_b = allocated[5].buffer();
    let pipelines = tree_pipelines(
        gpu.device.clone(),
        stats_b.clone(),
        surfels_b.clone(),
        nodes_b.clone(),
        discovered_b.clone(),
        scratch_b.clone(),
        queries_b.clone(),
    )?;

    assert_eq!(
        std::mem::size_of::<SurfelGpu>() / 4,
        SURFEL_FLOATS,
        "the GPU struct must stay 19 scalars"
    );

    // A few hundred, then a few thousand. The second case also has two
    // surfels in the upper half, which the tree does not contain.
    for (high_water, fresh) in [(256u32, 0u32), (2048u32, 2u32)] {
        let mut surfels = vec![surfel_at(0, DEAD); CAP];
        for i in 0..high_water as usize {
            surfels[i] = surfel_at(i as u32, 0);
        }
        surfels[10].flags = DEAD;
        let half = high_water as usize;
        if fresh > 0 {
            surfels[half] = surfel_at(0, 0);
            surfels[half].instance_id = 7;
            surfels[half].position_x = 800.0;
            surfels[half].position_y = 800.0;
            surfels[half].position_z = 800.0;
            surfels[half + 1] = surfel_at(0, 0);
            surfels[half + 1].instance_id = 8;
            surfels[half + 1].position_x = -800.0;
            surfels[half + 1].position_y = -800.0;
            surfels[half + 1].position_z = -800.0;
        }
        let before = surfels.clone();
        let anchor = 40usize;
        let queries = [
            query_at(surfels[0].position_x, surfels[0].position_y, 0.0, 0.25, 1),
            query_at(1_000.0, 1_000.0, 1_000.0, 0.1, 1),
            query_at(
                surfels[anchor].position_x + 1.5,
                surfels[anchor].position_y,
                0.0,
                1.0,
                99,
            ),
            query_at(
                surfels[anchor].position_x + 5.0,
                surfels[anchor].position_y,
                0.0,
                6.0,
                1,
            ),
            query_at(800.0, 800.0, 800.0, 0.2, 7),
        ];

        write_case(
            &stats_b,
            &surfels_b,
            &nodes_b,
            &scratch_b,
            &queries_b,
            Stats {
                total: (high_water * 2) as i32,
                unordered: fresh as i32,
                live: 0,
                high_water: high_water as i32,
                reserve: 0,
                discovered: 0,
                pad2: 0,
                pad3: 0,
            },
            &surfels,
            &queries,
        )?;

        let cmd = PrimaryCommandBuffer::new(gpu.pool.clone(), Some("tree_cb"))?;
        cmd.record_one_time_submit(|recorder| {
            dispatch_build(recorder, &pipelines, false);
            recorder.bind_compute_pipeline(pipelines.query.clone());
            recorder.bind_descriptor_sets_for_compute_pipeline(
                pipelines.layout.clone(),
                0,
                &[pipelines.set.clone()],
            );
            recorder.dispatch(queries.len() as u32, 1, 1);
        })?;
        let fence = Fence::new(gpu.device.clone(), false, Some("tree_fence"))?;
        let buffers: Vec<Arc<dyn CommandBufferTrait>> = vec![cmd];
        drop(gpu.queue.submit(buffers.as_slice(), &[], &[], fence)?);

        let surfel_out = {
            let map = MemoryMap::new(surfels_b.get_backing_memory_pool())?;
            let range = map.range::<SurfelGpu>(surfels_b.clone() as Arc<dyn MemoryPoolBacked>)?;
            range.as_slice()[..CAP].to_vec()
        };
        for i in 0..CAP {
            assert_eq!(surfel_out[i].position_x.to_bits(), before[i].position_x.to_bits());
            assert_eq!(surfel_out[i].flags, before[i].flags);
            assert_eq!(surfel_out[i].instance_id, before[i].instance_id);
        }

        let nodes = {
            let map = MemoryMap::new(nodes_b.get_backing_memory_pool())?;
            let range = map.range::<BvhNode>(nodes_b.clone() as Arc<dyn MemoryPoolBacked>)?;
            range.as_slice()[..NODES].to_vec()
        };
        let live = live_indices(&surfel_out, high_water as usize);
        assert_eq!(live.len(), high_water as usize - 1);
        assert_tree(&nodes, &surfel_out, &live);

        let got_queries = {
            let map = MemoryMap::new(queries_b.get_backing_memory_pool())?;
            let range = map.range::<QueryItem>(queries_b.clone() as Arc<dyn MemoryPoolBacked>)?;
            range.as_slice()[..queries.len()].to_vec()
        };
        let mut upper = Vec::new();
        for i in 0..fresh {
            upper.push(half as u32 + i);
        }
        for (item, src) in got_queries.iter().zip(queries.iter()) {
            if fresh == 0 && src.x == 800.0 {
                continue;
            }
            assert_eq!(item.overflow, 0, "query radius must stay under the gather cap");
            let mut brute = Vec::new();
            for id in live.iter().chain(upper.iter()) {
                let s = &surfel_out[*id as usize];
                if overlaps(
                    [src.x, src.y, src.z],
                    src.radius,
                    [s.position_x, s.position_y, s.position_z],
                    s.radius,
                ) {
                    brute.push(*id);
                }
            }
            let mut got = item.ids[..item.count as usize].to_vec();
            got.sort_unstable();
            brute.sort_unstable();
            assert_eq!(got, brute);

            let ordered = cpu_spawn(&surfel_out, &live, [src.x, src.y, src.z], src.radius, src.instance_id);
            let expected_spawn = if ordered != SURFELS_MISSED {
                ordered
            } else {
                cpu_spawn(&surfel_out, &upper, [src.x, src.y, src.z], src.radius, src.instance_id)
            };
            assert_eq!(item.spawn_result, expected_spawn);
        }
    }
    Ok(())
}

#[test]
fn commit_fills_holes_without_moving_survivors() -> VulkanResult<()> {
    let Some(gpu) = skip_if_no_device() else {
        return Ok(());
    };

    const N: usize = 64;
    let surfel_bytes = (N * std::mem::size_of::<SurfelGpu>()) as u64;
    let mut memory = DefaultMemoryManager::new(gpu.device.clone());
    let allocated = memory.allocate_resources(
        &MemoryType::host_visible_and_coherent(),
        &MemoryPoolFeatures::new(false),
        vec![
            storage(gpu.device.clone(), std::mem::size_of::<Stats>() as u64, "c_stats")?,
            storage(gpu.device.clone(), surfel_bytes, "c_surfels")?,
            storage(gpu.device.clone(), 128 * std::mem::size_of::<BvhNode>() as u64, "c_nodes")?,
            storage(gpu.device.clone(), 8182 * 4, "c_discovered")?,
            storage(gpu.device.clone(), (BUILD_WORDS as u64) * 4, "c_scratch")?,
            storage(gpu.device.clone(), std::mem::size_of::<QueryItem>() as u64, "c_queries")?,
        ],
        MemoryManagementTags::default(),
    )?;
    let stats_b = allocated[0].buffer();
    let surfels_b = allocated[1].buffer();
    let nodes_b = allocated[2].buffer();
    let discovered_b = allocated[3].buffer();
    let scratch_b = allocated[4].buffer();
    let queries_b = allocated[5].buffer();
    let pipelines = tree_pipelines(
        gpu.device.clone(),
        stats_b.clone(),
        surfels_b.clone(),
        nodes_b.clone(),
        discovered_b.clone(),
        scratch_b.clone(),
        queries_b.clone(),
    )?;

    let mut surfels = vec![surfel_at(0, 0); N];
    for i in 0..4 {
        surfels[i] = surfel_at(i as u32, 0);
        surfels[i].instance_id = 10 + i as u32;
    }
    surfels[1].flags = DEAD;
    let half = N / 2;
    surfels[half] = surfel_at(50, 0);
    surfels[half].instance_id = 70;
    surfels[half].position_x = 50.0;
    surfels[half + 1] = surfel_at(60, 0);
    surfels[half + 1].instance_id = 80;
    surfels[half + 1].position_x = 60.0;
    let survivors = [surfels[0], surfels[2], surfels[3]];

    write_case(
        &stats_b,
        &surfels_b,
        &nodes_b,
        &scratch_b,
        &queries_b,
        Stats {
            total: N as i32,
            unordered: 2,
            live: 0,
            high_water: 4,
            reserve: 9,
            discovered: 3,
            pad2: 0,
            pad3: 0,
        },
        &surfels,
        &[query_at(0.0, 0.0, 0.0, -1.0, 0)],
    )?;

    let cmd = PrimaryCommandBuffer::new(gpu.pool.clone(), Some("commit_cb"))?;
    cmd.record_one_time_submit(|recorder| {
        dispatch_build(recorder, &pipelines, true);
    })?;
    let fence = Fence::new(gpu.device.clone(), false, Some("commit_fence"))?;
    let buffers: Vec<Arc<dyn CommandBufferTrait>> = vec![cmd];
    drop(gpu.queue.submit(buffers.as_slice(), &[], &[], fence)?);

    let out = {
        let map = MemoryMap::new(surfels_b.get_backing_memory_pool())?;
        let range = map.range::<SurfelGpu>(surfels_b.clone() as Arc<dyn MemoryPoolBacked>)?;
        range.as_slice()[..N].to_vec()
    };
    assert_eq!(out[0].instance_id, survivors[0].instance_id);
    assert_eq!(out[0].position_x.to_bits(), survivors[0].position_x.to_bits());
    assert_eq!(out[2].instance_id, survivors[1].instance_id);
    assert_eq!(out[3].instance_id, survivors[2].instance_id);
    assert_eq!(out[1].instance_id, 70);
    assert_eq!(out[1].position_x, 50.0);
    assert_eq!(out[1].flags & DEAD, 0);
    assert_eq!(out[4].instance_id, 80);
    assert_eq!(out[4].position_x, 60.0);

    let stats = {
        let map = MemoryMap::new(stats_b.get_backing_memory_pool())?;
        let range = map.range::<Stats>(stats_b.clone() as Arc<dyn MemoryPoolBacked>)?;
        *range
    };
    assert_eq!(stats.unordered, 0);
    assert_eq!(stats.high_water, 5);
    assert_eq!(stats.live, 5);
    assert_eq!(stats.reserve, 0);

    let discovered = {
        let map = MemoryMap::new(discovered_b.get_backing_memory_pool())?;
        let range = map.range::<u32>(discovered_b.clone() as Arc<dyn MemoryPoolBacked>)?;
        range.as_slice()[..8182].to_vec()
    };
    assert_eq!(discovered[0], MISSING);
    assert_eq!(discovered[100], MISSING);

    let nodes = {
        let map = MemoryMap::new(nodes_b.get_backing_memory_pool())?;
        let range = map.range::<BvhNode>(nodes_b.clone() as Arc<dyn MemoryPoolBacked>)?;
        range.as_slice().to_vec()
    };
    let live = live_indices(&out, 5);
    assert_eq!(live, vec![0, 1, 2, 3, 4]);
    assert_tree(&nodes, &out, &live);
    Ok(())
}

fn write_case(
    stats_b: &Arc<vulkan_framework::buffer::AllocatedBuffer>,
    surfels_b: &Arc<vulkan_framework::buffer::AllocatedBuffer>,
    nodes_b: &Arc<vulkan_framework::buffer::AllocatedBuffer>,
    scratch_b: &Arc<vulkan_framework::buffer::AllocatedBuffer>,
    queries_b: &Arc<vulkan_framework::buffer::AllocatedBuffer>,
    stats: Stats,
    surfels: &[SurfelGpu],
    queries: &[QueryItem],
) -> VulkanResult<()> {
    {
        let map = MemoryMap::new(stats_b.get_backing_memory_pool())?;
        let mut range = map.range::<Stats>(stats_b.clone() as Arc<dyn MemoryPoolBacked>)?;
        *range = stats;
    }
    {
        let map = MemoryMap::new(surfels_b.get_backing_memory_pool())?;
        let mut range = map.range::<SurfelGpu>(surfels_b.clone() as Arc<dyn MemoryPoolBacked>)?;
        range.as_mut_slice()[..surfels.len()].copy_from_slice(surfels);
    }
    {
        let map = MemoryMap::new(nodes_b.get_backing_memory_pool())?;
        let mut range = map.range::<u32>(nodes_b.clone() as Arc<dyn MemoryPoolBacked>)?;
        range.as_mut_slice().fill(0);
    }
    {
        let map = MemoryMap::new(scratch_b.get_backing_memory_pool())?;
        let mut range = map.range::<u32>(scratch_b.clone() as Arc<dyn MemoryPoolBacked>)?;
        range.as_mut_slice().fill(0);
    }
    {
        let map = MemoryMap::new(queries_b.get_backing_memory_pool())?;
        let mut range = map.range::<QueryItem>(queries_b.clone() as Arc<dyn MemoryPoolBacked>)?;
        range.as_mut_slice()[..queries.len()].copy_from_slice(queries);
    }
    Ok(())
}

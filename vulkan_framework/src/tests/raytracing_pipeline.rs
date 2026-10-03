use crate::raytracing_pipeline::RaytracingPipeline;

fn shader_slots(group: &ash::vk::RayTracingShaderGroupCreateInfoKHR) -> [u32; 4] {
    [
        group.general_shader,
        group.closest_hit_shader,
        group.any_hit_shader,
        group.intersection_shader,
    ]
}

#[test]
fn simple_pipeline_keeps_raygen_miss_and_closest_hit_group_indices() {
    let unused = ash::vk::SHADER_UNUSED_KHR;
    let groups = RaytracingPipeline::shader_group_create_infos(None, None, None);
    assert_eq!(groups.len(), 3);
    assert_eq!(groups[0].ty, ash::vk::RayTracingShaderGroupTypeKHR::GENERAL);
    assert_eq!(shader_slots(&groups[0]), [0, unused, unused, unused]);
    assert_eq!(groups[1].ty, ash::vk::RayTracingShaderGroupTypeKHR::GENERAL);
    assert_eq!(shader_slots(&groups[1]), [1, unused, unused, unused]);
    assert_eq!(
        groups[2].ty,
        ash::vk::RayTracingShaderGroupTypeKHR::TRIANGLES_HIT_GROUP
    );
    assert_eq!(shader_slots(&groups[2]), [unused, 2, unused, unused]);
}

#[test]
fn every_optional_group_combination_has_correct_types_slots_and_order() {
    let unused = ash::vk::SHADER_UNUSED_KHR;
    for intersection in [false, true] {
        for any_hit in [false, true] {
            for callable in [false, true] {
                let mut next_stage = 3;
                let mut take_stage = |present: bool| {
                    present.then(|| {
                        let index = next_stage;
                        next_stage += 1;
                        index
                    })
                };
                let intersection_stage = take_stage(intersection);
                let any_hit_stage = take_stage(any_hit);
                let callable_stage = take_stage(callable);
                let groups = RaytracingPipeline::shader_group_create_infos(
                    intersection_stage,
                    any_hit_stage,
                    callable_stage,
                );
                assert_eq!(
                    groups.len(),
                    3 + usize::from(intersection) + usize::from(any_hit) + usize::from(callable)
                );
                assert_eq!(shader_slots(&groups[2]), [unused, 2, unused, unused]);
                let mut group_index = 3;
                if let Some(stage) = intersection_stage {
                    let group = &groups[group_index];
                    assert_eq!(
                        group.ty,
                        ash::vk::RayTracingShaderGroupTypeKHR::PROCEDURAL_HIT_GROUP
                    );
                    assert_eq!(shader_slots(group), [unused, unused, unused, stage]);
                    group_index += 1;
                }
                if let Some(stage) = any_hit_stage {
                    let group = &groups[group_index];
                    assert_eq!(
                        group.ty,
                        ash::vk::RayTracingShaderGroupTypeKHR::TRIANGLES_HIT_GROUP
                    );
                    assert_eq!(shader_slots(group), [unused, unused, stage, unused]);
                    group_index += 1;
                }
                if let Some(stage) = callable_stage {
                    let group = &groups[group_index];
                    assert_eq!(group.ty, ash::vk::RayTracingShaderGroupTypeKHR::GENERAL);
                    assert_eq!(shader_slots(group), [stage, unused, unused, unused]);
                    group_index += 1;
                }
                assert_eq!(groups.len(), group_index);
            }
        }
    }
}

pub mod final_rendering;
pub mod global_illumination;
pub mod hdr_transform;
pub mod mesh_rendering;
pub mod renderquad;

use std::sync::Arc;

use vulkan_framework::{
    descriptor_pool::{
        DescriptorPool, DescriptorPoolConcreteDescriptor, DescriptorPoolSizesConcreteDescriptor,
    },
    descriptor_set::DescriptorSet,
    descriptor_set_layout::DescriptorSetLayout,
    device::DeviceOwned,
    image::ImageLayout,
    image_view::ImageView,
    sampler::Sampler,
};

fn sampled_image_descriptor_set(
    layout: Arc<DescriptorSetLayout>,
    image: Arc<ImageView>,
    sampler: Arc<Sampler>,
    name: &str,
) -> crate::rendering::RenderingResult<Arc<DescriptorSet>> {
    // A GPU semaphore wait cannot protect host updates to a descriptor set in use.
    // The recorder retains this immutable set, its pool and its image until reset.
    let pool = DescriptorPool::new(
        layout.get_parent_device(),
        DescriptorPoolConcreteDescriptor::new(
            DescriptorPoolSizesConcreteDescriptor::new(0, 1, 0, 0, 0, 0, 0, 0, 0, None),
            1,
        ),
        Some(name),
    )?;
    let set = DescriptorSet::new(pool, layout)?;
    set.bind_resources(|binder| {
        binder
            .bind_combined_images_samplers(
                0,
                &[(ImageLayout::ShaderReadOnlyOptimal, image, sampler)],
            )
            .unwrap();
    })?;
    Ok(set)
}

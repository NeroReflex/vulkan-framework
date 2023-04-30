use std::sync::Arc;

use crate::{compute_pipeline::ComputePipeline, pipeline_layout::PipelineLayout, descriptor_set::DescriptorSet, buffer::BufferTrait};


// TODO: it would be better for performance to use smallvec...
pub struct ResourcesInUseByGPU {
    pipeline_layouts: Vec<Arc<PipelineLayout>>,
    compute_pipelines: Vec<Arc<ComputePipeline>>,
    descriptor_sets: Vec<Arc<DescriptorSet>>,
    buffers: Vec<Arc<dyn BufferTrait>>,
}

impl ResourcesInUseByGPU {
    pub fn create() -> Self {
        Self {
            pipeline_layouts: vec![],
            compute_pipelines: vec![],
            descriptor_sets: vec![],
            buffers: vec![]
        }
    }

    pub fn register_buffer_usage(&mut self, buffer: Arc<dyn BufferTrait>) {
        // TODO: think about having lots of copies of the same object and how it affect memory usage and performance
        self.buffers.push(buffer)
    }

    pub fn register_descriptor_set_usage(&mut self, descriptor_set: Arc<DescriptorSet>) {
        // TODO: think about having lots of copies of the same object and how it affect memory usage and performance
        self.descriptor_sets.push(descriptor_set)
    }

    pub fn register_pipeline_layout_usage(&mut self, pipeline_layout: Arc<PipelineLayout>) {
        // TODO: think about having lots of copies of the same object and how it affect memory usage and performance
        self.pipeline_layouts.push(pipeline_layout)
    }

    pub fn register_compute_pipeline_usage(&mut self, compute_pipeline: Arc<ComputePipeline>) {
        // TODO: think about having lots of copies of the same object and how it affect memory usage and performance
        self.compute_pipelines.push(compute_pipeline)
    }
}
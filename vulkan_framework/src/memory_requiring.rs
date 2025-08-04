use crate::{buffer::Buffer, image::Image};

#[derive(Debug, PartialEq, Clone)]
pub struct MemoryRequirements {
    memory_type_bits: u32,
    size: u64,
    alignment: u64,
}

impl MemoryRequirements {
    pub fn new(memory_type_bits: u32, size: u64, alignment: u64) -> Self {
        Self {
            memory_type_bits,
            size,
            alignment,
        }
    }

    pub fn memory_type_bits(&self) -> u32 {
        self.memory_type_bits.to_owned()
    }

    pub fn size(&self) -> u64 {
        self.size.to_owned()
    }

    pub fn alignment(&self) -> u64 {
        self.alignment.to_owned()
    }
}

pub trait MemoryRequiring {
    fn memory_requirements(&self) -> MemoryRequirements;
}

#[derive(Debug)]
pub enum UnallocatedResource {
    Buffer(Buffer),
    Image(Image),
}

impl UnallocatedResource {
    pub fn memory_requirements(&self) -> MemoryRequirements {
        let memory_requiring = match self {
            UnallocatedResource::Buffer(buffer) => buffer as &dyn MemoryRequiring,
            UnallocatedResource::Image(image) => image as &dyn MemoryRequiring,
        };

        memory_requiring.memory_requirements()
    }
}

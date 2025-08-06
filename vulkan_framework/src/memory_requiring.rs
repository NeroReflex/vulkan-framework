#[derive(Debug, PartialEq, Clone)]
pub struct AllocationRequirements {
    memory_type_bits: u32,
    size: u64,
    alignment: u64,
}

impl AllocationRequirements {
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

pub trait AllocationRequiring {
    fn allocation_requirements(&self) -> AllocationRequirements;
}

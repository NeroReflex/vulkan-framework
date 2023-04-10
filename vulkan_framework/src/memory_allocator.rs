pub struct MemoryAllocation<'allocator, AllocatorType>
where
    AllocatorType: MemoryAllocator
{
    allocator: &'allocator AllocatorType,
    required_size: u64,
    required_alignment: u64,
    resulting_address: u64,
}

impl<'allocator, AllocatorType> MemoryAllocation<'allocator, AllocatorType>
where
    AllocatorType: 'allocator + MemoryAllocator
{
    pub fn new(allocator: &'allocator AllocatorType, size: u64, alignment: u64, result: u64) -> Self {
        Self {
            allocator,
            required_size: size,
            required_alignment: alignment,
            resulting_address: result
        }
    }
}

pub trait MemoryAllocator {

    /**
     * Get the amount of memory managed by the current memory allocator.
     */
    fn total_size(&self) -> u64;

    /**
     * Allocates memory and tracks allocation in the current memory allocator.
     * 
     * @param size the memory required size (in bytes) to allocate
     * @param alignment the memory required alignment (in bytes)
     */
    fn alloc(&self, size: u64, alignment: u64) -> Option<MemoryAllocation<'static, Self>> where Self: Sized;

    fn dealloc(&self, ptr: MemoryAllocation<'static, Self>) where Self: Sized;
}
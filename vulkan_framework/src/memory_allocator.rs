pub struct MemoryAllocation<'allocator, AllocatorType>
where
    AllocatorType: MemoryAllocator
{
    allocator: &'allocator AllocatorType,
    required_size: usize,
    required_alignment: usize,
    resulting_address: usize,
}

impl<'allocator, AllocatorType> MemoryAllocation<'allocator, AllocatorType>
where
    AllocatorType: 'allocator + MemoryAllocator
{
    pub fn new(allocator: &'allocator AllocatorType, size: usize, alignment: usize, result: usize) -> Self {
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
     * Allocates memory and tracks allocation in the current memory allocator.
     * 
     * @param size the memory required size (in bytes) to allocate
     * @param alignment the memory required alignment (in bytes)
     */
    fn alloc(size: usize, alignment: usize) -> Option<MemoryAllocation<'static, Self>> where Self: Sized;

    fn dealloc(ptr: MemoryAllocation<'static, Self>) where Self: Sized;
}
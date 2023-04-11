pub struct MemoryAllocation<'allocator, AllocatorType>
where
    AllocatorType: MemoryAllocator,
{
    allocator: &'allocator AllocatorType,
    required_size: u64,
    required_alignment: u64,
    resulting_address: u64,
}

impl<'allocator, AllocatorType> Drop for MemoryAllocation<'allocator, AllocatorType>
where
    AllocatorType: 'allocator + MemoryAllocator,
{
    fn drop(&mut self) {
        todo!()
        //TODO: dealloc here
        //self.allocator.dealloc(&self);
    }
}

impl<'allocator, AllocatorType> MemoryAllocation<'allocator, AllocatorType>
where
    AllocatorType: 'allocator + MemoryAllocator,
{
    pub fn new(
        allocator: &'allocator AllocatorType,
        size: u64,
        alignment: u64,
        result: u64,
    ) -> Self {
        Self {
            allocator,
            required_size: size,
            required_alignment: alignment,
            resulting_address: result,
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
    fn alloc<'alloc_result>(&self, size: u64, alignment: u64) -> Option<MemoryAllocation<'alloc_result, Self>>
    where
        Self: Sized;

    fn dealloc<'a>(&self/*, ptr: &'a MemoryAllocation<'static, Self>*/)
    where
        Self: Sized;
}


pub struct StackAllocator {
    total_size: u64,

}

impl StackAllocator {
    pub fn new(total_size: u64) -> Self {
        Self { total_size }
    }
}

impl MemoryAllocator for StackAllocator {
    fn total_size(&self) -> u64 {
        self.total_size
    }

    fn alloc<'alloc_result>(&self, size: u64, alignment: u64) -> Option<MemoryAllocation<'alloc_result, Self>>
    where
        Self: Sized {
            None
        }

    fn dealloc<'a>(&self/*, ptr: &'a MemoryAllocation<'static, Self>*/)
    where
        Self: Sized {

        }
}
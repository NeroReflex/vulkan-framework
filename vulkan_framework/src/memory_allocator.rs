use std::sync::{Mutex, atomic::{AtomicU64, Ordering}};

pub struct AllocationResult {
    requested_size: u64,
    requested_alignment: u64,
    resulting_address: u64,
    allocation_start: u64,
    allocation_end: u64,
}

impl AllocationResult {
    pub fn offset_in_pool(&self) -> u64 {
        self.resulting_address
    }

    pub fn size(&self) -> u64 {
        self.requested_size
    }

    pub fn new(
        requested_size: u64,
        requested_alignment: u64,
        resulting_address: u64,
        allocation_start: u64,
        allocation_end: u64,
    ) -> Self {
        Self {
            requested_size,
            requested_alignment,
            resulting_address,
            allocation_start,
            allocation_end,
        }
    }
}

pub trait MemoryAllocator: Sync + Send {
    /**
     * Get the amount of memory managed by the current memory allocator.
     */
    fn total_size(&self) -> u64;

    /**
     * Allocates memory and tracks allocation in the current memory allocator.
     *
     * This method possibly returns an instance representing the performed allocation,
     * whoever has the ownership of the returned object holds the ownership of the allocated memory.
     *
     * @param size the memory required size (in bytes) to allocate
     * @param alignment the memory required alignment (in bytes)
     */
    fn alloc(&self, size: u64, alignment: u64) -> Option<AllocationResult>;

    /**
     * Deallocates memory and tracks deallocation in the current memory allocator.
     *
     * This method would like to take (back) the ownership of the allocation result returned to the alloc method,
     * this is because, as stated in the alloc method whoever owns the allocation result owns the
     * memory associated with it and therefore for the deallocation to be safe that object ownership
     * must return to the current memory allocator where can be re-used for subsequent allocations.
     *
     * However that is not really possible: drop cannot move out its members.
     */
    fn dealloc(&self, allocation: &mut AllocationResult);
}

pub struct StackAllocator {
    total_size: u64,
    allocated_size: AtomicU64,
}

impl StackAllocator {
    pub fn new(total_size: u64) -> Self {
        Self {
            total_size,
            allocated_size: AtomicU64::new(0),
        }
    }
}

impl MemoryAllocator for StackAllocator {
    fn total_size(&self) -> u64 {
        self.total_size
    }

    fn alloc(&self, size: u64, alignment: u64) -> Option<AllocationResult> {
        let mut previous_allocation_end = 0;

        loop {
            let allocation_start = previous_allocation_end;
            let allocated_number_of_aligned_blocks = allocation_start / alignment;
            let mut padding_to_respect_aligment = ((allocated_number_of_aligned_blocks + 1)
                * alignment)
                - (allocated_number_of_aligned_blocks * alignment);
            if padding_to_respect_aligment == alignment {
                padding_to_respect_aligment = 0;
            }
            let allocation_end = allocation_start + padding_to_respect_aligment + size;

            match self.allocated_size.compare_exchange(allocation_start, allocation_end, Ordering::Acquire, Ordering::Acquire) {
                Ok(prev) => {
                    return Some(AllocationResult::new(
                        size,
                        alignment,
                        allocation_start + padding_to_respect_aligment,
                        allocation_start,
                        allocation_end,
                    ))
                },
                Err(current) => {
                    previous_allocation_end = current
                }
            }
        }
    }

    fn dealloc(&self, allocation_to_undo: &mut AllocationResult) {
        loop {
            match self.allocated_size.compare_exchange(allocation_to_undo.allocation_end, allocation_to_undo.allocation_start, Ordering::Acquire, Ordering::Acquire) {
                Ok(_) => {
                    break
                },
                Err(current) => {
                    println!("Error in resource deallocation: out-of-order deallocation detected! I was expecting to deallocate memory that ends at address {}, instead the memory that ends at address {} has yet to be deallocated", allocation_to_undo.allocation_end, current);
                }
            }
        }
    }
}

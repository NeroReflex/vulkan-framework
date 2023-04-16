use std::sync::{Mutex};

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

pub trait MemoryAllocator {
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
    allocated_size: Mutex<u64>,
}

impl StackAllocator {
    pub fn new(total_size: u64) -> Self {
        Self {
            total_size,
            allocated_size: Mutex::new(0),
        }
    }
}

impl MemoryAllocator for StackAllocator {
    fn total_size(&self) -> u64 {
        self.total_size
    }

    fn alloc(&self, size: u64, alignment: u64) -> Option<AllocationResult> {
        match self.allocated_size.lock() {
            Ok(mut allocated_size_guard) => {
                let allocation_start: u64 = *allocated_size_guard;
                let allocated_number_of_aligned_blocks = allocation_start / alignment;
                let mut padding_to_respect_aligment = ((allocated_number_of_aligned_blocks + 1)
                    * alignment)
                    - (allocated_number_of_aligned_blocks * alignment);
                if padding_to_respect_aligment == alignment {
                    padding_to_respect_aligment = 0;
                }

                let allocation_end = allocation_start + padding_to_respect_aligment + size;

                match self.total_size >= allocation_end {
                    true => {
                        *allocated_size_guard += allocation_end - allocation_start;
                        Some(AllocationResult::new(
                            size,
                            alignment,
                            allocation_start + padding_to_respect_aligment,
                            allocation_start,
                            allocation_end,
                        ))
                    }
                    false => None,
                }
            }
            Err(err) => {
                #[cfg(debug_assertions)]
                {
                    println!("Error acquiring internal mutex: {}", err);
                    assert_eq!(true, false)
                }

                Option::None
            }
        }
    }

    fn dealloc(&self, allocation_to_undo: &mut AllocationResult) {
        match self.allocated_size.lock() {
            Ok(mut allocated_size_guard) => {
                let allocation_start: u64 = *allocated_size_guard;

                if allocation_to_undo.allocation_end == allocation_start {
                    *allocated_size_guard -=
                        allocation_to_undo.allocation_end - allocation_to_undo.allocation_start;
                } else {
                    #[cfg(debug_assertions)]
                    {
                        panic!("This is a stack allocator and cannot handle out-of-order memory deallocations!")
                    }
                }

                allocation_to_undo.allocation_end = 0;
                allocation_to_undo.allocation_start = 0;
                allocation_to_undo.requested_alignment = 0;
                allocation_to_undo.requested_size = 0;
                allocation_to_undo.resulting_address = 0;
            }
            Err(err) => {
                #[cfg(debug_assertions)]
                {
                    println!("Error acquiring internal mutex: {}", err);
                    println!("Given memory will be lost forever");
                    assert_eq!(true, false)
                }
            }
        }
    }
}

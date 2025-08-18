use std::sync::atomic::{AtomicU64, Ordering};

#[cfg(feature = "better_mutex")]
use parking_lot::{const_mutex, Mutex};

#[cfg(not(feature = "better_mutex"))]
use std::sync::Mutex;

pub struct SuccessfulAllocation {
    requested_size: u64,
    requested_alignment: u64,
    resulting_address: u64,
    allocation_start: u64,
    allocation_end: u64,
}

impl SuccessfulAllocation {
    pub fn offset_in_pool(&self) -> u64 {
        self.resulting_address
    }

    pub fn size(&self) -> u64 {
        self.requested_size
    }

    pub fn requested_alignment(&self) -> u64 {
        self.requested_alignment
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
    fn alloc(&self, size: u64, alignment: u64) -> Option<SuccessfulAllocation>;

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
    fn dealloc(&self, allocation: &mut SuccessfulAllocation);
}

pub struct DefaultAllocator {
    management_array: Mutex<smallvec::SmallVec<[u64; 4096]>>,
    total_size: u64,
    block_size: u64,
}

impl DefaultAllocator {
    pub fn with_blocksize(block_size: u64, number_of_blocks: u64) -> Self {
        #[cfg(debug_assertions)]
        {
            println!(
                "Managing {} blocks of {} bytes each",
                number_of_blocks, block_size
            );
        }

        let protected_resource = (0..(((number_of_blocks / 64) + 1) as usize))
            .map(|_idx| 0u64)
            .collect::<smallvec::SmallVec<[u64; 4096]>>();

        #[cfg(feature = "better_mutex")]
        let management_array = const_mutex(protected_resource);

        #[cfg(not(feature = "better_mutex"))]
        let management_array = Mutex::new(protected_resource);

        Self {
            management_array,
            total_size: number_of_blocks * block_size,
            block_size,
        }
    }

    pub fn new(total_size: u64) -> Self {
        let block_size = 4096u64;
        let number_of_blocks = total_size / block_size;

        #[cfg(debug_assertions)]
        {
            println!(
                "Managing {} blocks of {} bytes each",
                number_of_blocks, block_size
            );
        }

        Self::with_blocksize(block_size, number_of_blocks)
    }

    fn flag_used(mem_tracker: &mut [u64], used: bool, start_block: u64) {
        let index = start_block / 64u64;
        let bit_number = start_block % 64u64;
        let bitmask = 1u64 << bit_number;

        match used {
            true => mem_tracker[index as usize] |= bitmask,
            false => mem_tracker[index as usize] &= !bitmask,
        };
    }

    fn flag_blocks(mem_tracker: &mut [u64], used: bool, start_block: u64, blocks_count: u64) {
        for block in start_block..(start_block + blocks_count) {
            Self::flag_used(mem_tracker, used, block)
        }
    }

    fn check_free(mem_tracker: &[u64], start_block: u64) -> bool {
        let index = start_block / 64u64;
        let bit_number = start_block % 64u64;
        let bitmask = 1u64 << bit_number;

        (mem_tracker[index as usize] & bitmask) == 0u64
    }

    fn check_free_blocks(mem_tracker: &[u64], start_block: u64, blocks_count: u64) -> u64 {
        let mut used_blocks = 0u64;
        for block in start_block..(start_block + blocks_count) {
            if !Self::check_free(mem_tracker, block) {
                used_blocks += 1;
            }
        }

        used_blocks
    }
}

impl MemoryAllocator for DefaultAllocator {
    fn total_size(&self) -> u64 {
        self.total_size
    }

    fn alloc(&self, size: u64, alignment: u64) -> Option<SuccessfulAllocation> {
        let required_number_of_blocks = 2 + (size / self.block_size);
        let total_number_of_blocks = self.total_size / self.block_size;

        if total_number_of_blocks < required_number_of_blocks {
            return None;
        }

        let last_useful_first_allocation_block = total_number_of_blocks - required_number_of_blocks;

        #[cfg(feature = "better_mutex")]
        let mut lck = self.management_array.lock();

        #[cfg(not(feature = "better_mutex"))]
        let mut lck = match self.management_array.lock() {
            Ok(lock) => lock,
            Err(err) => {
                dbg!("Error locking mutex: memory won't be allocated -- {}", err);

                return None;
            }
        };

        let mut i: u64 = 0;
        'find_first_block: while i < last_useful_first_allocation_block {
            // if the start of the block is aligned then this will be 0 as I don't have to
            // waste the first block for aligning the rest, otherwise it will be 1
            // (meaning the first block if used to align the data)
            let block_offset_alignment = if ((i * self.block_size) % alignment) == 0 {
                0
            } else {
                1
            };

            let next_aligned_start_addr = ((i * self.block_size) / alignment) * alignment;

            // a boolean representing the fact that the current block contains a suitable start
            let contains_aligned_start = next_aligned_start_addr >= (i * self.block_size)
                && (next_aligned_start_addr < ((i + 1u64) * self.block_size));

            // if the address range represented by + and i+1 do not contains the aligned address
            // this block is skipped
            if !contains_aligned_start {
                i += 1;
                continue 'find_first_block;
            }

            let required_number_of_blocks =
                block_offset_alignment + 1u64 + (size / self.block_size);

            // make sure there is enough room to allocate the memory
            if (i + required_number_of_blocks) >= total_number_of_blocks {
                return None;
            }

            // make sure the requested memory is free
            let used_blocks = Self::check_free_blocks(lck.as_slice(), i, required_number_of_blocks);
            if used_blocks > 0 {
                i += used_blocks;
                continue 'find_first_block;
            }

            // found a suitable set of blocks: set them as occupied and retun the allocated memory
            Self::flag_blocks(lck.as_mut_slice(), true, i, required_number_of_blocks);

            // early drop the mutex lock when not needed anymore
            drop(lck);

            let allocation_start = i * self.block_size;
            let allocation_end =
                (i * self.block_size) + (required_number_of_blocks * self.block_size);

            return Some(SuccessfulAllocation::new(
                size,
                alignment,
                next_aligned_start_addr,
                allocation_start,
                allocation_end,
            ));
        }

        None
    }

    fn dealloc(&self, allocation: &mut SuccessfulAllocation) {
        let first_block = allocation.allocation_start / self.block_size;
        let number_of_allocated_blocks =
            (allocation.allocation_end / self.block_size) - first_block;

        if (first_block + number_of_allocated_blocks) > (self.total_size / self.block_size) {
            panic!("Memory was not allocated from this pool! :O");
        }

        #[cfg(feature = "better_mutex")]
        let mut lck = self.management_array.lock();

        #[cfg(not(feature = "better_mutex"))]
        let mut lck = match self.management_array.lock() {
            Ok(lock) => lock,
            Err(err) => {
                dbg!("Error in locking mutex: memory will be lost -- {}", err);

                return;
            }
        };

        let used_blocks =
            Self::check_free_blocks(lck.as_slice(), first_block, number_of_allocated_blocks);

        if used_blocks != number_of_allocated_blocks {
            panic!("Memory was not allocated from this pool! :O");
        }

        Self::flag_blocks(
            lck.as_mut_slice(),
            false,
            first_block,
            number_of_allocated_blocks,
        );
    }
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

    fn alloc(&self, size: u64, alignment: u64) -> Option<SuccessfulAllocation> {
        let mut previous_allocation_end = 0;

        loop {
            let allocation_start = previous_allocation_end;
            let allocated_number_of_aligned_blocks = allocation_start / alignment;
            let padding_to_respect_aligment = ((allocated_number_of_aligned_blocks + 1)
                * alignment)
                - (allocated_number_of_aligned_blocks * alignment);
            let padding_to_respect_aligment = if padding_to_respect_aligment == alignment {
                0
            } else {
                padding_to_respect_aligment
            };

            let allocation_end = allocation_start + padding_to_respect_aligment + size;

            if allocation_end > self.total_size {
                return None;
            }

            match self.allocated_size.compare_exchange(
                allocation_start,
                allocation_end,
                Ordering::Acquire,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    return Some(SuccessfulAllocation::new(
                        size,
                        alignment,
                        allocation_start + padding_to_respect_aligment,
                        allocation_start,
                        allocation_end,
                    ))
                }
                Err(current) => previous_allocation_end = current,
            }
        }
    }

    fn dealloc(&self, allocation_to_undo: &mut SuccessfulAllocation) {
        loop {
            match self.allocated_size.compare_exchange(
                allocation_to_undo.allocation_end,
                allocation_to_undo.allocation_start,
                Ordering::Acquire,
                Ordering::Acquire,
            ) {
                Ok(_) => break,
                Err(current) => {
                    println!("Error in resource deallocation: out-of-order deallocation detected! I was expecting to deallocate memory that ends at address {}, instead the memory that ends at address {} has yet to be deallocated", allocation_to_undo.allocation_end, current);
                }
            }
        }
    }
}

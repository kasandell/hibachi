/// # Constants with reserved meanings in Hibachi

/// In a given tensor shape, Hibachi reserves the `0th` dimension for batching
pub const BATCH_DIM: usize = 0;

/// In a given tensor shape, Hibachi reserves the `1st` dimension for sequence
pub const SEQ_DIM: usize = 1;

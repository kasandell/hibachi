use std::fmt::{Debug, Display};


/// # Backend
///
/// Core trait that must be implemented by any tensor backend that supports batching operations.
///
/// This trait defines the minimum set of operations required for a tensor implementation
/// to be compatible with the library. A Backend represents an n-dimensional tensor with
/// basic shape manipulation and transformation capabilities.
///
/// ## Requirements
///
/// The implementing type must also satisfy the following traits:
/// - `Debug`: For debugging output
/// - `Display`: For string representation
/// - `Clone`: For creating copies of tensors
/// - `Send` and `Sync`: For safe concurrent access across threads
/// - `'static`: No borrowed references in the type
///
/// ## Implementation Note
///
/// When implementing this trait, ensure all operations maintain dimension
/// consistency appropriate for tensor operations, particularly when operating
/// across dimensions.
pub trait Backend: Debug + Display + Clone + Send + Sync + 'static {
    /// Returns the shape of this tensor as a vector of dimension sizes.
    ///
    /// The returned vector contains the size of each dimension, where
    /// the first element is the outermost dimension.
    ///
    /// # Returns
    ///
    /// * `Vec<usize>` - A vector containing the size of each dimension
    fn shape(&self) -> Vec<usize>;

    /// Concatenates multiple tensors along the specified dimension.
    ///
    /// # Parameters
    ///
    /// * `tensors` - Slice of tensors to concatenate
    /// * `dim` - The dimension along which to concatenate (0-indexed)
    ///
    /// # Returns
    ///
    /// A new tensor containing the concatenated data
    ///
    /// # Panics
    ///
    /// This function may panic if:
    /// * `dim` is out of bounds for any of the tensors
    /// * The tensors have incompatible shapes outside of the concatenation dimension
    fn cat(tensors: &[Self], dim: usize) -> Self;

    /// Performs element-wise equality comparison against another tensor.
    ///
    /// # Parameters
    ///
    /// * `other` - The tensor to compare against (must have the same shape)
    ///
    /// # Returns
    ///
    /// A new boolean tensor where each element is `true` if the corresponding
    /// elements in the two input tensors are equal, `false` otherwise.
    ///
    /// # Panics
    ///
    /// May panic if the shapes of `self` and `other` are not identical.
    fn eq(&self, other: &Self) -> impl Backend;

    /// Splits a tensor into a vector of size-1 slices along the specified dimension.
    ///
    /// This is equivalent to splitting the tensor into individual slices along `dim`.
    ///
    /// # Parameters
    ///
    /// * `dim` - The dimension along which to split (0-indexed)
    ///
    /// # Returns
    ///
    /// A vector of tensors, each representing a slice of size 1 along the specified dimension
    ///
    /// # Panics
    ///
    /// May panic if `dim` is out of bounds for this tensor.
    fn vectorize_dim(&self, dim: usize) -> Vec<Self>;

    /// Extracts a contiguous slice from the tensor along the specified dimension.
    ///
    /// # Parameters
    ///
    /// * `dimension` - The dimension to slice along (0-indexed)
    /// * `seq_start_idx` - Starting index for the slice
    /// * `len` - Length of the slice to extract
    ///
    /// # Returns
    ///
    /// A new tensor containing the extracted slice
    ///
    /// # Panics
    ///
    /// May panic if:
    /// * `dimension` is out of bounds for this tensor
    /// * `seq_start_idx + len` exceeds the size of the specified dimension
    fn slice(&self, dimension: usize, seq_start_idx: usize, len: usize) -> Self;

    /// Returns indices along a dimension where all values in the corresponding slice are `true`.
    ///
    /// # Parameters
    ///
    /// * `dim` - The dimension to check (0-indexed)
    ///
    /// # Returns
    ///
    /// A vector of indices where all elements in the slice at that index are `true`
    ///
    /// # Panics
    ///
    /// May panic if:
    /// * `dim` is out of bounds for this tensor
    /// * The tensor does not contain boolean data
    fn idx_where_all_true(&self, dim: usize) -> Vec<usize>;

    /// Removes a single slice at the specified index from the tensor along the given dimension.
    ///
    /// # Parameters
    ///
    /// * `dim` - The dimension from which to remove the slice (0-indexed)
    /// * `index` - The index of the slice to remove
    ///
    /// # Returns
    ///
    /// A new tensor with the specified slice removed
    ///
    /// # Panics
    ///
    /// May panic if:
    /// * `dim` is out of bounds for this tensor
    /// * `index` is out of bounds for the specified dimension
    fn pop(&self, dim: usize, index: usize) -> Self;

    /// Repeats the tensor a specified number of times along the given dimension.
    ///
    /// # Parameters
    ///
    /// * `dim` - The dimension along which to repeat (0-indexed)
    /// * `times` - How many times to repeat the tensor
    ///
    /// # Returns
    ///
    /// A new tensor with the repeated data
    ///
    /// # Panics
    ///
    /// May panic if `dim` is out of bounds for this tensor.
    fn repeat(&self, dim: usize, times: usize) -> Self;
}

/// # Unsqueezeable
///
/// Trait for tensors that can be "unsqueezed" to add a dimension.
///
/// ## Purpose
///
/// This trait exists to handle the common tensor operation of adding a dimension
/// of size 1, which increases the rank of the tensor. It's separated from the
/// main `Backend` trait to avoid infinite recursion in implementations that use
/// const generics to track tensor dimensions.
///
/// ## Background
///
/// For backends using const generics (like those with types such as `Tensor<N>` where
/// `N` is the rank/dimensionality), implementing unsqueeze operations directly in the
/// `Backend` trait would require the implementing type to handle an unbounded range
/// of ranks, which isn't possible with const generics.
///
/// The pattern used here allows backends to implement `Backend` for tensors up to some
/// reasonable maximum rank (e.g., rank 10), and implement `LowerRankedTensorOps` for
/// ranks 0 through 9, ensuring type safety while supporting unsqueeze operations.
pub trait Unsqueezable: Backend {
    /// The tensor type resulting from an unsqueeze operation, which has one more
    /// dimension than `Self`.
    type Unsqueezed: Backend;

    /// Adds a dimension of size 1 at the specified position.
    ///
    /// # Parameters
    ///
    /// * `dim` - The position at which to insert the new dimension (0-indexed)
    ///
    /// # Returns
    ///
    /// A tensor of type `Self::Unsqueezed` with the new dimension added
    ///
    /// # Panics
    ///
    /// May panic if `dim` is greater than the current number of dimensions.
    fn unsqueeze(&self, dim: usize) -> Self::Unsqueezed;
}

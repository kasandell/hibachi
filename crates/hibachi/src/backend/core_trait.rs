use std::fmt::{Debug, Display};


/// The trait that must be fulfilled by any backend to support batching
pub trait LowerRankedTensorOps: Backend {
    /// the type we unsqueeze to
    type Unsqueezed: Backend;

    /// Unsqueeze the tensor along dimension `dim` with size 1
    fn unsqueeze(&self, dim: usize) -> Self::Unsqueezed;
}

/// The backend trait that must be fulfilled by any backend to support batching
pub trait Backend:  Debug + Display + Clone + Send + Sync + 'static {
    /// Return the shape of this tensor
    fn shape(&self) -> Vec<usize>;

    /// Concatenate several tensors to each other along dimension `dim`, in the order supplied
    fn cat(tensors: &[Self], dim: usize) -> Self;

    /// Element-wise equality check against another tensor of equal shape
    fn eq(&self, other: &Self) -> impl Backend;

    /// Slice a tensor into a vector size `1` tensors along the supplied `dim`
    fn vectorize_dim(&self, dim: usize) -> Vec<Self>;

    /// Slice a given `dimension` from `seq_start_idx` to `seq_start_idx + len`
    fn slice(&self, dimension: usize, seq_start_idx: usize, len: usize) -> Self;

    /// Return the indices along `dim` wherein the size `1` slice of each `i` of dim is all `true`
    fn idx_where_all_true(&self, dim: usize) -> Vec<usize>;

    /// Pop the given `index` slice from the tensor in dimension `dim`
    fn pop(&self, dim: usize, index: usize) -> Self;

    /// Repeat a given tensor `times` along `dim`
    fn repeat(&self, dim: usize, times: usize) -> Self;
}

use std::fmt::{Debug, Display};


pub trait LowerRankedTensorOps: Backend {
    type Unsqueezed: Backend;
    fn unsqueeze(&self, dim: usize) -> Self::Unsqueezed;
}

pub trait Backend:  Debug + Display + Clone + Send + Sync + 'static {
    fn shape(&self) -> Vec<usize>;
    fn cat(tensors: &[Self], dim: usize) -> Self;
    fn eq(&self, other: &Self) -> impl Backend;
    fn vectorize_dim(&self, dim: usize) -> Vec<Self>;
    fn slice(&self, dimension: usize, seq_start_idx: usize, len: usize) -> Self;
    /// collapse the tensor down to a rank 1 tensor, where
    /// by slicing on dim, all elems are true,
    fn idx_where_all_true(&self, dim: usize) -> Vec<usize>;
    fn pop(&self, dim: usize, index: usize) -> Self;
    fn repeat(&self, dim: usize, times: usize) -> Self;
}

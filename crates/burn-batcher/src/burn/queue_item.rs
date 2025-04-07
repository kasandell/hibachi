use burn::prelude::{
    Backend,
    Tensor
};
use tokio::sync::mpsc;

/// Queue item struct to store both the input tensor and the sender for results
pub(crate) struct QueueItem<B>//<B, const S_IN: usize, const S_OUT: usize>
where B: Backend {
    pub(crate) input: Tensor<B, 2>,
    pub(crate) sender: mpsc::UnboundedSender<Tensor<B, 1>>
}

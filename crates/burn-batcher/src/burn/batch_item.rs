use burn::prelude::{
    Backend,
    Tensor
};
use tokio::sync::mpsc;

/// Queue item struct to store both the input tensor and the sender for results
pub(crate) struct BatchItem<B>//<B, const S: usize>
where B: Backend {
    pub(crate) slot: usize,
    pub(crate) sequence_length: usize,
    pub(crate) sender: mpsc::UnboundedSender<Tensor<B, 1>>
}

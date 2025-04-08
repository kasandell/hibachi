use burn::prelude::{
    Backend,
    Tensor
};
use tokio::sync::mpsc;

/// Queue item struct to store both the input tensor and the sender for results
pub(crate) struct QueueItem<B>
where B: Backend {
    pub(crate) input: Tensor<B, 1>,
    pub(crate) sender: mpsc::UnboundedSender<Tensor<B, 1>>
}

use candle_core::{Tensor};
use tokio::sync::mpsc;

/// Queue item struct to store both the input tensor and the sender for results
pub(crate) struct QueueItem {
    pub(crate) input: Tensor,
    pub(crate) sender: mpsc::UnboundedSender<Tensor>
}

use tokio::sync::mpsc;
use candle_core::Tensor;

/// Queue item struct to store both the input tensor and the sender for results
pub(crate) struct BatchItem {
    pub(crate) slot: usize,
    pub(crate) sequence_length: usize,
    pub(crate) sender: mpsc::UnboundedSender<Tensor>
}


impl PartialEq for BatchItem {
    fn eq(&self, other: &Self) -> bool {
        self.slot == other.slot
    }
}


pub(crate) fn max_seq_len_for_batch_items<const S: usize>(
    batch_items: &mut [Option<BatchItem>; S]
) -> usize {
    batch_items.iter()
        .filter_map(|item| item.as_ref().map(|bi| bi.sequence_length))
        .max()
        .unwrap_or(0)
}

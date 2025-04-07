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


impl <B> PartialEq for BatchItem<B>
where B: Backend {
    fn eq(&self, other: &Self) -> bool {
        self.slot == other.slot
    }
}


pub(crate) fn max_seq_len_for_batch_items<B, const S: usize>(
    batch_items: &mut [Option<BatchItem<B>>; S]
) -> usize
where B: Backend{
    batch_items.iter()
        .filter_map(|item| item.as_ref().map(|bi| bi.sequence_length))
        .max()
        .unwrap_or(0)
}

use tokio::sync::mpsc;
use uuid::{Uuid};

/// Batch item struct to store the sender info, and the sequence tracking info per batch slot
pub struct BatchItem<T> {
    id: Uuid,
    sequence_length: usize,
    sender: mpsc::UnboundedSender<T>
}

impl <T> BatchItem<T> {
    pub fn new(sequence_length: usize, sender: mpsc::UnboundedSender<T>) -> Self {
        Self {
            id: Uuid::new_v4(),
            sequence_length,
            sender,
        }
    }

    pub fn sender(&self) -> &mpsc::UnboundedSender<T> {
        &self.sender
    }

    pub fn increment_sequence_length(&mut self, amount: usize) {
        self.sequence_length += amount
    }

    pub fn max_seq_len_for_batch_items(
        batch_items: &[BatchItem<T>]
    ) -> usize {
        batch_items.iter()
            .map(|item| item.sequence_length)
            .max()
            .unwrap_or(0)
    }

    pub fn len(&self) -> usize {
        self.sequence_length
    }
}


impl <T> PartialEq for BatchItem<T> {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}




use tokio::sync::mpsc;
use uuid::{Uuid};

/// Batch item struct to store the sender info, and the sequence tracking info per batch slot.
/// This item is intended to be kept in a vector, where its index represents the index in
/// a processing tensor, for which we'd want to slice the data and send through this item.
#[derive(Debug, Clone)]
pub struct BatchItem<T> {
    id: Uuid,
    sequence_length: usize,
    sender: mpsc::UnboundedSender<T>
}

impl <T> BatchItem<T> {
    /// Initialize a batch item with a sender and unspecified sequence length.
    /// `sequence_length` is not guaranteed to be 0, if say, the starting sequence has tokens in it already
    pub fn new(sequence_length: usize, sender: mpsc::UnboundedSender<T>) -> Self {
        Self {
            id: Uuid::new_v4(),
            sequence_length,
            sender,
        }
    }

    /// Retrieve a copy of the batch item sender, through which we can dispatch messages
    pub fn sender(&self) -> &mpsc::UnboundedSender<T> {
        &self.sender
    }

    /// Increment the tracked sequence length by a given `amount`
    pub fn increment_sequence_length(&mut self, amount: usize) {
        self.sequence_length += amount
    }

    /// Given a slice of [BatchItem], return the max sequence length contained by any of them
    /// or default to 0
    pub fn max_seq_len_for_batch_items(
        batch_items: &[BatchItem<T>]
    ) -> usize {
        batch_items.iter()
            .map(|item| item.len())
            .max()
            .unwrap_or(0)
    }

    /// The length of a batch item is defined as its sequence length.
    /// This sequence length will correspond 1:1 with the length of the
    /// sequence - padding in an actively running tensor
    pub fn len(&self) -> usize {
        self.sequence_length
    }
}


impl <T> PartialEq for BatchItem<T> {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}


#[cfg(test)]
mod test {
    use tokio::test;
    use tokio::sync::mpsc::unbounded_channel;
    use crate::communcation::BatchItem;

    #[test]
    async fn test_eq() {

        let (tx, _rx) = unbounded_channel::<i32>();
        let item1 = BatchItem::new(0, tx.clone());
        let item2 = BatchItem::new(0, tx.clone());

        assert_ne!(item1, item2, "equality should fail");
        assert_eq!(item1, item1, "equality should pass for item2");
        assert_eq!(item2, item2, "equality should pass for item2");
    }

    #[test]
    async fn test_max_batch_item_len_empty() {
        assert_eq!(0, BatchItem::<i32>::max_seq_len_for_batch_items(&[]), "Max len should be 0");
    }

    #[test]
    async fn test_increment_sequence_length() {
        let (tx, _rx) = unbounded_channel::<i32>();
        let mut item = BatchItem::new(0, tx.clone());
        assert_eq!(0, item.len());
        item.increment_sequence_length(100);
        assert_eq!(100, item.len());
    }

    #[test]
    async fn test_max_batch_item_len() {
        let (tx, _rx) = unbounded_channel::<i32>();
        let ranges = (0..=100).map(|e| {
            BatchItem::new(e, tx.clone())
        }).collect::<Vec<_>>();

        assert_eq!(100, BatchItem::max_seq_len_for_batch_items(&ranges));
    }
}

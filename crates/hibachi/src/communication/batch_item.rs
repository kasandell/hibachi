use tokio::sync::mpsc;
use tokio::sync::mpsc::UnboundedSender;
use uuid::Uuid;

/// # BatchItem
///
/// A container for managing individual elements within a processing batch.
///
/// `BatchItem` stores the necessary information to:
/// 1. Uniquely identify each item in a batch (`id`)
/// 2. Track the current sequence length for this item (`sequence_length`)
/// 3. Provide a communication channel for sending processed results (`sender`)
///
/// ## Usage Context
///
/// In tensor batch processing, each `BatchItem` represents a slot in the batch.
/// The index of the `BatchItem` in a collection typically corresponds to its
/// position in a processing tensor, allowing for efficient slicing and routing
/// of processed data.
///
/// ## Thread Safety
///
/// This struct uses Tokio's unbounded channels for communication, making it
/// suitable for asynchronous processing across task boundaries.
///
/// Note: This struct is crate-private and not intended for external use.
#[derive(Debug, Clone)]
pub struct BatchItem<T> {
    /// Unique identifier for this batch item
    id: Uuid,

    /// Current length of the sequence associated with this batch item
    #[allow(dead_code)]
    sequence_length: usize,

    /// Channel for sending processed results back to the requester
    sender: mpsc::UnboundedSender<T>
}

impl<T> BatchItem<T> {
    /// Creates a new `BatchItem` with the specified initial sequence length and sender.
    ///
    /// # Parameters
    ///
    /// * `sequence_length` - Initial sequence length, which may be non-zero if
    ///   the sequence already contains tokens
    /// * `sender` - Channel for sending processed results
    ///
    /// # Returns
    ///
    /// A new `BatchItem` instance with a randomly generated UUID
    #[allow(dead_code)]
    pub fn new(sequence_length: usize, sender: mpsc::UnboundedSender<T>) -> Self {
        Self {
            id: Uuid::new_v4(),
            sequence_length,
            sender,
        }
    }

    /// Returns a reference to the sender channel for dispatching messages.
    ///
    /// # Returns
    ///
    /// A reference to the unbounded sender channel
    #[allow(dead_code)]
    pub fn sender(&self) -> &mpsc::UnboundedSender<T> {
        &self.sender
    }

    /// Increases the tracked sequence length by the specified amount.
    ///
    /// This is typically called when new tokens are added to the sequence
    /// associated with this batch item.
    ///
    /// # Parameters
    ///
    /// * `amount` - The number of tokens/elements to add to the sequence length
    #[allow(dead_code)]
    pub fn increment_sequence_length(&mut self, amount: usize) {
        self.sequence_length += amount
    }

    /// Returns the current sequence length for this batch item.
    ///
    /// The sequence length corresponds to the number of tokens or elements
    /// in the sequence associated with this batch item.
    ///
    /// # Returns
    ///
    /// The current sequence length
    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.sequence_length
    }

    /// Checks if the sequence length is zero.
    ///
    /// # Returns
    ///
    /// `true` if the sequence length is zero, `false` otherwise
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.sequence_length == 0
    }

    /// Returns the unique identifier for this batch item.
    ///
    /// # Returns
    ///
    /// The UUID of this batch item
    #[allow(dead_code)]
    pub fn id(&self) -> Uuid {
        self.id
    }

    /// Finds the maximum sequence length across a collection of batch items.
    ///
    /// This is useful for determining padding requirements in batched processing.
    ///
    /// # Parameters
    ///
    /// * `batch_items` - A slice of `BatchItem` instances to examine
    ///
    /// # Returns
    ///
    /// The maximum sequence length found, or 0 if the collection is empty
    #[allow(dead_code)]
    pub fn max_seq_len_for_batch_items(
        batch_items: &[BatchItem<T>]
    ) -> usize {
        batch_items.iter()
            .map(|item| item.len())
            .max()
            .unwrap_or(0)
    }
}

impl<T> PartialEq for BatchItem<T> {
    /// Compares two `BatchItem` instances for equality based on their unique IDs.
    ///
    /// Note that equality is determined solely by the UUID, not by sequence length
    /// or the sender channels.
    ///
    /// # Parameters
    ///
    /// * `other` - Another `BatchItem` to compare with
    ///
    /// # Returns
    ///
    /// `true` if both batch items have the same ID, `false` otherwise
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl<T> Eq for BatchItem<T> {}

impl<T> AsRef<UnboundedSender<T>> for BatchItem<T> {
    fn as_ref(&self) -> &UnboundedSender<T> {
        &self.sender
    }
}

#[cfg(test)]
mod test {
    use tokio::test;
    use super::*;
    use tokio::sync::mpsc::unbounded_channel;

    #[test]
    async fn test_new_creates_unique_ids() {
        let (tx, _rx) = unbounded_channel::<i32>();
        let item1 = BatchItem::new(0, tx.clone());
        let item2 = BatchItem::new(0, tx);

        assert_ne!(item1.id(), item2.id(), "New batch items should have unique IDs");
    }

    #[test]
    async fn test_equality_based_on_id() {
        let (tx, _rx) = unbounded_channel::<i32>();
        let item1 = BatchItem::new(0, tx.clone());
        let item2 = BatchItem::new(0, tx.clone());
        let item1_clone = item1.clone();

        assert_ne!(item1, item2, "Different batch items should not be equal");
        assert_eq!(item1, item1_clone, "Cloned batch items should be equal");
        assert_eq!(item1, item1, "A batch item should equal itself");
        assert_eq!(item2, item2, "A batch item should equal itself");
    }

    #[test]
    async fn test_max_seq_len_for_empty_collection() {
        assert_eq!(
            0,
            BatchItem::<i32>::max_seq_len_for_batch_items(&[]),
            "Max length of empty collection should be 0"
        );
    }

    #[test]
    async fn test_max_seq_len_for_single_item() {
        let (tx, _rx) = unbounded_channel::<i32>();
        let item = BatchItem::new(42, tx);

        assert_eq!(
            42,
            BatchItem::max_seq_len_for_batch_items(&[item]),
            "Max length of single item collection should be the item's length"
        );
    }

    #[test]
    async fn test_max_seq_len_for_multiple_items() {
        let (tx, _rx) = unbounded_channel::<i32>();
        let items = vec![
            BatchItem::new(5, tx.clone()),
            BatchItem::new(10, tx.clone()),
            BatchItem::new(3, tx.clone())
        ];

        assert_eq!(
            10,
            BatchItem::max_seq_len_for_batch_items(&items),
            "Max length should be the largest sequence length among items"
        );
    }

    #[test]
    async fn test_increment_sequence_length() {
        let (tx, _rx) = unbounded_channel::<i32>();
        let mut item = BatchItem::new(0, tx);

        assert_eq!(0, item.len(), "Initial length should be 0");

        item.increment_sequence_length(5);
        assert_eq!(5, item.len(), "Length should be 5 after incrementing by 5");

        item.increment_sequence_length(3);
        assert_eq!(8, item.len(), "Length should be 8 after incrementing by 3");
    }

    #[test]
    async fn test_is_empty() {
        let (tx, _rx) = unbounded_channel::<i32>();
        let mut item = BatchItem::new(0, tx);

        assert!(item.is_empty(), "New item with length 0 should be empty");

        item.increment_sequence_length(1);
        assert!(!item.is_empty(), "Item with length > 0 should not be empty");
    }

    #[test]
    async fn test_sender_sends_messages() {
        let (tx, mut rx) = unbounded_channel::<String>();
        let item = BatchItem::new(0, tx);

        // Send a message through the batch item's sender
        item.sender().send("test message".to_string()).unwrap();

        // Verify the message was received
        let received = rx.recv().await;
        assert!(received.is_some(), "Should receive a message");
        assert_eq!("test message", received.unwrap(), "Should receive the correct message");
    }

    #[test]
    async fn test_clone_creates_equal_item() {
        let (tx, _rx) = unbounded_channel::<i32>();
        let original = BatchItem::new(5, tx);
        let cloned = original.clone();

        assert_eq!(original, cloned, "Cloned item should equal original");
        assert_eq!(original.len(), cloned.len(), "Cloned item should have same length");
        assert_eq!(original.id(), cloned.id(), "Cloned item should have same ID");
    }

    #[test]
    async fn test_as_ref_implementation() {
        let (tx, mut rx) = mpsc::unbounded_channel::<i32>();
        let queue_item = BatchItem::new(0, tx);

        // Use the AsRef implementation to get a reference to the sender
        let sender_ref: &UnboundedSender<i32> = queue_item.as_ref();

        // Send a message using the reference
        sender_ref.send(42).unwrap();

        // Verify we can receive the message
        let received = rx.recv().await;
        assert_eq!(received, Some(42), "Should receive message sent through the sender reference");
    }
}

// src/queue_item.rs

//! # Queue Item Module
//!
//! This module provides the `QueueItem` struct, a generic container for managing
//! items in an asynchronous processing queue with Tokio.
//!
//! The `QueueItem` struct is designed to track items as they move through a processing
//! pipeline, maintaining metadata like sequence length and providing a communication
//! channel for results.
//!
//! ## Features
//!
//! - Generic over the input type
//! - Unique identification via UUIDs
//! - Sequence length tracking
//! - Built-in communication channel for results
//! - Utility methods for batch processing
//!

use tokio::sync::mpsc;
use tokio::sync::mpsc::UnboundedSender;
use uuid::Uuid;

/// A generic container for items in a processing queue with built-in result channels.
///
/// `QueueItem<T>` wraps input data of type `T` with:
/// - A unique identifier
/// - A sequence length counter (useful for tracking tokens in sequence generation)
/// - A sender channel for returning results to the requester
///
/// This struct is particularly useful in asynchronous processing pipelines where:
/// - Items need to be uniquely identified
/// - Progress needs to be tracked (via sequence length)
/// - Results need to be communicated back to the original requester
///
/// The generic parameter `T` represents both the input type and the type
/// that will be sent back through the channel.
#[derive(Debug, Clone)]
pub struct QueueItem<T> {
    /// Unique identifier for this queue item
    id: Uuid,

    /// The input data to be processed
    #[allow(dead_code)]
    input: T,

    /// Current length of the sequence associated with this queue item
    ///
    /// This is typically used to track the number of tokens in text generation
    /// or similar sequence-based processing.
    #[allow(dead_code)]
    sequence_length: usize,

    /// Channel for sending processed results back to the requester
    ///
    /// This channel allows the processing system to communicate results
    /// back to whoever created this queue item.
    sender: mpsc::UnboundedSender<T>
}

impl<T> QueueItem<T> {
    /// Creates a new `QueueItem` with the specified input, initial sequence length, and sender channel.
    ///
    /// This is the main constructor for creating queue items to be processed.
    ///
    /// # Parameters
    ///
    /// * `input` - The data to be processed
    /// * `sequence_length` - Initial sequence length, which may be non-zero if
    ///   the sequence already contains tokens
    /// * `sender` - Channel for sending processed results back to the requester
    ///
    /// # Returns
    ///
    /// A new `QueueItem` instance with a randomly generated UUID

    #[allow(dead_code)]
    pub fn new(input: T, sequence_length: usize, sender: mpsc::UnboundedSender<T>) -> Self {
        Self {
            input,
            id: Uuid::new_v4(),
            sequence_length,
            sender,
        }
    }

    /// Returns a reference to the sender channel for dispatching results.
    ///
    /// This method provides access to the channel for sending results back
    /// to the original requester.
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
    /// This is typically called when new tokens or elements are added to the sequence
    /// associated with this queue item, such as during text generation.
    ///
    /// # Parameters
    ///
    /// * `amount` - The number of tokens/elements to add to the sequence length
    #[allow(dead_code)]
    pub fn increment_sequence_length(&mut self, amount: usize) {
        self.sequence_length += amount
    }

    /// Returns the current sequence length for this queue item.
    ///
    /// The sequence length corresponds to the number of tokens or elements
    /// in the sequence associated with this queue item.
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
    /// This is useful for determining if processing has started yet.
    ///
    /// # Returns
    ///
    /// `true` if the sequence length is zero, `false` otherwise
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.sequence_length == 0
    }

    /// Returns the unique identifier for this queue item.
    ///
    /// This UUID can be used to track the item through the processing pipeline.
    ///
    /// # Returns
    ///
    /// The ID of this queue item
    #[allow(dead_code)]
    pub fn id(&self) -> Uuid {
        self.id
    }

    /// Returns a reference to the input data for this queue item.
    ///
    /// This provides access to the original input that needs to be processed.
    ///
    /// # Returns
    ///
    /// A reference to the input data
    #[allow(dead_code)]
    pub fn input(&self) -> &T {
        &self.input
    }

    /// Finds the maximum sequence length across a collection of queue items.
    ///
    /// This is useful for determining padding requirements in batched processing,
    /// particularly for sequence models.
    ///
    /// # Parameters
    ///
    /// * `batch_items` - A slice of `QueueItem` instances to examine
    ///
    /// # Returns
    ///
    /// The maximum sequence length found, or 0 if the collection is empty
    ///
    #[allow(dead_code)]
    pub fn max_seq_len_for_batch_items(
        batch_items: &[QueueItem<T>]
    ) -> usize {
        batch_items.iter()
            .map(|item| item.len())
            .max()
            .unwrap_or(0)
    }
}

impl<T> PartialEq for QueueItem<T> {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl<T> Eq for QueueItem<T> {}

impl<T> AsRef<UnboundedSender<T>> for QueueItem<T> {
    fn as_ref(&self) -> &UnboundedSender<T> {
        &self.sender
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_new_queue_item() {
        // Setup
        let (tx, mut rx) = mpsc::unbounded_channel::<String>();
        let input = String::from("test input");
        let sequence_length = 5;

        // Create a new QueueItem
        let item = QueueItem::new(input.clone(), sequence_length, tx);

        // Assertions
        assert_eq!(item.len(), sequence_length);
        assert_eq!(*item.input(), input);
        assert!(!item.is_empty());

        // Test sending through the channel
        item.sender().send(String::from("response")).unwrap();

        // Check if the message was received
        let received = rx.recv().await.unwrap();
        assert_eq!(received, "response");
    }

    #[test]
    fn test_increment_sequence_length() {
        // Setup
        let (tx, _rx) = mpsc::unbounded_channel::<String>();
        let mut item = QueueItem::new(String::from("test"), 10, tx);

        // Test incrementing
        item.increment_sequence_length(5);
        assert_eq!(item.len(), 15);

        // Test incrementing again
        item.increment_sequence_length(7);
        assert_eq!(item.len(), 22);
    }

    #[test]
    fn test_is_empty() {
        // Setup
        let (tx, _rx) = mpsc::unbounded_channel::<String>();

        // Test with empty sequence
        let item_empty = QueueItem::new(String::from("test"), 0, tx.clone());
        assert!(item_empty.is_empty());

        // Test with non-empty sequence
        let item_not_empty = QueueItem::new(String::from("test"), 1, tx);
        assert!(!item_not_empty.is_empty());
    }

    #[test]
    fn test_equality() {
        // Setup
        let (tx1, _rx1) = mpsc::unbounded_channel::<String>();
        let (tx2, _rx2) = mpsc::unbounded_channel::<String>();

        let item1 = QueueItem::new(String::from("test1"), 5, tx1);

        // Create a second item with the same ID (need to use internal structure to force this)
        let item2 = QueueItem::new(String::from("test2"), 10, tx2);

        // For this test, we'll just check that an item equals itself
        assert_eq!(item1, item1);
        // And different items should not be equal
        assert_ne!(item1, item2);
    }

    #[test]
    fn test_max_seq_len_for_batch_items() {
        // Setup
        let (tx, _rx) = mpsc::unbounded_channel::<String>();

        let item1 = QueueItem::new(String::from("test1"), 5, tx.clone());
        let item2 = QueueItem::new(String::from("test2"), 10, tx.clone());
        let item3 = QueueItem::new(String::from("test3"), 7, tx);

        let items = vec![item1, item2, item3];

        // Test finding max sequence length
        let max_len = QueueItem::max_seq_len_for_batch_items(&items);
        assert_eq!(max_len, 10);
    }

    #[test]
    fn test_max_seq_len_empty_batch() {
        // Test with empty batch
        let empty_batch: Vec<QueueItem<String>> = vec![];
        let max_len = QueueItem::max_seq_len_for_batch_items(&empty_batch);
        assert_eq!(max_len, 0);
    }

    #[test]
    fn test_as_ref_implementation() {
        // Setup
        let (tx, _rx) = mpsc::unbounded_channel::<String>();
        let item = QueueItem::new(String::from("test"), 5, tx);

        // Test AsRef implementation
        let sender_ref: &mpsc::UnboundedSender<String> = item.as_ref();

        // We can only test that it doesn't panic when used
        // and potentially comparing memory addresses, but that's implementation-specific
        assert!(std::ptr::eq(sender_ref, item.sender()));
    }
}

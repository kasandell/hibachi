use tokio::sync::mpsc::UnboundedSender;

/// # QueueItem
///
/// A container for associating input data with a channel for sending results.
///
/// `QueueItem` pairs an input value with an unbounded sender channel,
/// creating a complete work item that can be processed asynchronously.
/// The input value (of type `Q`) is what needs to be processed, and the
/// sender (for type `T`) is the channel where results should be sent.
///
/// ## Usage Context
///
/// This struct is typically used in asynchronous processing queues
/// where work items are submitted for processing and results are
/// returned through channels to the original requesters.
///
/// ## Type Parameters
///
/// * `Q` - The type of the input value to be processed
/// * `T` - The type of the results that will be sent back
pub struct QueueItem<Q, T> {
    /// The input value to be processed
    #[allow(dead_code)]
    input: Q,

    /// Channel for sending results back to the requester
    sender: UnboundedSender<T>
}

impl<Q, T> QueueItem<Q, T> {
    /// Creates a new `QueueItem` with the specified input and sender.
    ///
    /// # Parameters
    ///
    /// * `input` - The value to be processed
    /// * `sender` - Channel for sending results back to the requester
    ///
    /// # Returns
    ///
    /// A new `QueueItem` instance
    #[allow(dead_code)]
    pub fn new(input: Q, sender: UnboundedSender<T>) -> Self {
        Self {
            input,
            sender
        }
    }

    /// Returns a reference to the input value.
    ///
    /// # Returns
    ///
    /// A reference to the input that needs to be processed
    #[allow(dead_code)]
    pub fn input(&self) -> &Q {
        &self.input
    }

    /// Returns a clone of the sender channel for this queue item.
    ///
    /// This method clones the sender, which is designed to be cheap
    /// as the `UnboundedSender` is internally an `Arc` and cloning it
    /// only increments the reference count.
    ///
    /// # Returns
    ///
    /// A clone of the sender for sending results
    #[allow(dead_code)]
    pub fn sender(&self) -> UnboundedSender<T> {
        self.sender.clone()
    }
}

impl<Q, T> AsRef<UnboundedSender<T>> for QueueItem<Q, T> {
    /// Implements the `AsRef` trait to allow borrowing the sender directly.
    ///
    /// This enables the `QueueItem` to be treated as a reference to its
    /// sender in contexts where a reference to the sender is needed.
    ///
    /// # Returns
    ///
    /// A reference to the sender channel
    fn as_ref(&self) -> &UnboundedSender<T> {
        &self.sender
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::test;
    use tokio::sync::mpsc::unbounded_channel;

    #[test]
    async fn test_queue_item_new() {
        let (tx, _rx) = unbounded_channel::<String>();
        let input = 42;

        let queue_item = QueueItem::new(input, tx);

        assert_eq!(*queue_item.input(), 42, "Input should match the provided value");
    }

    #[test]
    async fn test_queue_item_sender() {
        let (tx, mut rx) = unbounded_channel::<String>();
        let input = "test_input";

        let queue_item = QueueItem::new(input, tx);

        // Get the sender from the queue item and send a message
        let sender = queue_item.sender();
        sender.send("test message".to_string()).unwrap();

        // Verify we can receive the message
        let received = rx.recv().await;
        assert_eq!(received, Some("test message".to_string()), "Should receive message sent through the sender");
    }

    #[test]
    async fn test_as_ref_implementation() {
        let (tx, mut rx) = unbounded_channel::<i32>();
        let input = "as_ref_test";

        let queue_item = QueueItem::new(input, tx);

        // Use the AsRef implementation to get a reference to the sender
        let sender_ref: &UnboundedSender<i32> = queue_item.as_ref();

        // Send a message using the reference
        sender_ref.send(42).unwrap();

        // Verify we can receive the message
        let received = rx.recv().await;
        assert_eq!(received, Some(42), "Should receive message sent through the sender reference");
    }

    #[test]
    async fn test_with_different_types() {
        // Test with more complex types
        #[derive(Debug, PartialEq)]
        struct InputType {
            value: String,
        }

        #[derive(Debug, PartialEq)]
        struct ResultType {
            processed: bool,
            original: String,
        }

        let (tx, mut rx) = unbounded_channel::<ResultType>();
        let input = InputType { value: "complex input".to_string() };

        let queue_item = QueueItem::new(input, tx);

        // Verify the input can be accessed
        assert_eq!(queue_item.input().value, "complex input", "Should be able to access complex input");

        // Send a result
        queue_item.sender().send(ResultType {
            processed: true,
            original: "complex input".to_string(),
        }).unwrap();

        // Verify we can receive the complex result
        let received = rx.recv().await;
        assert!(received.is_some(), "Should receive a message");
        let result = received.unwrap();
        assert!(result.processed, "Result should be marked as processed");
        assert_eq!(result.original, "complex input", "Result should contain original input");
    }

    #[test]
    async fn test_multiple_senders() {
        let (tx, mut rx) = unbounded_channel::<&str>();
        let input = 100;

        let queue_item = QueueItem::new(input, tx);

        // Get multiple senders from the same queue item
        let sender1 = queue_item.sender();
        let sender2 = queue_item.sender();

        // Send messages from both senders
        sender1.send("from sender 1").unwrap();
        sender2.send("from sender 2").unwrap();

        // Verify we receive both messages
        let received1 = rx.recv().await;
        let received2 = rx.recv().await;

        assert_eq!(received1, Some("from sender 1"), "Should receive message from first sender");
        assert_eq!(received2, Some("from sender 2"), "Should receive message from second sender");
    }
}

use tokio::sync::oneshot::Sender;

/// A container associating input data with a result channel.
///
/// `QueueItem` pairs an input value with a oneshot sender channel, creating a
/// complete work item that can be processed asynchronously. This structure
/// facilitates the pattern where work is submitted by a client and results
/// are delivered back through a channel when processing is complete.
///
/// # Type Parameters
///
/// * `Q` - The type of the input value to be processed
/// * `T` - The type of the results that will be sent back

/// # Usage in Processing Systems
///
/// `QueueItem` is typically used in batch processing systems where:
///
/// 1. A client submits work and receives an awaitable handle (like [`Item`])
/// 2. The work is queued as a `QueueItem` containing the input and a channel
/// 3. A worker processes the input and sends the result through the channel
/// 4. The client receives the result by awaiting their handle
pub struct QueueItem<Q, T> {
    /// The input value to be processed
    input: Q,

    /// Channel for sending results back to the requester
    sender: Sender<T>
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
    pub fn new(input: Q, sender: Sender<T>) -> Self {
        Self {
            input,
            sender
        }
    }

    /// Returns a reference to the input value.
    ///
    /// This method provides read access to the input data that needs to be processed,
    /// allowing workers to perform operations on the input without taking ownership.
    ///
    /// # Returns
    ///
    /// A reference to the input that needs to be processed
    #[allow(dead_code)]
    pub fn input(&self) -> &Q {
        &self.input
    }

    /// Consumes the item and returns its sender channel.
    ///
    /// This method takes ownership of the `QueueItem` and returns the sender channel,
    /// which can be used to send the result back to the requester.
    ///
    /// # Returns
    ///
    /// The oneshot sender channel
    #[allow(dead_code)]
    pub fn sender(self) -> Sender<T> {
        self.sender
    }
}

impl<Q, T> AsRef<Sender<T>> for QueueItem<Q, T> {
    /// Implements the `AsRef` trait to allow borrowing the sender directly.
    ///
    /// This enables the `QueueItem` to be treated as a reference to its
    /// sender in contexts where a reference to the sender is needed without
    /// consuming the entire item.
    ///
    /// # Returns
    ///
    /// A reference to the sender channel
    fn as_ref(&self) -> &Sender<T> {
        &self.sender
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::sync::oneshot;

    #[test]
    fn test_new_queue_item() {
        let (tx, _rx) = oneshot::channel::<String>();
        let input = 42;

        // Create a new queue item
        let queue_item = QueueItem::new(input, tx);

        // Verify input can be accessed
        assert_eq!(*queue_item.input(), 42);
    }

    #[test]
    fn test_input_accessor() {
        let (tx, _rx) = oneshot::channel::<String>();
        let input = vec![1, 2, 3];

        let queue_item = QueueItem::new(input, tx);

        // Verify input reference works correctly
        let input_ref = queue_item.input();
        assert_eq!(input_ref.len(), 3);
        assert_eq!(input_ref[0], 1);
        assert_eq!(input_ref[1], 2);
        assert_eq!(input_ref[2], 3);
    }

    #[test]
    fn test_sender_accessor() {
        let (tx, rx) = oneshot::channel::<String>();
        let input = "test";

        let queue_item = QueueItem::new(input, tx);

        // Get the sender and use it
        let sender = queue_item.sender();
        sender.send("result".into()).unwrap();

        // Verify the value was sent
        let result = rx.blocking_recv().unwrap();
        assert_eq!(result, "result");
    }

    #[test]
    fn test_as_ref_implementation() {
        let (tx, rx) = oneshot::channel::<i32>();
        let input = 100;

        let queue_item = QueueItem::new(input, tx);

        // Get a reference to the sender
        let sender_ref = queue_item.as_ref();

        // Just verify we can get the reference - we can't actually call send()
        // on the reference because send() takes ownership of the sender
        assert!(std::ptr::eq(sender_ref, &queue_item.sender));

        // Now use the actual queue_item to send
        let sender = queue_item.sender();
        sender.send(42).unwrap();

        // Verify the value was received
        let result = rx.blocking_recv().unwrap();
        assert_eq!(result, 42);
    }

    #[tokio::test]
    async fn test_complete_workflow() {
        // Create a channel
        let (tx, rx) = oneshot::channel::<String>();

        // Create a queue item
        let input_data = "process me";
        let queue_item = QueueItem::new(input_data, tx);

        // Process in a task (simulating a worker)
        tokio::spawn(async move {
            // Access the input
            let input_str = queue_item.input();

            // Process it
            let result = format!("{} - processed", input_str);

            // Send the result
            queue_item.sender().send(result).unwrap();
        });

        // Wait for and verify the result
        let result = rx.await.unwrap();
        assert_eq!(result, "process me - processed");
    }
}

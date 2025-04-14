use std::pin::Pin;
use std::task::{Context, Poll};
use futures::Stream;
use tokio::sync::mpsc;

/// # ItemStream
///
/// An asynchronous stream wrapper around a Tokio unbounded channel receiver.
///
/// `ItemStream` implements the `Stream` trait from the `futures` crate,
/// allowing it to be used with stream combinators and async iteration.
/// It primarily serves as a way to adapt Tokio's channel receivers to
/// the `Stream` interface.
///
/// ## Usage Context
///
/// This stream is typically used to receive processed items from a batched
/// processing pipeline. Each item represents a processed result that is
/// ready to be consumed by downstream components.
///
/// ## Implementation Details
///
/// The stream is backed by a Tokio unbounded channel receiver, which means:
/// - It will never block on `poll_next` even if the channel is empty
/// - It will return `None` when all senders are dropped
/// - It has no backpressure mechanism
///
/// Note: As the TODO comment suggests, this might be better placed in
/// an autoregressive-specific module.
pub struct ItemStream<T> {
    /// The underlying channel receiver
    receiver: mpsc::UnboundedReceiver<T>
}

impl<T> ItemStream<T> {
    /// Creates a new `ItemStream` from a Tokio unbounded channel receiver.
    ///
    /// # Parameters
    ///
    /// * `receiver` - A Tokio unbounded channel receiver that will provide the items
    ///
    /// # Returns
    ///
    /// A new `ItemStream` instance wrapping the provided receiver
    pub fn new(receiver: mpsc::UnboundedReceiver<T>) -> Self {
        Self {
            receiver,
        }
    }
}

impl<T> Stream for ItemStream<T> {
    type Item = T;

    /// Attempts to pull out the next value of this stream.
    ///
    /// # Returns
    ///
    /// - `Poll::Ready(Some(item))` if a value is available
    /// - `Poll::Ready(None)` if the stream is exhausted
    /// - `Poll::Pending` if no value is ready yet
    ///
    /// # Implementation
    ///
    /// This method delegates directly to the underlying unbounded receiver.
    /// When all senders are dropped, this method will return `Poll::Ready(None)`.
    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Pin::new(&mut self.get_mut().receiver).poll_recv(cx)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use tokio::test;
    use futures::StreamExt;
    use tokio::sync::mpsc::unbounded_channel;

    #[test]
    async fn test_receives_items_in_order() {
        let (tx, rx) = unbounded_channel::<i32>();
        let mut stream = ItemStream::new(rx);

        // Send some test values
        tx.send(1).unwrap();
        tx.send(2).unwrap();
        tx.send(3).unwrap();

        // Drop the sender to close the stream
        drop(tx);

        // Collect all items from the stream
        let mut received = Vec::new();
        while let Some(item) = stream.next().await {
            received.push(item);
        }

        assert_eq!(received, vec![1, 2, 3], "Stream should receive items in order");
    }

    #[test]
    async fn test_stream_ends_when_sender_dropped() {
        let (tx, rx) = unbounded_channel::<String>();
        let mut stream = ItemStream::new(rx);

        // Send a value
        tx.send("test".to_string()).unwrap();

        // Get the first value
        let first = stream.next().await;
        assert_eq!(first, Some("test".to_string()), "Should receive the sent value");

        // Drop the sender
        drop(tx);

        // The stream should now be closed
        let end = stream.next().await;
        assert_eq!(end, None, "Stream should end when sender is dropped");
    }

    #[test]
    async fn test_empty_stream() {
        let (tx, rx) = unbounded_channel::<u64>();
        let mut stream = ItemStream::new(rx);

        // Drop the sender immediately without sending anything
        drop(tx);

        // The stream should be empty and closed
        let result = stream.next().await;
        assert_eq!(result, None, "Empty stream should return None");
    }

    #[test]
    async fn test_multiple_senders() {
        let (tx1, rx) = unbounded_channel::<&str>();
        let tx2 = tx1.clone();
        let mut stream = ItemStream::new(rx);

        // Send from both senders
        tx1.send("from sender 1").unwrap();
        tx2.send("from sender 2").unwrap();

        // Collect the results
        let first = stream.next().await;
        let second = stream.next().await;

        assert!(first.is_some(), "Should receive first item");
        assert!(second.is_some(), "Should receive second item");

        // Drop only one sender
        drop(tx1);

        // Stream should still be open
        tx2.send("another from sender 2").unwrap();
        let third = stream.next().await;
        assert_eq!(third, Some("another from sender 2"), "Should receive from remaining sender");

        // Drop the last sender
        drop(tx2);

        // Now the stream should be closed
        let end = stream.next().await;
        assert_eq!(end, None, "Stream should end when all senders are dropped");
    }
}

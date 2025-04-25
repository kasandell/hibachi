use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};
use tokio::sync::oneshot;
use tokio::sync::oneshot::Receiver;

/// A Future wrapper around a Tokio oneshot receiver.
///
/// `Item` provides a convenient way to await the result of an asynchronous computation.
/// It wraps a Tokio oneshot channel receiver, allowing it to be used as a Future that
/// can be awaited to obtain the result when it becomes available.
///
/// # Type Parameters
///
/// * `T` - The type of the value that will eventually be received
///
/// # Usage with Batchers
///
/// `Item` is typically returned by batch processing systems to represent
/// a computation that is queued for execution but not yet complete. The caller
/// can await the `Item` to receive the result once processing is finished.
pub struct Item<T> {
    /// The underlying channel receiver that will provide the result
    receiver: oneshot::Receiver<T>
}

impl<T> Item<T> {
    /// Creates a new `Item` from a Tokio oneshot channel receiver.
    ///
    /// # Parameters
    ///
    /// * `receiver` - The oneshot receiver that will provide the result
    ///
    /// # Returns
    ///
    /// A new `Item` instance wrapping the provided receiver
    #[allow(dead_code)]
    pub fn new(receiver: oneshot::Receiver<T>) -> Self {
        Self {
            receiver,
        }
    }
}

impl<T> AsRef<oneshot::Receiver<T>> for Item<T> {
    /// Provides access to the underlying receiver.
    ///
    /// This implementation allows the `Item` to be treated as a reference to its
    /// internal receiver in contexts where a reference to the receiver is needed.
    ///
    /// # Returns
    ///
    /// A reference to the underlying oneshot receiver
    fn as_ref(&self) -> &Receiver<T> {
        &self.receiver
    }
}

impl<T> Future for Item<T> {
    /// The type of value produced when the future completes.
    type Output = Result<T, oneshot::error::RecvError>;

    /// Polls the underlying oneshot receiver to check if the result is available.
    ///
    /// This implementation delegates to the polling behavior of the wrapped oneshot receiver.
    /// It enables the `Item` to be awaited directly in async contexts.
    ///
    /// # Parameters
    ///
    /// * `self` - A pinned mutable reference to this future
    /// * `cx` - The task context for polling
    ///
    /// # Returns
    ///
    /// * `Poll::Pending` if the result is not yet available
    /// * `Poll::Ready(Ok(value))` if a value was received
    /// * `Poll::Ready(Err(error))` if the sender was dropped without sending a value
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        Pin::new(&mut self.get_mut().receiver).poll(cx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::test;
    use tokio::sync::oneshot;

    #[test]
    async fn test_item_successful_receive() {
        // Create a channel and an Item
        let (tx, rx) = oneshot::channel();
        let item = Item::new(rx);

        // Send a value
        tx.send(42).unwrap();

        // Await the Item and verify the result
        let result = item.await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 42);
    }

    #[tokio::test]
    async fn test_item_sender_dropped() {
        // Create a channel and an Item
        let (tx, rx) = oneshot::channel::<i32>();
        let item = Item::new(rx);

        // Drop the sender without sending a value
        drop(tx);

        // Await the Item and verify it returns an error
        let result = item.await;
        assert!(result.is_err());
    }

    #[test]
    async fn test_as_ref() {
        // Create a channel and an Item
        let (tx, rx) = oneshot::channel::<i32>();
        let item = Item::new(rx);

        // Verify AsRef implementation
        let _receiver_ref: &oneshot::Receiver<i32> = item.as_ref();

        // Clean up
        drop(tx);
    }

    #[test]
    async fn test_future_behavior() {
        // Create a channel and an Item
        let (tx, rx) = oneshot::channel();
        let item = Item::new(rx);

        // Send a value after a delay
        let send_task = tokio::spawn(async move {
            tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
            tx.send("hello").unwrap();
        });

        // Await the Item and verify the result
        let result = item.await.unwrap();
        assert_eq!(result, "hello");

        // Ensure the send task completes
        send_task.await.unwrap();
    }
}

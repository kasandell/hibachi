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
/// ## Limitations
///
/// - No built-in backpressure mechanism (uses unbounded channels)
/// - No mechanism to peek at the next item without consuming it
///
/// ## Implementation Details
///
/// The stream is backed by a Tokio unbounded channel receiver, which means:
/// - It will never block on `poll_next` even if the channel is empty
/// - It will return `None` when all senders are dropped
/// - It has no backpressure mechanism
///
/// ## TODO's:
/// Back this with any type that can support a stream.
///
pub struct ItemStream<T> {
    /// The underlying channel receiver
    receiver: mpsc::UnboundedReceiver<T>
}

impl<T> ItemStream<T> {
    /// Creates a new `ItemStream` from a Tokio unbounded channel receiver.
    ///
    /// This is the only way to construct an `ItemStream`. You would typically
    /// create a Tokio unbounded channel using `tokio::sync::mpsc::unbounded_channel()`,
    /// and then pass the receiver to this constructor.
    #[allow(dead_code)]
    pub fn new(receiver: mpsc::UnboundedReceiver<T>) -> Self {
        Self {
            receiver,
        }
    }
}

impl<T> Stream for ItemStream<T> {
    type Item = T;
    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Pin::new(&mut self.get_mut().receiver).poll_recv(cx)
    }
}

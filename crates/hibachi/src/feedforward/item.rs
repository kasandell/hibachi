use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};
use tokio::sync::oneshot;
use tokio::sync::oneshot::Receiver;

/// # Item
///
/// An asynchronous wrapper around a Tokio oneshot receiver.
///
///
pub struct Item<T> {
    /// The underlying channel receiver
    receiver: oneshot::Receiver<T>
}

impl<T> Item<T> {
    /// Creates a new `Item` from a Tokio oneshot channel receiver.
    #[allow(dead_code)]
    pub fn new(receiver: oneshot::Receiver<T>) -> Self {
        Self {
            receiver,
        }
    }
}


impl<T> AsRef<oneshot::Receiver<T>> for Item<T> {
    fn as_ref(&self) -> &Receiver<T> {
       &self.receiver
    }
}

impl<T> Future for Item<T> {
    type Output = Result<T, oneshot::error::RecvError>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        Pin::new(&mut self.get_mut().receiver).poll(cx)
    }
}

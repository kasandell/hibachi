use std::pin::Pin;
use std::task::{Context, Poll};
use futures::Stream;
use tokio::sync::mpsc;


// TODO: this might belong in autoregressive only
/// Asynchronous stream of items to be returned by the batcher
pub struct ItemStream<T> {
    receiver: mpsc::UnboundedReceiver<T>
}

impl <T> ItemStream<T> {
    /// Instantiate a new item stream from a receiver
    pub fn new(receiver: mpsc::UnboundedReceiver<T>) -> Self {
        Self {
            receiver,
        }
    }
}

impl <T> Stream for ItemStream<T> {
    type Item = T;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Pin::new(&mut self.get_mut().receiver).poll_recv(cx)
    }
}

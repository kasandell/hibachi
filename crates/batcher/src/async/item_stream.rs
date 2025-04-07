use std::pin::Pin;
use std::task::{Context, Poll};
use futures::{Stream, StreamExt};
use tokio::sync::mpsc;


/// Asynchronous stream of items to be returned by the batcher
pub struct AsyncItemStream<T> {
    receiver: mpsc::UnboundedReceiver<T>
}

impl <T> AsyncItemStream<T> {
    /// Instantiate a new item stream from a receiver
    pub fn new(receiver: mpsc::UnboundedReceiver<T>) -> Self {
        Self {
            receiver,
        }
    }
}

impl <T> Stream for AsyncItemStream<T> {
    type Item = T;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        // IDK if i need to pin
        Pin::new(&mut self.get_mut().receiver).poll_recv(cx)
        //self.receiver.poll_recv(cx)
    }
}

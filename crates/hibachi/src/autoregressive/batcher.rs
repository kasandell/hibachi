use std::sync::Arc;
use async_trait::async_trait;
use tokio::sync::{mpsc, Mutex, Notify};
use crate::autoregressive::handler::AutoregressiveHandler;
use crate::autoregressive::queue_item::QueueItem;
use super::item_stream::ItemStream;
use crate::communication::Pill;
use crate::backend::{Backend, Unsqueezable};
use crate::core::batch::batch_inference_loop;
use crate::core::worker::BatchWorkerHandle;
use super::{Autoregressive, AutoregressiveBatcher};

pub struct AutoregressiveBatchInference<B, const S: usize>
where B: Backend + Unsqueezable
{
    waiting_requests: Arc<Mutex<Vec<QueueItem<B>>>>,
    handle: BatchWorkerHandle
}

impl <B, const S: usize> AutoregressiveBatchInference<B, S>
where B: Backend + Unsqueezable,
{
    pub fn new<M>(
        model: M,
        stop_token: &B,
        padding_token: &B,
    ) -> Self
    where M: Autoregressive<B> + Send + Sync + 'static,
    {
        let padding_shape = padding_token.shape();
        assert!(!padding_shape.is_empty(), "padding shape must be of rank 1 or higher");
        assert_eq!(padding_shape[0], 1, "padding dimension 1 must be rank 1");
        let waiting_requests = Arc::new(Mutex::new(vec![]));

        let pill = Pill::new();
        let worker_handle = BatchWorkerHandle::new( {
            let waiting_requests = waiting_requests.clone();
            let work_notifier = Arc::new(Notify::new());
            let pc = padding_token.clone();
            let sc = stop_token.clone();

            move |running, notifier| {
                tokio::spawn(async move {
                    let moved_pill = pill;
                    let inference_handler = AutoregressiveHandler{
                        model,
                        padding_token: pc,
                        stop_token: sc,
                    };

                    batch_inference_loop::<AutoregressiveHandler<M, B>, S>(
                        &inference_handler,
                        running,
                        work_notifier,
                        waiting_requests,
                    )
                        .await;
                })
            }
        });

        Self {
            waiting_requests,
            handle: worker_handle,
        }

    }

}

/// Implements resource cleanup for the inference engine.
///
/// This implementation ensures that the background task is properly
/// shut down when the inference engine is dropped.
#[async_trait]
impl <B, const S: usize> AutoregressiveBatcher<B, B> for AutoregressiveBatchInference<B, S>
where B: Backend + Unsqueezable,
{
    /// Processes a generation request and returns a stream of results.
    ///
    /// This method:
    /// 1. Creates a channel for receiving generated tokens
    /// 2. Adds the request to the waiting queue
    /// 3. Notifies the background task that new work is available
    /// 4. Returns a stream that will yield the generated tokens
    ///
    /// # Parameters
    ///
    /// * `item` - The input tensor for which to generate tokens
    ///
    /// # Returns
    ///
    /// An `ItemStream` that yields generated tokens as they become available
    async fn run(&self, item: B) -> ItemStream<B> {
        let (tx, rx) = mpsc::unbounded_channel();
        let size = item.shape()[0];
        let queue_item = QueueItem::new(
            item,
            size,
            tx,
        );
        {
            let mut senders = self.waiting_requests.lock().await;
            senders.push(queue_item);
        }
        // Notify the worker that new work is available
        self.handle.notify();
        ItemStream::new(rx)
    }
}


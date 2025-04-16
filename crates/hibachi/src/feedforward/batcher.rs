use std::marker::PhantomData;
use std::sync::Arc;
use tokio::sync::{oneshot, Mutex, Notify};
use crate::backend::{Backend, Unsqueezable};
use super::queue_item::QueueItem;
use async_trait::async_trait;
use oneshot::channel;
use crate::communication::Pill;
use crate::core::batch::batch_inference_loop;
use crate::core::worker::BatchWorkerHandle;
use crate::feedforward::core_trait::{Feedforward, FeedforwardBatcher};
use crate::feedforward::handler::FeedForwardHandler;
use crate::feedforward::item::Item;

pub struct FeedforwardBatchInference<B, O, const S: usize>
{
    waiting_requests: Arc<Mutex<Vec<QueueItem<B, O>>>>,
    handle: BatchWorkerHandle
}

impl <B, O, const S: usize> FeedforwardBatchInference<B, O, S>
where B: Backend + Unsqueezable, O: Backend
{
    pub fn new<M>(
        model: M,
    ) -> Self
    where M: Feedforward<B, O> + Send + Sync + 'static,
    {

        let waiting_requests = Arc::new(Mutex::new(vec![]));

        let pill = Pill::new();
        let worker_handle = BatchWorkerHandle::new( {
                let waiting_requests = waiting_requests.clone();
                let work_notifier = Arc::new(Notify::new());

                move |running, notifier| {
                    tokio::spawn(async move {
                        let moved_pill = pill;
                        let inference_handler = FeedForwardHandler{
                            _marker: PhantomData,
                            model
                        };

                        batch_inference_loop::<FeedForwardHandler<M, B, O>, S>(
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
impl <B, O, const S: usize> FeedforwardBatcher<B, O> for FeedforwardBatchInference<B, O, S>
where B: Backend, O: Backend
{
    async fn run(&self, item: B) -> Item<O> {
        let (tx, rx) = channel();
        let queue_item = QueueItem::new(
            item,
            tx,
        );
        {
            let mut senders = self.waiting_requests.lock().await;
            senders.push(queue_item);
        }
        // Notify the worker that new work is available
        self.handle.notify();
        Item::new(rx)
    }
}


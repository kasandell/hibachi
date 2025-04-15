use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;
use tokio::sync::{Mutex, Notify};
use tokio::task::JoinHandle;
use tokio::time::error::Elapsed;
use crate::communication::{BatchItem, QueueItem};
use crate::tensor::operations::slice_tensor_by_batch_dimension;

pub struct BatchInference<B, O, const S: usize>
{
    /// Flag indicating whether the background task should continue running
    pub(crate) running: Arc<AtomicBool>,

    /// Handle to the background task that processes batched requests
    pub(crate) background_task: Option<JoinHandle<()>>,

    /// Thread-safe queue of pending requests waiting to be processed
    pub(crate) waiting_requests: Arc<Mutex<Vec<QueueItem<B, O>>>>,

    /// Notification mechanism to signal when new work is available
    pub(crate) work_notifier: Arc<Notify>,

    /// Counter tracking the number of currently active sequences in the batch
    pub(crate) active_count: Arc<Mutex<usize>>
}

impl<B, O, const S: usize> BatchInference<B, O, S> {

    /// Sends model outputs to their respective result streams.
    ///
    /// After a forward pass of the model, this method distributes the generated
    /// tokens to the appropriate requesters through their dedicated channels.
    ///
    /// # Parameters
    ///
    /// * `outputs` - Tensor containing the model's output for all sequences in the batch
    /// * `batch_items` - List of batch items that track sequence state and output channels
    async fn send_outputs_to_stream(
        outputs: O,
        batch_items: &Vec<BatchItem<O>>
    ) {
        let sliced = slice_tensor_by_batch_dimension(outputs);
        batch_items.iter().enumerate().for_each(|(idx, item)| {
            match sliced.get(idx) {
                None => {
                    panic!("Unable to slice index {}", idx);
                }
                Some(slice) => {
                    match item.sender().send(slice.clone()) {
                        Ok(_) => {}
                        Err(_) => {
                            // TODO: we should close the slot and let someone else take
                            eprintln!("Failed to send output to stream; receiver likely dropped");
                        }
                    }
                }
            }
        });
    }

    /// Extracts pending requests that can be processed in the current batch.
    ///
    /// This method determines how many new requests can be added to the batch
    /// based on the current batch occupancy and maximum size, then moves those
    /// requests from the waiting queue to the active state.
    ///
    /// # Parameters
    ///
    /// * `batch_size` - Maximum number of sequences that can be processed in a batch
    /// * `waiting_requests` - Queue of pending requests
    /// * `active_count` - Counter of currently active sequences
    ///
    /// # Returns
    ///
    /// A vector of queue items that should be added to the active batch
    async fn drain_possible_requests(
        batch_size: usize,
        waiting_requests: Arc<Mutex<Vec<QueueItem<B, O>>>>,
        active_count: Arc<Mutex<usize>>,
    ) -> Vec<QueueItem<B, O>> {
        // TODO: this is probably inefficient on locking. we really just want to peek at batch size and active count
        // before we even get to the requests lock
        let mut requests = waiting_requests.lock().await;
        let mut active = active_count.lock().await;
        // Calculate how many items we can process
        let available_slots = batch_size.saturating_sub(*active);

        if available_slots > 0 && !requests.is_empty() {
            let items_to_take = std::cmp::min(available_slots, requests.len());
            // Take a batch of requests
            let batch = requests.drain(0..items_to_take).collect::<Vec<_>>();
            // Update active count
            *active += items_to_take;
            return batch;
        }
        vec![]
    }

    /// Returns the current number of active sequences being processed.
    ///
    /// This can be used to monitor the load on the batching system and
    /// to implement backpressure mechanisms if needed.
    ///
    /// # Returns
    ///
    /// The number of sequences currently being processed in the batch
    pub async fn active_count(&self) -> usize {
        *self.active_count.clone().lock().await
    }

    /// Waits for a notification with a timeout.
    ///
    /// This helper method attempts to wait for a notification from the work notifier,
    /// but will time out after a specified duration to allow for periodic checking
    /// of other conditions (like the running flag).
    ///
    /// # Parameters
    ///
    /// * `notifier` - The notify instance to wait on
    ///
    /// # Returns
    ///
    /// `Ok(())` if a notification was received, or `Err(Elapsed)` if the timeout occurred
    #[inline]
    async fn timeout_await_notifier(notifier: &Notify) -> Result<(), Elapsed>{
        tokio::time::timeout(
            Duration::from_millis(100),
            notifier.notified()
        ).await
    }

    /// Determines whether there is work to be processed.
    ///
    /// This helper method checks if there are either:
    /// 1. Active sequences currently being processed, or
    /// 2. Waiting requests that could be started
    ///
    /// # Parameters
    ///
    /// * `active_count` - Shared counter of active sequences
    /// * `waiting_requests` - Queue of pending requests
    ///
    /// # Returns
    ///
    /// `true` if there is work to process, `false` otherwise
    #[inline]
    async fn should_process(
        active_count: Arc<Mutex<usize>>,
        waiting_requests: Arc<Mutex<Vec<QueueItem<B, B>>>>
    ) -> bool {
        let active = active_count.lock().await;
        let has_active_items = *active > 0;

        if has_active_items {
            true
        } else {
            let waiting = waiting_requests.lock().await;
            !waiting.is_empty()
        }
    }

    /// Initiates a graceful shutdown of the batch inference engine.
    ///
    /// This method signals the background task to stop processing and
    /// awaits its completion.
    ///
    /// # Implementation Notes
    ///
    /// The shutdown process:
    /// 1. Sets the running flag to false
    /// 2. Notifies the worker to check the flag
    /// 3. Spawns a task to await the background task's completion
    fn shutdown(&mut self) {
        // Signal the background task to stop
        self.running.store(false, Ordering::SeqCst);
        // Notify the worker to wake up and check the running flag
        self.work_notifier.notify_one();

        // Await the background task if it exists
        if let Some(task) = self.background_task.take() {
            tokio::spawn(async move {
                let _ = task.await;
            });
        }
    }
}


/// Implements resource cleanup for the inference engine.
///
/// This implementation ensures that the background task is properly
/// shut down when the inference engine is dropped.
impl <B, O, const S: usize> Drop for BatchInference<B, O, S>
{
    fn drop(&mut self) {
        self.shutdown();
    }
}


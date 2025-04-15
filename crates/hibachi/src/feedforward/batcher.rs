use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;
use tokio::sync::{oneshot, Mutex, Notify};
use tokio::task::JoinHandle;
use tokio::time::error::Elapsed;
use crate::autoregressive::Autoregressive;
use crate::backend::{Backend, Unsqueezable};
use crate::communication::{BatchItem, Pill};
use super::queue_item::QueueItem;
use async_trait::async_trait;
use oneshot::channel;
use crate::feedforward::core_trait::{Feedforward, FeedforwardBatcher};
use crate::feedforward::item::Item;
use crate::tensor::operations::{add_sequence_to_outside_of_slot, slice_tensor_by_batch_dimension};

pub struct FeedforwardBatchInference<B, O, const S: usize>
where B: Backend, O: Backend
{
    /// Flag indicating whether the background task should continue running
    running: Arc<AtomicBool>,

    /// Handle to the background task that processes batched requests
    background_task: Option<JoinHandle<()>>,

    /// Thread-safe queue of pending requests waiting to be processed
    waiting_requests: Arc<Mutex<Vec<QueueItem<B, O>>>>,

    /// Notification mechanism to signal when new work is available
    work_notifier: Arc<Notify>,

    /// Counter tracking the number of currently active sequences in the batch
    active_count: Arc<Mutex<usize>>
}

impl <B, O, const S: usize> FeedforwardBatchInference<B, O, S>
where B: Backend, O: Backend

{

    pub fn new<M>(
        model: M,
    ) -> Self
    where M: Feedforward<B, O> + Send + Sync + 'static,
    {

        let waiting_requests: Arc<Mutex<Vec<QueueItem<B, O>>>> = Default::default();
        let running = Arc::new(AtomicBool::new(true));
        let active_count = Arc::new(Mutex::new(0));
        let work_notifier = Arc::new(Notify::new());

        let running_clone = running.clone();
        let active_count_clone = active_count.clone();
        let waiting_clone = waiting_requests.clone();
        let work_notifier_clone = work_notifier.clone();
        let pill = Pill::new();


        let background_task = Some(tokio::spawn(async move {
            // By moving this in here, when inference loop panics and this thread dies
            // drop will be called, causing a panic escalation
            #[allow(unused_variables)]
            let panic_escalator = pill;
            Self::run_inference_loop_batch_item(
                model,
                running_clone,
                active_count_clone.clone(),
                waiting_clone.clone(),
                work_notifier_clone
            ).await;
        }));


        Self {
            running,
            background_task,
            waiting_requests,
            work_notifier,
            active_count
        }
    }

    /// Runs the main inference loop for processing batched requests.
    ///
    /// This is the core internal method that handles:
    /// 1. Waiting for and collecting new requests
    /// 2. Building batches of appropriate size
    /// 3. Running the model on batched inputs
    /// 4. Dispatching outputs to the respective streams
    /// 5. Managing the active tensor state and sequence tracking
    ///
    /// # Parameters
    ///
    /// * `model` - The autoregressive model to use for generation
    /// * `running` - Atomic flag indicating whether the loop should continue
    /// * `stop_token` - Token that signals generation completion
    /// * `padding_token` - Token used for sequence padding
    /// * `active_count` - Shared counter of active sequences
    /// * `waiting_requests` - Queue of pending requests
    /// * `work_notifier` - Notification mechanism for signaling new work
    ///
    /// # Implementation Notes
    ///
    /// This method maintains an internal state of active sequences and manages
    /// the dynamic batching process. It handles:
    /// - Starting new sequences when batch capacity is available
    /// - Padding sequences to uniform length within a batch
    /// - Detecting when sequences have completed (by generating stop tokens)
    /// - Removing completed sequences from the batch
    /// - Streaming generated tokens back to requesters
    async fn run_inference_loop_batch_item<M>(
        model: M,
        running: Arc<AtomicBool>,
        active_count: Arc<Mutex<usize>>,
        waiting_requests: Arc<Mutex<Vec<QueueItem<B, B>>>>,
        work_notifier: Arc<Notify>,
    )
    where M: Autoregressive<B> + Send + Sync + 'static,
    {
        let active_tensor: Mutex<Option<B>> = Mutex::new(None);
        let mut batch_items: Vec<BatchItem<B>>= vec![];

        while running.load(Ordering::SeqCst) {
            // Check if there's work to do (either active items or waiting requests)
            let should_process = Self::should_process(active_count.clone(), waiting_requests.clone()).await;

            if !should_process {
                // No work to do, wait for notification or check periodically
                let timeout = Self::timeout_await_notifier(&work_notifier).await;
                if timeout.is_err() {
                    // Timeout occurred, loop back and check again
                    continue;
                }
            }

            // drain new requests
            let items = Self::drain_possible_requests(
                S, waiting_requests.clone(), active_count.clone()
            ).await;

            // If we have items to process or active tensors, do work
            if !items.is_empty() || {
                let active = active_count.lock().await;
                *active > 0
            } {
                let mut lock = active_tensor.lock().await;
                Self::push_new_requests(&mut lock, items, &mut batch_items).await;
                let input = (*lock).clone();
                match input {
                    None => {}
                    Some(input) => {
                        let output = model.forward(input.clone()).await;
                        Self::send_outputs_to_stream(output.clone(), &batch_items).await;
                        batch_items.clear();
                        *lock = None;
                    }
                }
            }
        }
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
        waiting_requests: Arc<Mutex<Vec<QueueItem<B, O>>>>
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

    /// Adds new requests to the active batch.
    ///
    /// This helper method processes a list of new queue items, creates batch items
    /// for tracking them, and adds their input tensors to the active tensor.
    ///
    /// # Parameters
    ///
    /// * `active_tensor` - The current active tensor containing all batched sequences
    /// * `new_queue_items` - New requests to be added to the batch
    /// * `batch_items` - List of batch items for tracking sequence state
    /// * `padding_token` - Token used for padding sequences to uniform length
    async fn push_new_requests(
        active_tensor: &mut Option<B::Unsqueezed>,
        new_queue_items: Vec<QueueItem<B, B>>,
        batch_items: &mut Vec<BatchItem<B>>,
    ) {
        for item in new_queue_items.into_iter() {
            let batch_item = BatchItem::new(
                0,
                item.sender().clone(),
            );
            batch_items.push(batch_item);
            Self::add_to_active_tensor(active_tensor, item.input()).await;
        }
    }


    /// Adds a new sequence to the active tensor.
    ///
    /// This helper method handles the tensor manipulation required to add a new
    /// sequence to the active batch, including padding to ensure uniform sequence length.
    ///
    /// # Parameters
    ///
    /// * `active_tensor` - The current active tensor containing all batched sequences
    /// * `request_tensor` - The new sequence tensor to add to the batch
    /// * `padding_token` - Token used for padding sequences to uniform length
    ///
    /// # Tensor Shape Handling
    ///
    /// This method automatically handles cases where:
    /// - The active tensor is empty (initializes with the new sequence)
    /// - The new sequence is longer than existing ones (pads existing sequences)
    /// - The new sequence is shorter than existing ones (pads the new sequence)
    async fn add_to_active_tensor(
        active_tensor: &mut Option<B>,
        request_tensor: &B,
    ) {
        let is_some = active_tensor.is_some();
        if !is_some {
            *active_tensor = Some(request_tensor);
            return
        }
        match active_tensor {
            None => *active_tensor = Some(request_tensor),
            Some(active_tensor) => {
                *active_tensor = add_sequence_to_outside_of_slot(active_tensor, &request_tensor);
            }
        }

    }

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
        outputs: B,
        batch_items: &Vec<BatchItem<B>>
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
        waiting_requests: Arc<Mutex<Vec<QueueItem<B, B>>>>,
        active_count: Arc<Mutex<usize>>,
    ) -> Vec<QueueItem<B, B>> {
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
impl <B, O, const S: usize> Drop for FeedforwardBatchInference<B, O, S>
where B: Backend, O: Backend
{
    fn drop(&mut self) {
        self.shutdown();
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
        self.work_notifier.notify_one();
        Item::new(rx)
    }
}


use std::collections::HashSet;
use std::panic;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;
use async_trait::async_trait;
use tokio::sync::{mpsc, Mutex, Notify};
use tokio::task::JoinHandle;
use tokio::time::error::Elapsed;
use crate::communication::{ItemStream, BatchItem, QueueItem, Pill};
use crate::backend::{Backend, Unsqueezable};
use super::{Autoregressive, AutoregressiveBatcher};
use crate::tensor::*;

/// # AutoregressiveBatchInference
///
/// An implementation of the `AutoregressiveBatcher` trait that provides efficient
/// batched processing for autoregressive models.
///
/// This struct manages an inference pipeline that:
/// 1. Collects multiple generation requests
/// 2. Dynamically batches them up to a maximum batch size (`S`)
/// 3. Processes them through an autoregressive model
/// 4. Streams results back to each requester
///
/// ## Type Parameters
///
/// * `B` - The tensor backend type that implements `Backend` and `Unsqueezable`
/// * `S` - A const generic parameter specifying the maximum batch size
///
/// ## Implementation Details
///
/// The implementation uses a background task that continuously processes requests
/// and manages the active tensor state. Key internal mechanisms include:
///
/// - A thread-safe queue of waiting requests
/// - A notification system to efficiently signal when work is available
/// - Dynamic tensor manipulation to handle variable-length sequences
/// - Automatic detection and handling of completed sequences
/// - Resource management to ensure bounded memory usage
pub struct AutoregressiveBatchInference<B, const S: usize>
where B: Backend + Unsqueezable
{
    /// Flag indicating whether the background task should continue running
    running: Arc<AtomicBool>,

    /// Handle to the background task that processes batched requests
    background_task: Option<JoinHandle<()>>,

    /// Thread-safe queue of pending requests waiting to be processed
    waiting_requests: Arc<Mutex<Vec<QueueItem<B, B>>>>,

    /// Notification mechanism to signal when new work is available
    work_notifier: Arc<Notify>,

    /// Counter tracking the number of currently active sequences in the batch
    active_count: Arc<Mutex<usize>>
}

impl <B, const S: usize> AutoregressiveBatchInference<B, S>
where B: Backend + Unsqueezable,

{
    /// Creates a new batched autoregressive inference engine using the provided model.
    ///
    /// This constructor initializes the batching system and starts the background
    /// processing task that will handle incoming requests.
    ///
    /// # Parameters
    ///
    /// * `model` - The autoregressive model implementation that will process batched requests
    /// * `stop_token` - Token that signals the end of generation for a sequence
    /// * `padding_token` - Token used to pad sequences to uniform length within a batch
    ///
    /// # Tensor Shape Requirements
    ///
    /// Both `stop_token` and `padding_token` must be of one dimension higher than intended.
    /// For example, for a typical text-based autoregressive model with input shape
    /// `(batch_size, seq_len)`, both tokens should have shape `[1]`, rather than a rank-0 value.
    /// This is due to limitations in handling zero-sized tensors.
    ///
    /// # Panics
    ///
    /// This method will panic if:
    /// - `padding_token` has an empty shape (rank 0)
    /// - The first dimension of `padding_token` is not 1
    ///
    /// # Returns
    ///
    /// A new `AutoregressiveBatchInference` instance configured with the provided model and tokens
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

        let waiting_requests: Arc<Mutex<Vec<QueueItem<B, B>>>> = Default::default();
        let running = Arc::new(AtomicBool::new(true));
        let active_count = Arc::new(Mutex::new(0));
        let work_notifier = Arc::new(Notify::new());

        let running_clone = running.clone();
        let active_count_clone = active_count.clone();
        let waiting_clone = waiting_requests.clone();
        let work_notifier_clone = work_notifier.clone();
        let stop_clone = stop_token.clone();
        let padding_clone = padding_token.clone();
        let pill = Pill::new();


        let background_task = Some(tokio::spawn(async move {
            /// By moving this in here, when inference loop panics and this thread dies
            /// drop will be called, causing a panic escalation
            #[allow(unused_variables)]
            let panic_escalator = pill;
            Self::run_inference_loop_batch_item(
                model,
                running_clone,
                stop_clone,
                padding_clone,
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
        stop_token: B,
        padding_token: B,
        active_count: Arc<Mutex<usize>>,
        waiting_requests: Arc<Mutex<Vec<QueueItem<B, B>>>>,
        work_notifier: Arc<Notify>,
    )
    where M: Autoregressive<B> + Send + Sync + 'static,
    {
        let active_tensor: Mutex<Option<B::Unsqueezed>> = Mutex::new(None);
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
                Self::push_new_requests(&mut lock, items, &mut batch_items, &padding_token).await;
                let input = (*lock).clone();
                match input {
                    None => {}
                    Some(input) => {
                        let output = model.forward(input.clone()).await;
                        Self::send_outputs_to_stream(output.clone(), &batch_items).await;
                        let mut concatenated: Option<B::Unsqueezed> = Some(concat_output(&input, &output));
                        Self::update_sequence_lengths(&mut batch_items).await;
                        Self::update_state_for_output(&mut concatenated, &output, &stop_token, &mut batch_items, active_count.clone()).await;
                        *lock = concatenated;
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
        padding_token: &B
    ) {
        for item in new_queue_items.into_iter() {
            let batch_item = BatchItem::new(
                item.input().shape()[0],
                item.sender().clone(),
            );
            batch_items.push(batch_item);
            Self::add_to_active_tensor(active_tensor, item.input(), padding_token).await;
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
        active_tensor: &mut Option<B::Unsqueezed>,
        request_tensor: &B,
        padding_token: &B,
    ) {
        let is_some = active_tensor.is_some();
        if !is_some {
            *active_tensor = Some(request_tensor.unsqueeze(0));
            return
        }
        match active_tensor {
            None => *active_tensor = Some(request_tensor.unsqueeze(0)),
            Some(active_tensor) => {
                let sequence_dims = request_tensor.shape();
                let mut rqt = request_tensor.clone();
                let seq_len = sequence_dims[0];
                let active_dims = active_tensor.shape();
                let batch_len = active_dims[1];

                if seq_len > batch_len {
                    // need to expand active tensor length at the front with padding
                    let diff = seq_len - batch_len;
                    let mut new_dims = active_dims.to_vec();
                    new_dims[1] = seq_len - batch_len;
                    // zero padding
                    *active_tensor = pad_all_sequences(active_tensor, diff, padding_token);
                } else if batch_len > seq_len {
                    let diff = batch_len - seq_len;
                    rqt = pad_single_sequence(&rqt, diff, padding_token);
                }
                *active_tensor = add_sequence_to_outside_of_slot(active_tensor, &rqt);
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


    /// Updates state after processing model outputs.
    ///
    /// This method performs several important state management tasks:
    /// 1. Identifies sequences that have generated a stop token
    /// 2. Removes completed sequences from the batch
    /// 3. Updates the active tensor to reflect removed sequences
    /// 4. Trims the sequence length if needed
    /// 5. Updates the active sequence count
    ///
    /// # Parameters
    ///
    /// * `inputs` - The current active tensor
    /// * `output` - The most recent model output
    /// * `stop_token` - Token that signals sequence completion
    /// * `batch_items` - List of batch items tracking sequence state
    /// * `active_sequence_count` - Shared counter of active sequences
    async fn update_state_for_output(
        inputs: &mut Option<B::Unsqueezed>,
        output: &B,
        stop_token: &B,
        batch_items: &mut Vec<BatchItem<B>>,
        active_sequence_count: Arc<Mutex<usize>>
    ) {
        let where_end: HashSet<usize> = where_equals_stop_token(output, stop_token).into_iter().collect();

        let num_sequences_ended = where_end.len();
        if num_sequences_ended ==  0 {
            return
        }
        let ended = where_end.len();
        let mut idx = 0;
        batch_items.retain( |_| {
            let keep = !where_end.contains(&idx);
            idx += 1;
            keep
        });
        let mut sorted_ends: Vec<usize> = where_end.into_iter().collect();
        sorted_ends.sort_by(|a, b| b.cmp(a));
        sorted_ends.iter().for_each(|&idx| {
            *inputs = pop_sequence_from_slot(inputs, idx)
        });
        let max_sequence_length = BatchItem::max_seq_len_for_batch_items(batch_items);
        *inputs = trim_sequence(inputs, max_sequence_length);
        {
            let mut sequence_count = active_sequence_count.lock().await;
            *sequence_count -= ended;
        }
    }


    /// Updates sequence length tracking for all active batch items.
    ///
    /// This method increments the sequence length counter for each active
    /// batch item after a successful generation step.
    ///
    /// # Parameters
    ///
    /// * `batch_items` - List of batch items to update
    async fn update_sequence_lengths(batch_items: &mut Vec<BatchItem<B>>) {
        for batch_item in batch_items.iter_mut() {
                batch_item.increment_sequence_length(1);
        }
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
impl <B, const S: usize> Drop for AutoregressiveBatchInference<B, S>
where B: Backend + Unsqueezable,
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
        ItemStream::new(rx)
    }
}


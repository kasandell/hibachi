use std::collections::HashSet;
use std::marker::PhantomData;
use std::panic;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;
use async_trait::async_trait;
use tokio::sync::{mpsc, Mutex, Notify};
use tokio::task::JoinHandle;
use tokio::time::error::Elapsed;
use crate::communcation::{ItemStream, BatchItem, QueueItem, Pill};
use crate::{Autoregressive, AutoregressiveBatcher};
use crate::backend::Backend;
use crate::tensor::*;

pub struct BatchedRegressiveInference<B, M, const S: usize>
where B: Backend, M: Autoregressive<Sequence=B, Output=B> + Send + Sync + 'static
{
    _marker: PhantomData<M>,
    running: Arc<AtomicBool>,
    background_task: Option<JoinHandle<()>>,
    waiting_requests: Arc<Mutex<Vec<QueueItem<B, B>>>>,
    work_notifier: Arc<Notify>,
    active_count: Arc<Mutex<usize>>
}

impl <B, M, const S: usize> BatchedRegressiveInference<B, M, S>
where B: Backend,
M: Autoregressive<Sequence=B, Output=B> + Send + Sync + 'static
{
    /// Instantiate a new batched regressive inference engine on top of `model`,
    /// stopping when we hit stop_token.
    /// stop token & padding token must be of one dimension higher than they are intended.
    /// this is bc there's no support for 0 size tensor
    /// so, for a typical text based autoregressive model, of (batch_size, seq_len),
    /// we expect stop token and padding token to be of dimension [1], rathar than a 0 rank value
    pub fn new(
        model: M,
        stop_token: &B,
        padding_token: &B,
    ) -> Self {
        let padding_shape = padding_token.shape();
        assert!(padding_shape.len() > 0, "padding shape must be of rank 1 or higher");
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
        let pill = Pill {};
        let background_task = Some(tokio::spawn(async move {
            // by moving this in here, when inference loop panics and this thread dies
            // drop will be called, causing a panic escalation
            #[allow(unused_variables)]
            let panic_escalator = pill;
            Self::run_inference_loop_batch_item(
                model,
                //active_tensor,
                running_clone,
                stop_clone,
                padding_clone,
                active_count_clone.clone(),
                waiting_clone.clone(),
                work_notifier_clone
            ).await;
            // *active_count_clone.lock().await = 0;
            // *waiting_clone.lock().await = vec![];
        }));
        Self {
            _marker: PhantomData,
            running,
            background_task,
            waiting_requests,
            work_notifier,
            active_count
        }
    }

    async fn run_inference_loop_batch_item(
        model: M,
        running: Arc<AtomicBool>,
        stop_token: B,
        padding_token: B,
        active_count: Arc<Mutex<usize>>,
        waiting_requests: Arc<Mutex<Vec<QueueItem<B, B>>>>,
        work_notifier: Arc<Notify>,
    ) {
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
                Self::push_new_requests(&mut lock, items, &mut batch_items, &padding_token).await;
                let input = (*lock).clone();
                match input {
                    None => {}
                    Some(input) => {
                        let output = model.forward(input.clone()).await;
                        Self::send_outputs_to_stream(output.clone(), &batch_items).await;
                        let mut concatenated = Some(concat_output(&input, &output));
                        Self::update_sequence_lengths(&mut batch_items).await;
                        Self::update_state_for_output(&mut concatenated, &output, &stop_token, &mut batch_items, active_count.clone()).await;
                        *lock = concatenated;
                    }
                }
            }
        }
    }

    pub async fn active_count(&self) -> usize {
        self.active_count.clone().lock().await.clone()
    }

    #[inline]
    async fn timeout_await_notifier(notifier: &Notify) -> Result<(), Elapsed>{
        tokio::time::timeout(
            Duration::from_millis(100),
            notifier.notified()
        ).await
    }

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

    async fn push_new_requests(
        active_tensor: &mut Option<B>,
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


    // request tensor comes in as (seq_len, ...)
    async fn add_to_active_tensor(
        active_tensor: &mut Option<B>,
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


    async fn update_state_for_output(
        inputs: &mut Option<B>,
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


    async fn update_sequence_lengths(batch_items: &mut Vec<BatchItem<B>>) {
        for batch_item in batch_items.iter_mut() {
                batch_item.increment_sequence_length(1);
        }
    }

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


impl <B, M, const S: usize> Drop for BatchedRegressiveInference<B, M, S>
where B: Backend,
      M: Autoregressive<Sequence=B, Output=B> + Send + Sync + 'static
{
    fn drop(&mut self) {
        self.shutdown();
    }
}


#[async_trait]
impl <B, M, const S: usize> AutoregressiveBatcher<B, B> for BatchedRegressiveInference<B, M, S>
where B: Backend,
      M: Autoregressive<Sequence=B, Output=B> + Send + Sync + 'static {
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

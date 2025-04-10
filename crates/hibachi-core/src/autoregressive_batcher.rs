use std::marker::PhantomData;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;
use async_trait::async_trait;
use tokio::sync::{mpsc, Mutex, Notify};
use tokio::task::JoinHandle;
use tokio::time::error::Elapsed;
use crate::{AsyncItemStream, Autoregressive, AutoregressiveBatcher, BatchItem, QueueItem};
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
}

impl <B, M, const S: usize> BatchedRegressiveInference<B, M, S>
where B: Backend,
M: Autoregressive<Sequence=B, Output=B> + Send + Sync + 'static
{
    /// Instantiate a new batched regressive inference engine on top of `model`,
    /// stopping when we hit stop_token.
    pub fn new(
        model: M,
        stop_token: &B,
    ) -> Self {

        let waiting_requests: Arc<Mutex<Vec<QueueItem<B, B>>>> = Default::default();
        let running = Arc::new(AtomicBool::new(true));
        let active_count = Arc::new(Mutex::new(0));
        let work_notifier = Arc::new(Notify::new());

        let device = stop_token.device();
        let dtype = stop_token.dtype();
        let stop_token_dims = stop_token.shape();
        let token_size = stop_token_dims[0];
        assert_eq!(token_size, 1, "token size must be of length 1");
        let active_tensor = Arc::new(Mutex::new(
            B::zeros(
                &[S, 1],
                dtype,
                &device
            )
        ));

        let running_clone = running.clone();
        let active_count_clone = active_count.clone();
        let waiting_clone = waiting_requests.clone();
        let work_notifier_clone = work_notifier.clone();
        let stop_clone = stop_token.clone();
        let background_task = Some(tokio::spawn(async move {
            Self::run_inference_loop_batch_item(
                model,
                active_tensor,
                running_clone,
                stop_clone,
                active_count_clone,
                waiting_clone,
                work_notifier_clone
            ).await;
        }));
        Self {
            _marker: PhantomData,
            running,
            background_task,
            waiting_requests,
            work_notifier,
        }
    }

    async fn run_inference_loop_batch_item(
        model: M,
        active_tensor: Arc<Mutex<B>>,
        running: Arc<AtomicBool>,
        stop_token: B,
        active_count: Arc<Mutex<usize>>,
        waiting_requests: Arc<Mutex<Vec<QueueItem<B, B>>>>,
        work_notifier: Arc<Notify>,
    ) {
        let mut batch_items: [Option<BatchItem<B>>; S] = [ const {None}; S];

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
                //println!("Input shape: {:?}", input.dims());
                let output = model.forward(input.clone()).await;
                //println!("Output shape: {:?}", output.dims());
                Self::update_sequence_lengths(&mut batch_items).await;
                Self::send_outputs_to_stream(output.clone(), &batch_items).await;
                let mut concatenated = concat_output(&input, &output);
                Self::update_state_for_output(&mut concatenated, &output, &stop_token, &mut batch_items, active_count.clone()).await;
                *lock = concatenated;
            }
        }
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
        active_tensor: &mut B,
        mut new_queue_items: Vec<QueueItem<B, B>>,
        batch_items: &mut [Option<BatchItem<B>>; S],
    ) {
        for (idx, item) in batch_items.iter_mut().enumerate() {
            if new_queue_items.is_empty() {return};
            if item.is_none() {
            let pop = new_queue_items.pop().expect("Expected a queue item");
            let batch_item = BatchItem::new(
                idx,
                pop.input().shape()[0],
                pop.sender().clone(),
            );
            *item = Some(batch_item);
            Self::set_slot_in_active_tensor(active_tensor, pop.input(), idx).await;}
        }

        assert_eq!(new_queue_items.len(), 0, "Should be no outstanding queue items!!!");
    }

    async fn set_slot_in_active_tensor(
        active_tensor: &mut B,
        request_tensor: &B,
        index: usize
    ) {
        let sequence_dims = request_tensor.shape();
        let seq_len = sequence_dims[0];
        let active_dims = active_tensor.shape();
        let mut batch_len = active_dims[1];
        //assert_eq!(seq_tok_width, batch_tok_width, "tokens must be same size");

        //println!("Seq len: {} batch len: {}", seq_len, batch_len);
        if seq_len > batch_len {
            // need to expand active tensor length at the front with padding
            //println!("updating seq len");
            let diff = seq_len - batch_len;
            let mut new_dims = active_dims.to_vec();
            new_dims[1] = seq_len - batch_len;
            // zero padding
            *active_tensor = zero_pad_sequence(active_tensor, seq_len - batch_len);
            //let new = B::zeros(&new_dims, active_tensor.dtype(), active_tensor.device());
            //*active_tensor = B::cat(&[new, active_tensor], 1);
            batch_len += diff;
        }
        // active tensor now big enough to fit. good
        //println!("{:?} {:?}", request_tensor.dims(), active_tensor.dims());
        *active_tensor = assign_sequence_to_slot(active_tensor, request_tensor, index, batch_len - seq_len, batch_len);
    }

    async fn send_outputs_to_stream(
        outputs: B,
        batch_items: &[Option<BatchItem<B>>; S],
    ) {
        let sliced = slice_tensor_by_batch_dimension(outputs);
        batch_items.iter().enumerate().for_each(|(idx, e)| {
            match e {
                None => {}
                Some(item) => {
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
                }
            }
        });
    }


    async fn update_state_for_output(
        inputs: &mut B,
        output: &B,
        stop_token: &B,
        batch_items: &mut [Option<BatchItem<B>>; S],
        active_sequence_count: Arc<Mutex<usize>>
    ) {
        let where_end = where_equals_stop_token(output, stop_token);
        let num_sequences_ended = where_end.len();
        if num_sequences_ended ==  0 {
            return
        }
        let mut ended = 0;
        where_end.iter().for_each(|&i| {
            if batch_items[i].is_some() {
                ended += 1;
            }
            batch_items[i] = None;
            zero_sequence(inputs, i);
        });

        let max_sequence_length = BatchItem::max_seq_len_for_batch_items(batch_items);
        *inputs = trim_sequence(inputs, max_sequence_length);

        {
            let mut sequence_count = active_sequence_count.lock().await;
            *sequence_count -= ended;
        }
    }


    async fn update_sequence_lengths(batch_items: &mut [Option<BatchItem<B>>; S]) {
        for batch_item in batch_items.iter_mut() {
            if let Some(item) = batch_item {
                item.increment_sequence_length(1);
            }
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
        //println!("Available: {}, reqlen: {}", available_slots, requests.len());

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
    async fn run(&self, item: B) -> AsyncItemStream<B> {
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
        AsyncItemStream::new(rx)
    }
}

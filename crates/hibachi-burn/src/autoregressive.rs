use std::marker::PhantomData;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;
use async_trait::async_trait;
use burn::prelude::{Backend, Tensor};
use burn::tensor::Shape;
use tokio::sync::{mpsc, Mutex, Notify};
use tokio::task::JoinHandle;
use hibachi_core::{Batcher, AsyncItemStream, BatchItem, QueueItem, Autoregressive};
use crate::tensor::*;

type Tensor1D<B> = Tensor<B, 1>;
type Tensor2D<B> = Tensor<B, 2>;
type QueueItemType<B> = QueueItem<Tensor1D<B>, Tensor1D<B>>;
type BatchItemType<B> = BatchItem<Tensor1D<B>>;

pub struct BatchedRegressiveInference<B, const S: usize>
where B: Backend
{
    _marker: PhantomData<B>,
    running: Arc<AtomicBool>,
    background_task: Option<JoinHandle<()>>,
    waiting_requests: Arc<Mutex<Vec<QueueItemType<B>>>>,
    work_notifier: Arc<Notify>,
}

impl <B, const S: usize> BatchedRegressiveInference<B, S>
where B: Backend {

    /// Instantiate a new batched regressive inference engine on top of `model`,
    /// stopping when we hit stop_token.
    pub fn new(
        model: Box<dyn Autoregressive<Sequence=Tensor2D<B>, Output=Tensor1D<B>> + Send>,
        stop_token: Tensor1D<B>
    ) -> Self {

        let waiting_requests: Arc<Mutex<Vec<QueueItemType<B>>>> = Default::default();
        let running = Arc::new(AtomicBool::new(true));
        let active_count = Arc::new(Mutex::new(0));
        let work_notifier = Arc::new(Notify::new());

        let device = stop_token.device();
        let stop_token_dims = stop_token.dims();
        let token_size = stop_token_dims[0];
        assert_eq!(token_size, 1, "token size must be of length 1");
        let active_tensor = Arc::new(Mutex::new(
            Tensor2D::<B>::zeros(
                Shape::new([S, 1]),
                &device
            )
        ));

        let running_clone = running.clone();
        let active_count_clone = active_count.clone();
        let waiting_clone = waiting_requests.clone();
        let work_notifier_clone = work_notifier.clone();
        let background_task = Some(tokio::spawn(async move {
            Self::run_inference_loop_batch_item(
                model,
                active_tensor,
                running_clone,
                stop_token,
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
        model: Box<dyn Autoregressive<Sequence=Tensor2D<B>, Output=Tensor1D<B>> + Send>,
        active_tensor: Arc<Mutex<Tensor2D<B>>>,
        running: Arc<AtomicBool>,
        stop_token: Tensor1D<B>,
        active_count: Arc<Mutex<usize>>,
        waiting_requests: Arc<Mutex<Vec<QueueItemType<B>>>>,
        work_notifier: Arc<Notify>,
    ) {
        let mut batch_items: [Option<BatchItem<Tensor1D<B>>>; S] = [ const {None}; S];

        while running.load(Ordering::SeqCst) {
            // Check if there's work to do (either active items or waiting requests)
            let should_process = Self::should_process(active_count.clone(), waiting_requests.clone()).await;

            if !should_process {
                // No work to do, wait for notification or check periodically
                let timeout = tokio::time::timeout(
                    Duration::from_millis(100),
                    work_notifier.notified()
                ).await;

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
                let output = model.forward(input.clone());
                Self::update_sequence_lengths(&mut batch_items).await;
                Self::send_outputs_to_stream(output.clone(), &batch_items).await;
                let mut concatenated = concat_output(input, output.clone());
                Self::update_state_for_output(&mut concatenated, &output, &stop_token, &mut batch_items, active_count.clone()).await;
                *lock = concatenated;
            }
        }
    }

    #[inline]
    async fn should_process(
        active_count: Arc<Mutex<usize>>,
        waiting_requests: Arc<Mutex<Vec<QueueItemType<B>>>>
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
        active_tensor: &mut Tensor2D<B>,
        mut new_queue_items: Vec<QueueItemType<B>>,
        batch_items: &mut [Option<BatchItemType<B>>; S],
    ) {
        for (idx, item) in batch_items.iter_mut().enumerate() {
            if new_queue_items.is_empty() {return};
            if item.is_none() {
            let pop = new_queue_items.pop().expect("Expected a queue item");
            let batch_item = BatchItem::new(
                idx,
                pop.input().shape().dims[0],
                pop.sender()
            );
            *item = Some(batch_item);
            Self::set_slot_in_active_tensor(active_tensor, pop.input(), idx).await;}
        }

        assert_eq!(new_queue_items.len(), 0, "Should be no outstanding queue items!!!");
    }

    async fn set_slot_in_active_tensor(
        active_tensor: &mut Tensor2D<B>,
        request_tensor: &Tensor1D<B>,
        index: usize
    ) {
        let sequence_dims = request_tensor.dims();
        let seq_len = sequence_dims[0];
        let active_dims = active_tensor.dims();
        let mut batch_len = active_dims[1];
        //assert_eq!(seq_tok_width, batch_tok_width, "tokens must be same size");

        if seq_len > batch_len {
            // need to expand active tensor length at the front with padding
            let diff = seq_len - batch_len;
            zero_pad_sequence(active_tensor, seq_len - batch_len);
            batch_len += diff;


        }
        // active tensor now big enough to fit. good
        *active_tensor = active_tensor.clone().slice_assign([index..index+1, (batch_len-seq_len)..batch_len], request_tensor.clone().unsqueeze_dim(0));
    }

    async fn send_outputs_to_stream(
        outputs: Tensor1D<B>,
        batch_items: &[Option<BatchItemType<B>>; S],
    ) {
        let sliced = slice_tensor_by_batch_dimension::<B, S>(outputs);
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
        inputs: &mut Tensor2D<B>,
        output: &Tensor1D<B>,
        stop_token: &Tensor1D<B>,
        batch_items: &mut [Option<BatchItemType<B>>; S],
        active_sequence_count: Arc<Mutex<usize>>
    ) {
        let where_end = where_equals_stop_token(output, stop_token);
        let num_sequences_ended = where_end.len();
        if num_sequences_ended ==  0 {
            return
        }
        let current_dims = inputs.shape().dims;
        let zeros = Tensor::zeros(Shape::from([1, current_dims[1]]), &inputs.device());

        let mut ended = 0;
        where_end.iter().for_each(|&i| {
            if batch_items[i].is_some() {
                ended += 1;
            }
            batch_items[i] = None;
            *inputs = inputs.clone().slice_assign([0..1, 0..current_dims[1]], zeros.clone());
        });

        let max_sequence_length = BatchItem::max_seq_len_for_batch_items(batch_items);
        *inputs = trim_sequence(inputs, max_sequence_length);

        {
            let mut sequence_count = active_sequence_count.lock().await;
            *sequence_count -= ended;
        }
    }


    async fn update_sequence_lengths(batch_items: &mut [Option<BatchItemType<B>>; S]) {
        for batch_item in batch_items.iter_mut() {
            if let Some(item) = batch_item {
                item.increment_sequence_length(1)
            }
        }
    }

    async fn drain_possible_requests(
        batch_size: usize,
        waiting_requests: Arc<Mutex<Vec<QueueItemType<B>>>>,
        active_count: Arc<Mutex<usize>>,
    ) -> Vec<QueueItemType<B>> {
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


impl <B, const S: usize> Drop for BatchedRegressiveInference<B, S>
where B: Backend {
    fn drop(&mut self) {
        self.shutdown();
    }
}


#[async_trait]
impl <B, const S: usize> Batcher<Tensor1D<B>, Tensor1D<B>> for BatchedRegressiveInference<B, S>
where B: Backend {
    async fn run(&self, item: Tensor1D<B>) -> AsyncItemStream<Tensor1D<B>> {
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

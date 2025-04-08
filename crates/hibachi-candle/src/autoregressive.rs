use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;
use async_trait::async_trait;
use tokio::sync::{mpsc, Mutex, Notify};
use tokio::task::JoinHandle;
use hibachi_core::{AutoregressiveBatcher, AsyncItemStream, Autoregressive};
use hibachi_core::{BatchItem, QueueItem};
use crate::tensor::*;
use candle_core::Tensor;
use tokio::time::error::Elapsed;

type Tensor1D = Tensor;
type Tensor2D = Tensor;
type QueueItemType = QueueItem<Tensor1D, Tensor1D>;
type BatchItemType = BatchItem<Tensor1D>;

pub struct BatchedRegressiveInference<const S: usize>
{
    running: Arc<AtomicBool>,
    background_task: Option<JoinHandle<()>>,
    waiting_requests: Arc<Mutex<Vec<QueueItemType>>>,
    work_notifier: Arc<Notify>,
}

impl <const S: usize> BatchedRegressiveInference<S> {
    /// Instantiate a new batched regressive inference engine on top of `model`,
    /// stopping when we hit stop_token.
    pub fn new(
        model: Box<dyn Autoregressive<Sequence=Tensor2D, Output=Tensor1D> + Send>,
        stop_token: Tensor
    ) -> Self {

        let waiting_requests: Arc<Mutex<Vec<QueueItemType>>> = Default::default();
        let running = Arc::new(AtomicBool::new(true));
        let active_count = Arc::new(Mutex::new(0));
        let work_notifier = Arc::new(Notify::new());

        let device = stop_token.device();
        let dtype = stop_token.dtype();
        let stop_token_dims = stop_token.dims();
        let token_size = stop_token_dims[0];
        assert_eq!(token_size, 1, "token size must be of length 1");
        let active_tensor = Arc::new(Mutex::new(
            Tensor::zeros(
                &[S, 1],
                dtype,
                &device
            ).expect("Creates active tensor")
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
            running,
            background_task,
            waiting_requests,
            work_notifier,
        }
    }

    async fn run_inference_loop_batch_item(
        model: Box<dyn Autoregressive<Sequence=Tensor2D, Output=Tensor1D> + Send>,
        active_tensor: Arc<Mutex<Tensor>>,
        running: Arc<AtomicBool>,
        stop_token: Tensor,
        active_count: Arc<Mutex<usize>>,
        waiting_requests: Arc<Mutex<Vec<QueueItemType>>>,
        work_notifier: Arc<Notify>,
    ) {
        let mut batch_items: [Option<BatchItemType>; S] = [ const {None}; S];

        while running.load(Ordering::SeqCst) {
            // Check if there's work to do (either active items or waiting requests)
            let should_process = Self::should_process(active_count.clone(), waiting_requests.clone()).await;

            if !should_process {
                // No work to do, wait for notification or check periodically
                let timeout = Self::timeout_await_notifier(&work_notifier);
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
                let mut concatenated = concat_output(input, output.clone());
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
        waiting_requests: Arc<Mutex<Vec<QueueItemType>>>
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
        active_tensor: &mut Tensor,
        mut new_queue_items: Vec<QueueItemType>,
        batch_items: &mut [Option<BatchItemType>; S],
    ) {
        for (idx, item) in batch_items.iter_mut().enumerate() {
            if new_queue_items.is_empty() {return};
            if item.is_none() {
            let pop = new_queue_items.pop().expect("Expected a queue item");
            let batch_item = BatchItem::new(
                idx,
                pop.input().shape().dims()[0],
                pop.sender().clone(),
            );
            *item = Some(batch_item);
            Self::set_slot_in_active_tensor(active_tensor, pop.input(), idx).await;}
        }

        assert_eq!(new_queue_items.len(), 0, "Should be no outstanding queue items!!!");
    }

    async fn set_slot_in_active_tensor(
        active_tensor: &mut Tensor,
        request_tensor: &Tensor,
        index: usize
    ) {
        let sequence_dims = request_tensor.dims();
        let seq_len = sequence_dims[0];
        let active_dims = active_tensor.dims();
        let mut batch_len = active_dims[1];
        //assert_eq!(seq_tok_width, batch_tok_width, "tokens must be same size");

        //println!("Seq len: {} batch len: {}", seq_len, batch_len);
        if seq_len > batch_len {
            // need to expand active tensor length at the front with padding
            //println!("updating seq len");
            let diff = seq_len - batch_len;
            zero_pad_sequence(active_tensor, seq_len - batch_len);
            batch_len += diff;


        }
        // active tensor now big enough to fit. good
        //println!("{:?} {:?}", request_tensor.dims(), active_tensor.dims());
        *active_tensor = active_tensor.clone().slice_assign(
            &[index..index + 1, (batch_len - seq_len)..batch_len],
            &request_tensor.clone().unsqueeze(0).expect("Unsqueezed request")
        ).expect("Slice assign")
    }

    async fn send_outputs_to_stream(
        outputs: Tensor,
        batch_items: &[Option<BatchItemType>; S],
    ) {
        let sliced = slice_tensor_by_batch_dimension::<S>(outputs);
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
        inputs: &mut Tensor,
        output: &Tensor,
        stop_token: &Tensor,
        batch_items: &mut [Option<BatchItemType>; S],
        active_sequence_count: Arc<Mutex<usize>>
    ) {
        let where_end = where_equals_stop_token(output, stop_token);
        let num_sequences_ended = where_end.len();
        if num_sequences_ended ==  0 {
            return
        }
        let current_dims = inputs.dims().clone();
        let zeros = Tensor::zeros(&[1, current_dims[1]], inputs.dtype(), &inputs.device())
            .expect("creates zeros in update state");

        let dim = current_dims[1];
        let mut ended = 0;
        where_end.iter().for_each(|&i| {
            if batch_items[i].is_some() {
                ended += 1;
            }
            batch_items[i] = None;
            *inputs = inputs.clone().slice_assign(&[0..1, 0..dim], &zeros)
                .expect("update state inputs slice assign");
        });

        let max_sequence_length = BatchItem::max_seq_len_for_batch_items(batch_items);
        *inputs = trim_sequence(inputs, max_sequence_length);

        {
            let mut sequence_count = active_sequence_count.lock().await;
            *sequence_count -= ended;
        }
    }


    async fn update_sequence_lengths(batch_items: &mut [Option<BatchItemType>; S]) {
        for batch_item in batch_items.iter_mut() {
            if let Some(item) = batch_item {
                item.increment_sequence_length(1);
            }
        }
    }

    async fn drain_possible_requests(
        batch_size: usize,
        waiting_requests: Arc<Mutex<Vec<QueueItemType>>>,
        active_count: Arc<Mutex<usize>>,
    ) -> Vec<QueueItemType> {
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


impl <const S: usize> Drop for BatchedRegressiveInference<S> {
    fn drop(&mut self) {
        self.shutdown();
    }
}


#[async_trait]
impl <const S: usize> AutoregressiveBatcher<Tensor, Tensor> for BatchedRegressiveInference<S> {
    async fn run(&self, item: Tensor) -> AsyncItemStream<Tensor> {
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

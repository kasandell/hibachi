use std::marker::PhantomData;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use async_trait::async_trait;
use burn::prelude::{Backend, Int, Tensor};
use burn::tensor::Shape;
use tokio::sync::{mpsc, Mutex};
use tokio::task::JoinHandle;
use batcher::r#async::batcher::Batcher;
use batcher::r#async::item_stream::AsyncItemStream;
use crate::burn::forward::Forward;
use crate::burn::queue_item::QueueItem;

pub struct BatchedRegressiveInference<B, const S: usize>
where B: Backend
{
    _marker: PhantomData<B>,
    running: Arc<AtomicBool>,
    background_task: Option<JoinHandle<()>>,
    batch_size: usize,
    active_count: Arc<Mutex<usize>>,
    waiting_requests: Arc<Mutex<Vec<QueueItem<B>>>>,
}

impl <B, const S: usize> BatchedRegressiveInference<B, S>
where B: Backend {
    pub fn new(
        model: Box<dyn Forward<B> + Send + Sync>,
        stop_token: Tensor<B, 1>
    ) -> Self {
        println!("Of size {}", S);
        let running = Arc::new(AtomicBool::new(true));
        let device = B::Device::default();
        let rc = running.clone();
        let waiting_requests: Arc<Mutex<Vec<QueueItem<B>>>> = Default::default();
        let stop_token_dims = stop_token.dims();
        let token_size = stop_token_dims[0];
        let active_tensor = Arc::new(Mutex::new(
            Tensor::<B, 3>::zeros(
                Shape::new([S, 1, token_size]),
                &device
            )
        ));
        let active_count= Arc::new(Mutex::new(0));
        let acc = active_count.clone();
        let wrc = waiting_requests.clone();
        let background_task = Some(tokio::spawn(async move {
            Self::_run_inference_loop(
                Arc::new(model),
                active_tensor,
                rc,
                stop_token,
                acc,
                wrc
            ).await;
        }));
        Self {
            _marker: PhantomData,
            running,
            background_task,
            batch_size: S,
            active_count,
            waiting_requests
        }
    }

    async fn _run_inference_loop(
        model: Arc<Box<dyn Forward<B> + Send + Sync>>,
        active_tensor: Arc<Mutex<Tensor<B, 3>>>,
        running: Arc<AtomicBool>,
        stop_token: Tensor<B, 1>,
        active_count: Arc<Mutex<usize>>,
        waiting_requests: Arc<Mutex<Vec<QueueItem<B>>>>,
        //waker: Waker
    ) {
        let mut sequence_lengths = [0usize; S];
        let mut in_use = [false; S];
        let mut senders: [Option<mpsc::UnboundedSender<Tensor<B, 1>>>; S] = [const {None}; S];
        while running.load(Ordering::SeqCst) {
            let items = Self::_drain_possible_requests(
                S, waiting_requests.clone(), active_count.clone()
            ).await;
            if items.len() > 0 {
                println!("Drained {} items", items.len());
            }
            let mut lock = active_tensor.lock();
            if let mut t = lock.await {
                Self::push_new_requests(&mut t, items, &mut in_use, &mut sequence_lengths, &mut senders).await;
                let input = (*t).clone();
                let output = model.forward(input.clone());
                Self::update_output_sequences(&mut sequence_lengths, &mut in_use).await;
                //println!("Updated seqlen: {:?}", &sequence_lengths);
                Self::send_outputs(output.clone(), &in_use, &mut senders).await;
                let mut concatenated = Self::concat_output(input, output.clone()).await;
                Self::update_state_for_output(&mut concatenated, &output, &stop_token, &mut sequence_lengths, &mut in_use, &mut senders, active_count.clone()).await;
                *t = concatenated;
            }
        }
    }

    async fn push_new_requests(
        active_tensor: &mut Tensor<B, 3>,
        mut new_queue_items: Vec<QueueItem<B>>,
        in_use_channels: &mut [bool; S],
        sequence_lengths: &mut [usize; S],
        senders: &mut [Option<mpsc::UnboundedSender<Tensor<B, 1>>>; S],
    ) {
        for (idx, in_use) in in_use_channels.iter_mut().enumerate() {
            if new_queue_items.len() == 0 {return};
            if new_queue_items.len() != 0 && !*in_use {
                let item = new_queue_items.pop().expect("Expected a queue item");
                Self::set_active_channel(item.sender, idx, in_use, senders).await;
                Self::set_slot_in_active_tensor(active_tensor, item.input, sequence_lengths, idx).await;
            }
        }

        assert_eq!(new_queue_items.len(), 0, "Should be no outstanding queue items!!!");
    }

    async fn set_slot_in_active_tensor(
        active_tensor: &mut Tensor<B, 3>,
        request_tensor: Tensor<B, 2>,
        sequence_lengths: &mut [usize; S],
        index: usize
    ) {
        let sequence_dims = request_tensor.dims();
        let (seq_len, seq_tok_width) = (sequence_dims[0], sequence_dims[1]);
        let active_dims = active_tensor.dims();
        let (batch_len, batch_tok_width) = (active_dims[1], active_dims[2]);
        assert_eq!(seq_tok_width, batch_tok_width, "tokens must be same size");
        sequence_lengths[index] = seq_len;

        if seq_len > batch_len {
            // need to expand active tensor length at the front with padding
            Self::pad_sequence(active_tensor, seq_len - batch_len).await;
        }
        // active tensor now big enough to fit. good
        *active_tensor = active_tensor.clone().slice_assign([index..index+1, batch_len-seq_len..batch_len, 0..seq_tok_width], request_tensor.clone().unsqueeze_dim(0));
    }

    async fn pad_sequence(
        active_tensor: &mut Tensor<B, 3>,
        amount: usize
    ) {
        let active_dims = active_tensor.dims();
        let device = active_tensor.device();
        let (batch_size, tok_width) = (active_dims[0], active_dims[2]);
        let padding = Tensor::zeros(Shape::from([batch_size, amount, tok_width]), &device);
        *active_tensor = Tensor::cat(vec![active_tensor.clone(), padding], 1);
    }


    async fn set_active_channel(
        sender: mpsc::UnboundedSender<Tensor<B, 1>>,
        index: usize,
        in_use: &mut bool,
        senders: &mut [Option<mpsc::UnboundedSender<Tensor<B, 1>>>; S],
    ) {
        // TODO: verify channel not taken
        *in_use = true;
        senders[index] = Some(sender);

    }

    async fn send_outputs(
        outputs: Tensor<B, 2>,
        in_use: &[bool; S],
        senders: &mut [Option<mpsc::UnboundedSender<Tensor<B, 1>>>; S]
    ) {
        let sliced = Self::slice_output_by_channel(outputs).await;
        for (idx, in_use) in in_use.iter().enumerate() {
            if *in_use {
                match senders[idx].clone() {
                    None => {panic!("Channel in use with no active sender")}
                    Some(sender) => {
                        sender.send(
                            sliced.get(idx).unwrap().clone()
                        ).unwrap()
                    }
                }
            }
        }
    }

    async fn slice_output_by_channel(
        outputs: Tensor<B, 2>,
    ) -> Vec<Tensor<B, 1>> {
        outputs.chunk(S, 0).into_iter().map(|tensor| tensor.squeeze(0)).collect()
    }

    async fn update_state_for_output(
        inputs: &mut Tensor<B, 3>,
        output: &Tensor<B, 2>,
        stop_token: &Tensor<B, 1>,
        sequence_lengths: &mut [usize; S],
        in_use: &mut [bool; S],
        senders: &mut [Option<mpsc::UnboundedSender<Tensor<B, 1>>>; S],
        active_sequence_count: Arc<Mutex<usize>>
    ) {
        let where_end = Self::where_equals_stop_token_vec(output, stop_token).await;
        let num_sequences_ended = where_end.len();
        if num_sequences_ended ==  0 {
            return
        }
        let current_dims = inputs.shape().dims;
        let zeros = Tensor::zeros(Shape::from([1, current_dims[1], current_dims[2]]), &inputs.device());
        let mut ended = 0;
        let mut max_sequence_length = sequence_lengths.iter().max();
        match max_sequence_length {
            None => {}
            Some(&length) => {
                //println!("Max length pre trim: {}", length);
            }
        }
        for i in where_end {
            sequence_lengths[i] = 0;
            if in_use[i] {
                ended += 1;
            }
            in_use[i] = false;
            senders[i] = None;
            *inputs = inputs.clone().slice_assign([0..1, 0..current_dims[1], 0..current_dims[2]], zeros.clone());
        }

        max_sequence_length = sequence_lengths.iter().max();
        match max_sequence_length {
            None => {}
            Some(&length) => {
                //println!("Max length post trim: {}", length);
                *inputs = Self::trim_sequence(inputs, length).await;
            }
        }
        {
            let mut sequence_count = active_sequence_count.lock().await;
            *sequence_count -= ended;
        }
    }

    async fn trim_sequence(
        tensor: &Tensor<B, 3>,
        max_sequence_length: usize
    ) -> Tensor<B, 3> {
        if max_sequence_length == 0 {
            return tensor.clone();
        }
        let dims = tensor.shape().dims;
        let batch = dims[0] as i64;
        let seq = dims[1] as i64;
        let tok = dims[2] as i64;
        let msl = max_sequence_length as i64;
        //println!("shape: {:?} msl: {} seq:{} batch:{} tok:{}", tensor.shape(), max_sequence_length, seq, batch, tok);
        let start_idx = seq-msl;
        let t = tensor.clone().slice([None, Some((start_idx, seq)), None]);
        //println!("Trim clear?");
        t
    }


    async fn update_output_sequences(sequence_lengths: &mut [usize], in_use: &[bool]) {
        for (idx, i) in sequence_lengths.iter_mut().enumerate() {
            if in_use[idx] {
                *i += 1
            }
        }
    }

    async fn concat_output(input: Tensor<B, 3>, output: Tensor<B, 2>) -> Tensor<B, 3> {

        let concatenated = Tensor::cat(
            vec![ input, output.unsqueeze_dim(1)], 1
        );
        concatenated
    }

    async fn where_equals_stop_token_vec(
        outputs: &Tensor<B, 2>,
        stop_token: &Tensor<B, 1>
    ) -> Vec<usize> {
        // repeat stop token across batch
        let element_wise_stop = stop_token.clone().unsqueeze_dim(0)
            .repeat_dim(0, outputs.shape().dims[0]);
        // element wise equal
        let eq =  outputs.clone().equal(element_wise_stop);
        // collapse equal by row
        let collapsed = eq.all_dim(1);
        // squeeze to rows
        let collapsed_squeezed = collapsed.squeeze::<1>(1);
        let indices = collapsed_squeezed.argwhere();
        let t = indices.transpose();
        if t.shape().dims[1] == 0 {
            return vec![]
        }
        let items = t.squeeze::<1>(0);
        let data = items.to_data();
        let mut final_indices = vec![];
        for row in data.iter::<u32>() {
            final_indices.push(row as usize);
            //println!("Row: {}", row);
        }
        final_indices
    }

    fn shutdown(&mut self) {
        // Signal the background task to stop
        self.running.store(false, Ordering::SeqCst);

        // Await the background task if it exists
        if let Some(task) = self.background_task.take() {
            tokio::spawn(async move {
                let _ = task.await;
            });
        }
    }

    async fn _drain_possible_requests(
        batch_size: usize,
        waiting_requests: Arc<Mutex<Vec<QueueItem<B>>>>,
        active_count: Arc<Mutex<usize>>,
    ) -> Vec<QueueItem<B>> {
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
}


impl <B, const S: usize> Drop for BatchedRegressiveInference<B, S>
where B: Backend {
    fn drop(&mut self) {
        self.shutdown();
    }
}


#[async_trait]
impl <B, const S: usize> Batcher<Tensor<B, 2>, Tensor<B, 1>> for BatchedRegressiveInference<B, S>
    where B: Backend {
    async fn run(&self, item: Tensor<B, 2>) -> AsyncItemStream<Tensor<B, 1>> {
        let (tx, rx) = mpsc::unbounded_channel();
        let queue_item = QueueItem {
            input: item,
            sender: tx,
        };
        println!("Pushed item!");
        if let mut senders = self.waiting_requests.lock().await {
            senders.push(queue_item);
        } else {
            panic!("Unable to enqueue items!!");
        }
        let item_stream = AsyncItemStream::new(rx);
        item_stream
    }
}

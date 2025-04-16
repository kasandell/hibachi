use std::collections::HashSet;
use std::sync::Arc;
use async_trait::async_trait;
use crate::backend::Backend;
use tokio::sync::Mutex;
use crate::autoregressive::Autoregressive;
use crate::backend::Unsqueezable;
use super::queue_item::QueueItem;
use crate::core::handler::BatchHandler;
use crate::tensor::operations::{add_sequence_to_outside_of_slot, concat_output, pad_all_sequences, pad_single_sequence, pop_sequence_from_slot, slice_tensor_by_batch_dimension, trim_sequence, where_equals_stop_token};

pub struct AutoregressiveHandler<M, B>
{
    pub model: M,
    pub padding_token: B,
    pub stop_token: B
}

#[async_trait]
impl<M, B> BatchHandler for AutoregressiveHandler<M, B>
where
    B: Backend + Unsqueezable,
    M: Autoregressive<B> + 'static + Sync + Send
{
    type Request = QueueItem<B>;
    type ModelInput = B::Unsqueezed;
    type ModelOutput = B;

    async fn make_batch_input(&self, model_input: &mut Option<Self::ModelInput>, requests: &[Self::Request]) {
        for item in requests.iter() {
            let request_tensor = item.input();
            match model_input {
                None => *model_input = Some(request_tensor.unsqueeze(0)),
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
                        *active_tensor = pad_all_sequences(active_tensor, diff, &self.padding_token);
                    } else if batch_len > seq_len {
                        let diff = batch_len - seq_len;
                        rqt = pad_single_sequence(&rqt, diff, &self.padding_token);
                    }
                    *active_tensor = add_sequence_to_outside_of_slot(active_tensor, &rqt);
                }
            }
        }
    }

    async fn forward(&self, model_input: &Self::ModelInput) -> Self::ModelOutput {
        self.model.forward(model_input.clone()).await
    }

    async fn handle_outputs(&self, batch: &mut Vec<Self::Request>, input: &mut Option<Self::ModelInput>, output: Self::ModelOutput, active_count: Arc<Mutex<usize>>) {
        // concat expects input to exist
        input.as_mut().map(|input_val| {
            *input_val = concat_output(input_val, &output);
        });
        for batch_item in batch.iter_mut() {
            batch_item.increment_sequence_length(1);
        }
        // Use drain to consume the elements while keeping ownership of the vector
        let split = slice_tensor_by_batch_dimension(output.clone());
        for (sender, slice) in batch.iter().zip(split.iter()) {
            sender.sender().send(slice.clone()).unwrap();
        }
        let where_end: HashSet<usize> = where_equals_stop_token(&output, &self.stop_token).into_iter().collect();
        let num_sequences_ended = where_end.len();
        if num_sequences_ended ==  0 {
            return
        }
        let ended = where_end.len();
        let mut idx = 0;
        batch.retain( |_| {
            let keep = !where_end.contains(&idx);
            idx += 1;
            keep
        });
        let mut sorted_ends: Vec<usize> = where_end.into_iter().collect();
        sorted_ends.sort_by(|a, b| b.cmp(a));
        sorted_ends.iter().for_each(|&idx| {
            *input = pop_sequence_from_slot(input, idx)
        });
        let max_sequence_length = QueueItem::max_seq_len_for_batch_items(batch);
        *input = trim_sequence(input, max_sequence_length);
        {
            let mut sequence_count = active_count.lock().await;
            *sequence_count = sequence_count.saturating_sub(ended);
        }
    }
}


use std::marker::PhantomData;
use std::sync::Arc;
use async_trait::async_trait;
use crate::backend::Backend;
use tokio::sync::Mutex;
use crate::backend::Unsqueezable;
use crate::core::handler::BatchHandler;
use crate::feedforward::core_trait::Feedforward;
use crate::feedforward::queue_item::QueueItem;
use crate::tensor::operations::{
    add_sequence_to_outside_of_slot,
    slice_tensor_by_batch_dimension
};

pub struct FeedForwardHandler<M, B, O>
{
    pub _marker: PhantomData<(B, O)>,
    pub model: M
}

#[async_trait]
impl<M, B, O> BatchHandler for FeedForwardHandler<M, B, O>
where B: Backend + Unsqueezable, O: Backend,
    M: Feedforward<B, O> + 'static + Sync + Send
{
    type Request = QueueItem<B, O>;
    type ModelInput = B::Unsqueezed;
    type ModelOutput = O;


    async fn make_batch_input(&self, model_input: &mut Option<Self::ModelInput>, requests: &[Self::Request]) {
        for item in requests.iter() {
            match model_input {
                None => *model_input = Some(item.input().unsqueeze(0)),
                Some(active_tensor) => {
                    *model_input = Some(add_sequence_to_outside_of_slot(active_tensor, &item.input().clone()));
                }
            }
        }
    }

    async fn forward(&self, model_input: &Self::ModelInput) -> Self::ModelOutput {
        self.model.forward(model_input.clone()).await
    }

    async fn handle_outputs(&self, batch: &mut Vec<Self::Request>, input: &mut Option<Self::ModelInput>, output: Self::ModelOutput, active_count: Arc<Mutex<usize>>) {
        *input = None;
        *active_count.lock().await = 0;
        let to_send = slice_tensor_by_batch_dimension(output);

        // Use drain to consume the elements while keeping ownership of the vector
        for (sender, slice) in batch.drain(..).zip(to_send.iter()) {
            sender.sender().send(slice.clone()).unwrap();
        }
    }
}


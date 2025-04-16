use std::sync::Arc;
use async_trait::async_trait;
use tokio::sync::Mutex;

#[async_trait]
pub trait BatchHandler {
    type Request;
    type ModelInput: Clone;
    type ModelOutput: Clone;

    /// Build a tensor batch from queued requests.
    async fn make_batch_input(&self, model_input: &mut Option<Self::ModelInput>, requests: &[Self::Request]);

    /// Run the model forward pass.
    async fn forward(&self, model_input: &Self::ModelInput) -> Self::ModelOutput;

    /// Stream or send outputs, return requests still active.
    async fn handle_outputs(
        &self,
        batch: &mut Vec<Self::Request>,
        input: &mut Option<Self::ModelInput>,
        output: Self::ModelOutput,
        active_count: Arc<Mutex<usize>>,
    );
}

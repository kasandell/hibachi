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

/// A handler for feed-forward model inference that implements the `BatchHandler` trait.
///
/// This handler is specialized for models that process inputs in a single forward pass,
/// such as classifiers, encoders, or other non-autoregressive models.
///
/// # Type Parameters
///
/// * `M` - The model type that implements the [`Feedforward`] trait
/// * `B` - The backend tensor type for inputs, must implement [`Backend`] and [`Unsqueezable`]
/// * `O` - The backend tensor type for outputs, must implement [`Backend`]
pub struct FeedForwardHandler<M, B, O>
{
    /// Phantom data for tracking tensor types at compile time
    pub _marker: PhantomData<(B, O)>,

    /// The model that will perform the forward inference
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

    /// Creates a batched input tensor from individual requests.
    ///
    /// This method takes individual tensors from each request and combines them into a
    /// batched tensor with an extra outer dimension representing the batch.
    ///
    /// # Parameters
    ///
    /// * `model_input` - A mutable reference to the current model input, which may be None
    /// * `requests` - A slice of queued requests to be processed
    ///
    /// # Implementation Details
    ///
    /// If no model input exists, the first request's input is unsqueezed to create a batch dimension.
    /// For subsequent requests, their inputs are appended to the existing batch tensor.
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

    /// Performs the forward pass of the model on the batched input.
    ///
    /// This method simply delegates to the underlying model implementation.
    ///
    /// # Parameters
    ///
    /// * `model_input` - The batched input tensor
    ///
    /// # Returns
    ///
    /// The output tensor produced by the model
    async fn forward(&self, model_input: &Self::ModelInput) -> Self::ModelOutput {
        self.model.forward(model_input.clone()).await
    }

    /// Processes the model outputs and sends results back to requesters.
    ///
    /// For feed-forward models, this method:
    /// 1. Resets the model input since processing is complete
    /// 2. Resets the active count since all requests are completed
    /// 3. Slices the output tensor by batch dimension
    /// 4. Sends each slice to the corresponding requester
    ///
    /// # Parameters
    ///
    /// * `batch` - A mutable reference to the current batch of requests being processed
    /// * `input` - A mutable reference to the current model input
    /// * `output` - The output from the model execution
    /// * `active_count` - A shared counter of active requests
    async fn handle_outputs(&self, batch: &mut Vec<Self::Request>, input: &mut Option<Self::ModelInput>, output: Self::ModelOutput, active_count: Arc<Mutex<usize>>) {
        // Clear the input since all processing is complete
        *input = None;

        // Reset active count since all requests will be completed
        *active_count.lock().await = 0;

        // Slice output tensor by batch dimension to get individual results
        let to_send = slice_tensor_by_batch_dimension(output);

        // Use drain to consume the elements while keeping ownership of the vector
        for (sender, slice) in batch.drain(..).zip(to_send.iter()) {
            sender.sender().send(slice.clone()).unwrap();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use tokio::sync::oneshot;
    use crate::backend::{Backend, Unsqueezable};
    use crate::backend::mock_tensor::MockTensor;
    use crate::feedforward::core_trait::Feedforward;
    use crate::feedforward::queue_item::QueueItem;
    use async_trait::async_trait;

    // Mock model implementation for testing
    struct MockModel;

    #[async_trait]
    impl Feedforward<MockTensor, MockTensor> for MockModel {
        async fn forward(&self, input: MockTensor) -> MockTensor {
            // Double the value, but keep the shape
            // Remove the batch dimension (first dimension) for the output
            let mut shape = input.shape();
            let value = input.value * 2;

            // For our tests, we'll just keep the same shape but double the value
            MockTensor::new(shape, value)
        }
    }

    #[tokio::test]
    async fn test_make_batch_input_empty() {
        let handler = FeedForwardHandler {
            _marker: PhantomData,
            model: MockModel,
        };

        let mut model_input: Option<MockTensor> = None;
        let requests: Vec<QueueItem<MockTensor, MockTensor>> = vec![];

        handler.make_batch_input(&mut model_input, &requests).await;

        // Should still be None as there were no requests
        assert!(model_input.is_none());
    }

    #[tokio::test]
    async fn test_make_batch_input_single() {
        let handler = FeedForwardHandler {
            _marker: PhantomData,
            model: MockModel,
        };

        let mut model_input: Option<MockTensor> = None;

        // Create a single request
        let (tx, _rx) = oneshot::channel();
        let input_tensor = MockTensor::new(vec![3, 4], 5);
        let request = QueueItem::new(input_tensor.clone(), tx);

        let requests = vec![request];

        handler.make_batch_input(&mut model_input, &requests).await;

        // Check that model_input was updated correctly
        assert!(model_input.is_some());
        let input = model_input.unwrap();

        // Should have added a batch dimension
        assert_eq!(input.shape(), vec![1, 3, 4]);

        // Value should be preserved
        assert_eq!(input.value, 5);
    }

    #[tokio::test]
    async fn test_make_batch_input_multiple() {
        let handler = FeedForwardHandler {
            _marker: PhantomData,
            model: MockModel,
        };

        let mut model_input: Option<MockTensor> = None;

        // Create multiple requests
        let (tx1, _rx1) = oneshot::channel();
        let input_tensor1 = MockTensor::new(vec![3, 4], 5);
        let request1 = QueueItem::new(input_tensor1.clone(), tx1);

        let (tx2, _rx2) = oneshot::channel();
        let input_tensor2 = MockTensor::new(vec![3, 4], 7);
        let request2 = QueueItem::new(input_tensor2.clone(), tx2);

        let requests = vec![request1, request2];

        handler.make_batch_input(&mut model_input, &requests).await;

        // Check that model_input was updated correctly
        assert!(model_input.is_some());
        let input = model_input.unwrap();

        // Should have a batch dimension of 2
        assert_eq!(input.shape(), vec![2, 3, 4]);

        // For our mock, value is taken from the last tensor
        assert_eq!(input.value, 5);
    }

    #[tokio::test]
    async fn test_forward() {
        let handler = FeedForwardHandler {
            _marker: PhantomData,
            model: MockModel,
        };

        let input = MockTensor::new(vec![2, 3, 4], 5);
        let output = handler.forward(&input).await;

        // Value should be doubled by our mock model
        assert_eq!(output.value, 10);

        // For our mock, shape is preserved
        assert_eq!(output.shape(), vec![2, 3, 4]);
    }

    #[tokio::test]
    async fn test_handle_outputs() {
        let handler = FeedForwardHandler {
            _marker: PhantomData,
            model: MockModel,
        };

        // Setup test data
        let (tx1, rx1) = oneshot::channel();
        let (tx2, rx2) = oneshot::channel();

        let mut batch = vec![
            QueueItem::new(MockTensor::new(vec![3, 4], 5), tx1),
            QueueItem::new(MockTensor::new(vec![3, 4], 7), tx2),
        ];

        let mut input = Some(MockTensor::new(vec![2, 3, 4], 5));
        let output = MockTensor::new(vec![2, 3, 4], 10);
        let active_count = Arc::new(Mutex::new(2));

        handler.handle_outputs(&mut batch, &mut input, output, active_count.clone()).await;

        // Check that the batch was fully processed
        assert!(batch.is_empty());

        // Check that input was reset
        assert!(input.is_none());

        // Check that active count was reset
        assert_eq!(*active_count.lock().await, 0);

        // Check that results were sent correctly
        let result1 = rx1.await.unwrap();
        let result2 = rx2.await.unwrap();

        assert_eq!(result1.value, 10);
        assert_eq!(result2.value, 10);
    }
}

use super::queue_item::QueueItem;
use crate::autoregressive::Autoregressive;
use crate::backend::Backend;
use crate::backend::Unsqueezable;
use crate::core::handler::BatchHandler;
use crate::tensor::operations::{
    add_sequence_to_outside_of_slot,
    concat_output,
    pad_all_sequences,
    pad_single_sequence,
    pop_sequence_from_slot,
    slice_tensor_by_batch_dimension,
    trim_sequence,
    where_equals_stop_token
};
use async_trait::async_trait;
use std::collections::HashSet;
use std::sync::Arc;
use tokio::sync::Mutex;

/// Handler for autoregressive model inference that processes input sequences token by token.
///
/// The `AutoregressiveHandler` implements the [`BatchHandler`] trait to provide specialized
/// batch processing for autoregressive models, which generate outputs sequentially. Unlike
/// feedforward models that process entire inputs at once, autoregressive models process inputs
/// token-by-token, updating the active tokens and streaming partial results until completion
/// conditions are met.
///
/// # Type Parameters
///
/// * `M` - The autoregressive model type that can process the tensor inputs
/// * `B` - The tensor backend type implementing the [`Backend`] and [`Unsqueezable`] traits
///
pub struct AutoregressiveHandler<M, B> {
    /// The autoregressive model that will process inputs
    pub model: M,

    /// Token used for padding sequences to match batch dimensions
    pub padding_token: B,

    /// Token that indicates the end of generation for a sequence
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

    /// Constructs a batched input tensor from the current requests.
    ///
    /// For autoregressive models, this method handles:
    /// - Creating a new batch tensor if none exists
    /// - Adding new sequences to an existing batch
    /// - Ensuring all sequences in the batch have consistent dimensions through padding
    ///
    /// # Parameters
    ///
    /// * `model_input` - Current model input tensor (if any)
    /// * `requests` - Collection of requests to be processed
    async fn make_batch_input(&self, model_input: &mut Option<Self::ModelInput>, requests: &[Self::Request]) {
        for item in requests.iter() {
            let request_tensor = item.input();

            // If no model input exists yet, create one from the first request
            if model_input.is_none() {
                *model_input = Some(request_tensor.unsqueeze(0));
                continue;
            }

            // Handle adding the request to an existing model input
            if let Some(active_tensor) = model_input {
                let sequence_dims = request_tensor.shape();
                let mut request_tensor_copy = request_tensor.clone();
                let sequence_length = sequence_dims[0];
                let active_dims = active_tensor.shape();
                let batch_length = active_dims[1];

                // Adjust dimensions to ensure compatibility
                if sequence_length > batch_length {
                    // Expand active tensor with padding at the front
                    let padding_amount = sequence_length - batch_length;
                    *active_tensor = pad_all_sequences(active_tensor, padding_amount, &self.padding_token);
                } else if batch_length > sequence_length {
                    // Pad the incoming sequence to match batch length
                    let padding_amount = batch_length - sequence_length;
                    request_tensor_copy = pad_single_sequence(&request_tensor_copy, padding_amount, &self.padding_token);
                }

                // Add the prepared sequence to the batch
                *active_tensor = add_sequence_to_outside_of_slot(active_tensor, &request_tensor_copy);
            }
        }
    }

    /// Executes the model's forward pass on the input batch.
    ///
    /// For autoregressive models, this typically generates the next token
    /// for each sequence in the batch.
    ///
    /// # Parameters
    ///
    /// * `model_input` - The batched input tensor
    ///
    /// # Returns
    ///
    /// The model's output tensor containing the next token predictions
    async fn forward(&self, model_input: &Self::ModelInput) -> Self::ModelOutput {
        self.model.forward(model_input.clone()).await
    }

    /// Processes model outputs, streams results, and updates the batch state.
    ///
    /// This method:
    /// 1. Concatenates new outputs to the input history
    /// 2. Increments sequence lengths for all active items
    /// 3. Sends newly generated tokens to their respective requesters
    /// 4. Identifies completed sequences (those that generated a stop token)
    /// 5. Removes completed sequences from the batch
    /// 6. Updates the active request count
    ///
    /// # Parameters
    ///
    /// * `batch` - The current batch of requests being processed
    /// * `input` - The current model input state
    /// * `output` - The output generated by the model
    /// * `active_count` - Counter tracking active requests across all batches
    async fn handle_outputs(
        &self,
        batch: &mut Vec<Self::Request>,
        input: &mut Option<Self::ModelInput>,
        output: Self::ModelOutput,
        active_count: Arc<Mutex<usize>>,
    ) {
        // Concatenate the new output to the input history
        if let Some(input_val) = input.as_mut() {
            *input_val = concat_output(input_val, &output);
        }

        // Increment sequence length for all items in the batch
        for batch_item in batch.iter_mut() {
            batch_item.increment_sequence_length(1);
        }

        // Split the output tensor by batch dimension and send slices to requesters
        let split_outputs = slice_tensor_by_batch_dimension(output.clone());
        for (sender, slice) in batch.iter().zip(split_outputs.iter()) {
            sender.sender().send(slice.clone()).unwrap();
        }

        // Find indices of sequences that generated a stop token
        let completed_indices: HashSet<usize> = where_equals_stop_token(&output, &self.stop_token)
            .into_iter()
            .collect();

        // If no sequences ended, we're done
        let num_completed = completed_indices.len();
        if num_completed == 0 {
            return;
        }

        // Remove completed sequences from the batch
        let mut idx = 0;
        batch.retain(|_| {
            let keep = !completed_indices.contains(&idx);
            idx += 1;
            keep
        });

        // Remove completed sequences from the input tensor, starting from the highest index
        let mut sorted_indices: Vec<usize> = completed_indices.into_iter().collect();
        sorted_indices.sort_by(|a, b| b.cmp(a));  // Sort in descending order

        for &idx in &sorted_indices {
            *input = pop_sequence_from_slot(input, idx);
        }

        // Trim input tensor to the maximum sequence length of remaining items
        let max_sequence_length = QueueItem::max_seq_len_for_batch_items(batch);
        *input = trim_sequence(input, max_sequence_length);

        // Update the active request count
        {
            let mut sequence_count = active_count.lock().await;
            *sequence_count = sequence_count.saturating_sub(num_completed);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{
        mock_tensor::MockTensor
        ,
        Backend
    };
    use async_trait::async_trait;
    use std::sync::Arc;
    use tokio::sync::mpsc;
    use tokio::test;

    // A mock autoregressive model for testing
    struct MockAutoregressive;

    #[async_trait]
    impl Autoregressive<MockTensor> for MockAutoregressive {
        async fn forward(&self, input: MockTensor) -> MockTensor {
            // Simple forward pass - use same shape as input minus batch dimension
            let mut output_shape = input.shape();
            output_shape.remove(0); // Remove batch dimension

            // Return a tensor with a predictable value
            MockTensor::new(output_shape, 42)
        }
    }


    #[test]
    async fn test_make_batch_input_creates_new_batch() {
        // Create handler
        let handler = AutoregressiveHandler {
            model: MockAutoregressive,
            padding_token: MockTensor::new(vec![10], 0),
            stop_token: MockTensor::new(vec![10], 1)
        };

        // Create a request
        let (tx, _rx) = mpsc::unbounded_channel();
        let request = QueueItem::new(MockTensor::new(vec![10], 42), 10, tx);

        // Test make_batch_input with an empty model input
        let mut model_input = None;
        handler.make_batch_input(&mut model_input, &[request]).await;

        // Verify model input was created with expected shape
        assert!(model_input.is_some());
        if let Some(input) = &model_input {
            let shape = input.shape();
            assert_eq!(shape.len(), 2); // Original dimension + batch dimension
            assert_eq!(shape[0], 1);    // Batch size of 1
            assert_eq!(shape[1], 10);   // Original dimension size
        }
    }

    #[test]
    async fn test_make_batch_input_adds_to_existing_batch() {
        // Create handler
        let handler = AutoregressiveHandler {
            model: MockAutoregressive,
            padding_token: MockTensor::new(vec![10], 0),
            stop_token: MockTensor::new(vec![10], 1)
        };

        // Create initial model input
        let mut model_input = Some(MockTensor::new(vec![1, 10], 42));

        // Create a request
        let (tx, _rx) = mpsc::unbounded_channel();
        let request = QueueItem::new(MockTensor::new(vec![10], 42), 10, tx);

        // Test make_batch_input with existing model input
        handler.make_batch_input(&mut model_input, &[request]).await;

        // Verify batch dimension was increased
        assert!(model_input.is_some());
        if let Some(input) = &model_input {
            let shape = input.shape();
            assert_eq!(shape[0], 2); // Batch size should be increased to 2
        }
    }

    #[test]
    async fn test_forward_returns_expected_output() {
        // Create handler
        let handler = AutoregressiveHandler {
            model: MockAutoregressive,
            padding_token: MockTensor::new(vec![10], 0),
            stop_token: MockTensor::new(vec![10], 1)
        };

        // Create model input
        let model_input = MockTensor::new(vec![2, 10], 42);

        // Call forward
        let output = handler.forward(&model_input).await;

        // Verify output has expected shape and value
        let shape = output.shape();
        assert_eq!(shape.len(), 1);     // Batch dimension removed
        assert_eq!(shape[0], 10);       // Original dimension preserved
        assert_eq!(output.value, 42);   // Value preserved by forward
    }

    #[test]
    async fn test_handle_outputs_processes_completions() {
        // Create handler with stop token value matching output value
        let handler = AutoregressiveHandler {
            model: MockAutoregressive,
            padding_token: MockTensor::new(vec![10], 0),
            stop_token: MockTensor::new(vec![10], 42) // Same as output value to trigger completion
        };

        // Create active count
        let active_count = Arc::new(Mutex::new(2));

        // Create batch with two items
        let (tx1, mut rx1) = mpsc::unbounded_channel();
        let (tx2, mut rx2) = mpsc::unbounded_channel();
        let mut batch = vec![
            QueueItem::new(MockTensor::new(vec![10], 42), 10, tx1),
            QueueItem::new(MockTensor::new(vec![10], 42), 10, tx2)
        ];

        // Create model input and output
        let mut model_input = Some(MockTensor::new(vec![2, 10], 42));
        let output = MockTensor::new(vec![2], 42); // Value matches stop token

        // Call handle_outputs
        handler.handle_outputs(&mut batch, &mut model_input, output, active_count.clone()).await;

        // Verify both items received outputs
        assert!(rx1.try_recv().is_ok());
        assert!(rx2.try_recv().is_ok());

        // Verify batch is empty (all items completed)
        assert_eq!(batch.len(), 0);

        // Verify active count was decremented
        let count = *active_count.lock().await;
        assert_eq!(count, 0);
    }

    #[test]
    async fn test_handle_outputs_continues_generation() {
        // Create handler with stop token different from output value
        let handler = AutoregressiveHandler {
            model: MockAutoregressive,
            padding_token: MockTensor::new(vec![10], 0),
            stop_token: MockTensor::new(vec![10], 99) // Different from output value
        };

        // Create active count
        let active_count = Arc::new(Mutex::new(2));

        // Create batch with two items
        let (tx1, mut rx1) = mpsc::unbounded_channel();
        let (tx2, mut rx2) = mpsc::unbounded_channel();
        let mut batch = vec![
            QueueItem::new(MockTensor::new(vec![10], 42), 10, tx1),
            QueueItem::new(MockTensor::new(vec![10], 42), 10, tx2)
        ];

        // Create model input and output
        let mut model_input = Some(MockTensor::new(vec![2, 10], 42));
        let output = MockTensor::new(vec![2], 42); // Different from stop token

        // Call handle_outputs
        handler.handle_outputs(&mut batch, &mut model_input, output, active_count.clone()).await;

        // Verify both items received outputs
        assert!(rx1.try_recv().is_ok());
        assert!(rx2.try_recv().is_ok());

        // Verify batch still contains both items (no completions)
        assert_eq!(batch.len(), 2);

        // Verify sequence length was incremented
        assert_eq!(batch[0].len(), 11);
        assert_eq!(batch[1].len(), 11);

        // Verify active count was not decremented
        let count = *active_count.lock().await;
        assert_eq!(count, 2);
    }

    #[test]
    async fn test_end_to_end_processing() {
        // Create handler with different stop token for first and second iterations
        let mut handler = AutoregressiveHandler {
            model: MockAutoregressive,
            padding_token: MockTensor::new(vec![10], 0),
            stop_token: MockTensor::new(vec![10], 99) // Different from both output values initially
        };

        // Create active count - start with 1 to match our single request
        let active_count = Arc::new(Mutex::new(1));

        // Create batch with one item
        let (tx, mut rx) = mpsc::unbounded_channel();
        let mut batch = vec![QueueItem::new(MockTensor::new(vec![10], 21), 10, tx)];

        // First iteration - create input
        let mut model_input = None;
        handler.make_batch_input(&mut model_input, &batch).await;

        // Verify input was created
        assert!(model_input.is_some());

        // First iteration - forward pass (value 42, different from stop token 99)
        let output = handler.forward(model_input.as_ref().unwrap()).await;

        // First iteration - handle outputs (no completions yet)
        handler.handle_outputs(&mut batch, &mut model_input, output, active_count.clone()).await;

        // Verify token was received
        assert!(rx.try_recv().is_ok());

        // Verify batch still contains the item
        assert_eq!(batch.len(), 1);

        // Verify active count remains unchanged after first iteration
        {
            let count = *active_count.lock().await;
            assert_eq!(count, 1);
        }

        // Change the stop token to match the next output value
        // This is a workaround for our test since we can't control the mock output value easily
        handler.stop_token = MockTensor::new(vec![10], 42);

        // Second iteration - forward pass
        let output = handler.forward(model_input.as_ref().unwrap()).await;

        // Second iteration - handle outputs (should complete now)
        handler.handle_outputs(&mut batch, &mut model_input, output, active_count.clone()).await;

        // Verify another token was received
        assert!(rx.try_recv().is_ok());

        // Verify batch is now empty
        assert_eq!(batch.len(), 0);

        // Verify active count was decremented to 0
        let count = *active_count.lock().await;
        assert_eq!(count, 0);
    }
}

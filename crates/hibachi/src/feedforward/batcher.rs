use std::marker::PhantomData;
use std::sync::Arc;
use tokio::sync::{oneshot, Mutex};
use crate::backend::{Backend, Unsqueezable};
use super::queue_item::QueueItem;
use async_trait::async_trait;
use oneshot::channel;
use crate::communication::Pill;
use crate::core::batch::batch_inference_loop;
use crate::core::worker::BatchWorkerHandle;
use crate::feedforward::core_trait::{Feedforward, FeedforwardBatcher};
use crate::feedforward::handler::FeedForwardHandler;
use crate::feedforward::item::Item;

/// A batched inference engine for feed-forward models.
///
/// This struct implements dynamic batching for feed-forward models, allowing
/// efficient processing of multiple inference requests by grouping them into
/// batches up to a specified maximum size.
///
/// # Type Parameters
///
/// * `B` - The backend tensor type for inputs, must implement [`Backend`] and [`Unsqueezable`]
/// * `O` - The backend tensor type for outputs, must implement [`Backend`]
/// * `S` - Const generic parameter specifying the maximum batch size
///
/// # Example
///
/// ```ignore
/// use hibachi::feedforward::{FeedforwardBatchInference, Feedforward};
///
/// // Define a model
/// struct MyModel;
/// impl Feedforward<Tensor, Tensor> for MyModel {
///     async fn forward(&self, input: Tensor) -> Tensor {
///         // Model implementation
///         input
///     }
/// }
///
/// // Create a batcher with batch size 8
/// let batcher = FeedforwardBatchInference::<Tensor, Tensor, 8>::new(MyModel);
///
/// // Submit inference requests
/// async fn run_inference(batcher: &FeedforwardBatchInference<Tensor, Tensor, 8>) {
///     let input = Tensor::new(vec![3, 4], 5);
///     let result_item = batcher.run(input).await;
///     let result = result_item.await.unwrap();
/// }
/// ```
pub struct FeedforwardBatchInference<B, O, const S: usize>
{
    /// Thread-safe queue of pending inference requests
    waiting_requests: Arc<Mutex<Vec<QueueItem<B, O>>>>,

    /// Handle to the batch inference worker task
    handle: BatchWorkerHandle
}

impl <B, O, const S: usize> FeedforwardBatchInference<B, O, S>
where B: Backend + Unsqueezable, O: Backend
{
    /// Creates a new batched inference engine with the specified model.
    ///
    /// Initializes the inference engine and starts the background batch processing task.
    ///
    /// # Parameters
    ///
    /// * `model` - The model that implements the `Feedforward` trait
    ///
    /// # Returns
    ///
    /// A new `FeedforwardBatchInference` instance ready to process requests
    ///
    /// # Type Parameters
    ///
    /// * `M` - The model type implementing the `Feedforward` trait
    pub fn new<M>(
        model: M,
    ) -> Self
    where M: Feedforward<B, O> + Send + Sync + 'static,
    {
        // Create shared request queue
        let waiting_requests = Arc::new(Mutex::new(vec![]));

        // Setup communication channel for worker
        let pill = Pill::new();

        // Create worker handle that manages the background task
        let worker_handle = BatchWorkerHandle::new( {
            let waiting_requests = waiting_requests.clone();

            move |running, notifier| {
                tokio::spawn(async move {
                    #[allow(unused_variables)]
                    let moved_pill = pill;
                    let inference_handler = FeedForwardHandler{
                        _marker: PhantomData,
                        model
                    };

                    batch_inference_loop::<FeedForwardHandler<M, B, O>, S>(
                        &inference_handler,
                        running,
                        notifier,
                        waiting_requests,
                    )
                        .await;
                })
            }
        });

        Self {
            waiting_requests,
            handle: worker_handle,
        }
    }
}


/// Implements resource cleanup for the inference engine.
///
/// This implementation ensures that the background task is properly
/// shut down when the inference engine is dropped.
#[async_trait]
impl <B, O, const S: usize> FeedforwardBatcher<B, O> for FeedforwardBatchInference<B, O, S>
where B: Backend, O: Backend
{
    /// Submits an input tensor for processing and returns an `Item` that will
    /// resolve to the output once processing is complete.
    ///
    /// This method:
    /// 1. Creates a channel for receiving the result
    /// 2. Wraps the input and sender in a QueueItem
    /// 3. Adds the item to the waiting queue
    /// 4. Notifies the worker that new work is available
    /// 5. Returns an Item that will resolve to the output when ready
    ///
    /// # Parameters
    ///
    /// * `item` - The input tensor to process
    ///
    /// # Returns
    ///
    /// An `Item` that will resolve to the output tensor
    async fn run(&self, item: B) -> Item<O> {
        // Create a channel for receiving the result
        let (tx, rx) = channel();

        // Create a queue item wrapping the input and sender
        let queue_item = QueueItem::new(
            item,
            tx,
        );

        // Add the item to the waiting queue
        {
            let mut senders = self.waiting_requests.lock().await;
            senders.push(queue_item);
        }

        // Notify the worker that new work is available
        self.handle.notify();

        // Return an Item that will resolve to the output when ready
        Item::new(rx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::test;
    use std::sync::Arc;
    use tokio::sync::oneshot;
    use crate::backend::{Backend};
    use crate::backend::mock_tensor::MockTensor;
    use crate::feedforward::core_trait::Feedforward;
    use crate::feedforward::queue_item::QueueItem;
    use async_trait::async_trait;
    use crate::core::handler::BatchHandler;

    // Mock model implementation for testing
    struct MockModel;

    #[async_trait]
    impl Feedforward<MockTensor, MockTensor> for MockModel {
        async fn forward(&self, input: MockTensor) -> MockTensor {
            // Double the value, but keep the shape
            let shape = input.shape();
            let value = input.value * 2;

            MockTensor::new(shape, value)
        }
    }

    #[test]
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

    #[test]
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

    #[test]
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

    #[test]
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

    #[test]
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

        // Both results should have the same value from our mock
        assert_eq!(result1.value, 10);
        assert_eq!(result2.value, 10);
    }
}

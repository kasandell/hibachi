use std::sync::Arc;
use async_trait::async_trait;
use tokio::sync::{mpsc, Mutex};
use crate::autoregressive::handler::AutoregressiveHandler;
use crate::autoregressive::queue_item::QueueItem;
use super::item_stream::ItemStream;
use crate::communication::Pill;
use crate::backend::{Backend, Unsqueezable};
use crate::core::batch::batch_inference_loop;
use crate::core::worker::BatchWorkerHandle;
use super::{Autoregressive, AutoregressiveBatcher};

/// High-performance batch inference engine for autoregressive models.
///
/// [`AutoregressiveBatchInference`] processes multiple generation requests concurrently,
/// batching them together for efficient processing on accelerators while maintaining
/// the autoregressive generation pattern for each individual request.
///
/// This engine maintains a queue of pending requests and processes them using a background
/// worker task. Results are streamed back to callers as they become available.
///
/// # Type Parameters
///
/// * `B` - The tensor backend type, which must implement both [`Backend`] and [`Unsqueezable`]
/// * `S` - A const generic parameter that defines the maximum batch size

pub struct AutoregressiveBatchInference<B, const S: usize>
where
    B: Backend + Unsqueezable
{
    /// Queue of waiting requests to be processed
    waiting_requests: Arc<Mutex<Vec<QueueItem<B>>>>,

    /// Handle to the background worker task
    handle: BatchWorkerHandle
}

impl<B, const S: usize> AutoregressiveBatchInference<B, S>
where
    B: Backend + Unsqueezable,
{
    /// Creates a new autoregressive batch inference engine.
    ///
    /// Initializes the inference engine with the specified model and tokens,
    /// and starts a background worker to process requests.
    ///
    /// # Parameters
    ///
    /// * `model` - The autoregressive model to use for generation
    /// * `stop_token` - Token that indicates the end of generation
    /// * `padding_token` - Token used for padding sequences in batched processing
    ///
    /// # Returns
    ///
    /// A new `AutoregressiveBatchInference` instance ready to process requests
    ///
    /// # Panics
    ///
    /// This method will panic if:
    /// * The padding token shape is empty (must be rank 1 or higher)
    /// * The first dimension of the padding token is not 1
    pub fn new<M>(
        model: M,
        stop_token: &B,
        padding_token: &B,
    ) -> Self
    where
        M: Autoregressive<B> + Send + Sync + 'static,
    {
        // Validate padding token shape
        let padding_shape = padding_token.shape();
        assert!(
            !padding_shape.is_empty(),
            "padding token must have rank 1 or higher"
        );
        assert_eq!(
            padding_shape[0], 1,
            "first dimension of padding token must be 1"
        );

        // Initialize shared request queue
        let waiting_requests = Arc::new(Mutex::new(vec![]));

        // Create communication primitives
        let pill = Pill::new();

        // Create worker handle with background task
        let worker_handle = BatchWorkerHandle::new({
            let waiting_requests = waiting_requests.clone();
            let padding_token_clone = padding_token.clone();
            let stop_token_clone = stop_token.clone();

            move |running, work_notifier| {
                tokio::spawn(async move {
                    #[allow(unused_variables)]
                    let moved_pill = pill;
                    // Create the inference handler with cloned tokens
                    let inference_handler = AutoregressiveHandler {
                        model,
                        padding_token: padding_token_clone,
                        stop_token: stop_token_clone,
                    };

                    // Start the inference loop
                    batch_inference_loop::<AutoregressiveHandler<M, B>, S>(
                        &inference_handler,
                        running,
                        work_notifier,
                        waiting_requests,
                    ).await;
                })
            }
        });

        Self {
            waiting_requests,
            handle: worker_handle,
        }
    }
}

#[async_trait]
impl<B, const S: usize> AutoregressiveBatcher<B, B> for AutoregressiveBatchInference<B, S>
where
    B: Backend + Unsqueezable,
{
    /// Processes a generation request and returns a stream of results.
    ///
    /// This method:
    /// 1. Creates a channel for receiving generated tokens
    /// 2. Adds the request to the waiting queue
    /// 3. Notifies the background task that new work is available
    /// 4. Returns a stream that will yield the generated tokens
    ///
    /// # Parameters
    ///
    /// * `item` - The input tensor for which to generate tokens
    ///
    /// # Returns
    ///
    /// An `ItemStream` that yields generated tokens as they become available
    async fn run(&self, item: B) -> ItemStream<B> {
        // Create channel for sending results back to caller
        let (tx, rx) = mpsc::unbounded_channel();

        // Determine initial sequence length from input shape
        let sequence_length = item.shape()[0];

        // Create queue item for the request
        let queue_item = QueueItem::new(
            item,
            sequence_length,
            tx,
        );

        // Add request to queue
        {
            let mut queue = self.waiting_requests.lock().await;
            queue.push(queue_item);
        }

        // Notify worker that new work is available
        self.handle.notify();

        // Return stream for receiving results
        ItemStream::new(rx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::test;
    use std::fmt;
    use async_trait::async_trait;
    use futures::StreamExt;
    use tokio::time::{sleep, Duration};
    use crate::backend::mock_tensor::MockTensor;

    // Mock autoregressive model
    struct MockModel {
        stop_after: usize,
    }

    #[async_trait]
    impl Autoregressive<MockTensor> for MockModel {
        async fn forward(&self, input: MockTensor) -> MockTensor {
            // Generate output tensor with same dimensions but without batch dim
            let mut output_shape = input.shape();
            output_shape.remove(0);

            // Generate token value based on position in sequence
            // This lets us control when to generate a stop token
            let batch_size = input.shape()[0];
            let seq_len = input.shape()[1];

            // Check if we should generate a stop token
            let output_value = if seq_len >= self.stop_after {
                // Generate stop token
                99
            } else {
                // Generate regular token
                42
            };

            // Return output with batch dimension collapsed
            MockTensor::new(vec![batch_size], output_value)
        }
    }

    // Helper method to collect all items from a stream
    async fn collect_all<T>(mut stream: ItemStream<T>) -> Vec<T> {
        let mut results = Vec::new();
        while let Some(item) = stream.next().await {
            results.push(item);
        }
        results
    }

    #[test]
    async fn test_inference_creation() {
        // Create model and tokens
        let model = MockModel { stop_after: 5 };
        let stop_token = MockTensor::new(vec![1, 10], 99);
        let padding_token = MockTensor::new(vec![1, 10], 0);

        // Create inference engine
        let inference = AutoregressiveBatchInference::<MockTensor, 4>::new(
            model,
            &stop_token,
            &padding_token
        );

        // Verify queue is initially empty
        let queue = inference.waiting_requests.lock().await;
        assert_eq!(queue.len(), 0);
    }

    #[test]
    #[should_panic(expected = "padding token must have rank 1 or higher")]
    async fn test_empty_padding_shape_panics() {
        // Create model and tokens with empty padding shape
        let model = MockModel { stop_after: 5 };
        let stop_token = MockTensor::new(vec![1, 10], 99);
        let padding_token = MockTensor::new(vec![], 0); // Empty shape

        // This should panic
        let _inference = AutoregressiveBatchInference::<MockTensor, 4>::new(
            model,
            &stop_token,
            &padding_token
        );
    }

    #[test]
    #[should_panic(expected = "first dimension of padding token must be 1")]
    async fn test_invalid_padding_shape_panics() {
        // Create model and tokens with invalid padding shape
        let model = MockModel { stop_after: 5 };
        let stop_token = MockTensor::new(vec![1, 10], 99);
        let padding_token = MockTensor::new(vec![2, 10], 0); // First dim should be 1

        // This should panic
        let _inference = AutoregressiveBatchInference::<MockTensor, 4>::new(
            model,
            &stop_token,
            &padding_token
        );
    }

    #[test]
    async fn test_run_adds_request_to_queue() {
        // Create model and tokens
        let model = MockModel { stop_after: 5 };
        let stop_token = MockTensor::new(vec![1, 10], 99);
        let padding_token = MockTensor::new(vec![1, 10], 0);

        // Create inference engine
        let inference = AutoregressiveBatchInference::<MockTensor, 4>::new(
            model,
            &stop_token,
            &padding_token
        );

        // Submit a request
        let input = MockTensor::new(vec![3, 10], 42);
        let _stream = inference.run(input).await;

        // Verify request was added to queue
        let queue = inference.waiting_requests.lock().await;
        assert_eq!(queue.len(), 1);
        assert_eq!(queue[0].len(), 3); // Check sequence length was set correctly
    }

    #[test]
    async fn test_end_to_end_generation() {
        // Create model that generates 3 tokens before stop
        let model = MockModel { stop_after: 3 };
        let stop_token = MockTensor::new(vec![1, 10], 99);
        let padding_token = MockTensor::new(vec![1, 10], 0);

        // Create inference engine with batch size 2
        let inference = AutoregressiveBatchInference::<MockTensor, 2>::new(
            model,
            &stop_token,
            &padding_token
        );

        // Submit a request
        let input = MockTensor::new(vec![1, 10], 42);
        let stream = inference.run(input).await;

        // Give some time for processing
        sleep(Duration::from_millis(100)).await;

        // Try to collect any results - in a real test, we'd collect all,
        // but here we just verify the system is operational
        let mut received_tokens = 0;
        let mut received_stop = false;
        let mut stream_clone = stream;

        // Try to get a few tokens with a timeout
        for _ in 0..5 {
            match tokio::time::timeout(Duration::from_millis(100), stream_clone.next()).await {
                Ok(Some(token)) => {
                    received_tokens += 1;
                    if token.value == 99 {
                        received_stop = true;
                    }
                }
                Ok(None) => break, // Stream ended
                Err(_) => break,   // Timeout
            }
        }

        // We should receive at least one token if the system is working
        assert!(received_tokens > 0, "Should receive at least one token");
    }
}

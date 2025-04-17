use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;
use tokio::sync::{Mutex, Notify};
use super::handler::BatchHandler;

/// Runs a continuous batch inference processing loop.
///
/// This function implements an asynchronous loop that efficiently processes batches of inference
/// requests through a model. It manages dynamic batching, request processing, and output distribution.
///
/// # Parameters
///
/// * `handler` - A reference to a [`BatchHandler`] implementation that defines how to process requests,
///   execute the model, and handle outputs.
/// * `running` - Atomic flag controlling loop execution; set to `false` to gracefully terminate the loop.
/// * `notifier` - Notify instance used to wake the loop when new requests arrive.
/// * `waiting_requests` - Thread-safe queue of pending inference requests to be processed.
///
/// # Type Parameters
///
/// * `BH` - Type implementing the [`BatchHandler`] trait.
/// * `S` - Const generic parameter specifying maximum batch size.
///
/// # Behavior
///
/// The loop:
/// 1. Checks for pending requests or active processing items
/// 2. Waits for notification if no work is available
/// 3. Collects requests to form a batch up to size `S`
/// 4. Processes the batch through the model
/// 5. Distributes results and updates active requests
/// 6. Repeats until `running` is set to `false`
///
pub async fn batch_inference_loop<BH: BatchHandler, const S: usize>(
    handler: &BH,
    running: Arc<AtomicBool>,
    notifier: Arc<Notify>,
    waiting_requests: Arc<Mutex<Vec<BH::Request>>>,
) {
    let active_count = Arc::new(Mutex::new(0));
    let active_tensor: Arc<Mutex<Option<BH::ModelInput>>> = Default::default();
    let mut active_requests: Vec<BH::Request> = vec![];

    loop {
        // Exit loop if not running
        if !running.load(Ordering::SeqCst) {
            break;
        }

        // Check if we should process in this iteration
        let should_process = should_process(active_count.clone(), waiting_requests.clone()).await;

        if !should_process {
            // Wait for notification or timeout if no work is available
            tokio::select! {
                _ = notifier.notified() => {},
                _ = tokio::time::sleep(Duration::from_millis(100)) => {}
            }
        }

        // Collect eligible requests to process
        let items = drain_possible_requests(
            S, waiting_requests.clone(), active_count.clone(),
        ).await;

        // Process if we have new items or active items from previous iteration
        if !items.is_empty() || {
            let active = active_count.lock().await;
            *active > 0
        } {
            // Update the model input with new requests
            let mut tensor_lock = active_tensor.lock().await;
            handler.make_batch_input(&mut tensor_lock, &items).await;
            active_requests.extend(items);

            // Execute model inference if input is available
            let input = tensor_lock.clone();
            match input {
                None => {}
                Some(input) => {
                    let output = handler.forward(&input).await;
                    handler.handle_outputs(
                        &mut active_requests,
                        &mut tensor_lock,
                        output,
                        active_count.clone()
                    ).await;
                }
            }
        }
    }
}

/// Determines whether processing should occur in the current loop iteration.
///
/// This function checks if there are any active requests currently being processed,
/// or if there are waiting requests that need to be picked up for processing.
///
/// # Parameters
///
/// * `active_count` - Shared counter of currently active requests.
/// * `waiting_requests` - Queue of requests waiting to be processed.
///
/// # Returns
///
/// `true` if there is work to be done, `false` otherwise.
///
#[inline]
async fn should_process<T>(
    active_count: Arc<Mutex<usize>>,
    waiting_requests: Arc<Mutex<Vec<T>>>,
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

/// Extracts a batch of requests from the waiting queue.
///
/// This function calculates how many requests can be added to the current batch
/// based on the maximum batch size and current active count, then extracts up to
/// that many requests from the waiting queue.
///
/// # Parameters
///
/// * `batch_size` - Maximum number of requests that can be in a batch.
/// * `waiting_requests` - Queue of requests waiting to be processed.
/// * `active_count` - Counter tracking the number of active requests.
///
/// # Returns
///
/// A vector containing the requests that were extracted from the queue.
///
async fn drain_possible_requests<T>(
    batch_size: usize,
    waiting_requests: Arc<Mutex<Vec<T>>>,
    active_count: Arc<Mutex<usize>>,
) -> Vec<T> {
    let mut requests = waiting_requests.lock().await;
    let mut active = active_count.lock().await;
    let available_slots = batch_size.saturating_sub(*active);

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

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::time::Duration;
    use tokio::sync::{Mutex, Notify, mpsc};
    use tokio::time::timeout;

    // Define a simple request type for testing
    struct TestRequest {
        id: usize,
        input: Vec<f32>,
        sender: mpsc::Sender<Vec<f32>>,
        is_complete: Arc<AtomicBool>,
    }

    // Define a simple model input/output type for testing
    #[derive(Clone)]
    struct TestModelInput {
        batch: Vec<(usize, Vec<f32>)>, // (request_id, input_data)
    }

    #[derive(Clone)]
    struct TestModelOutput {
        results: Vec<(usize, Vec<f32>)>, // (request_id, output_data)
    }

    // Implement a test batch handler for our tests
    #[derive(Clone)]
    struct TestBatchHandler {
        // Track internal state for testing
        forward_calls: Arc<AtomicUsize>,
        make_batch_calls: Arc<AtomicUsize>,
        handle_output_calls: Arc<AtomicUsize>,
    }

    impl TestBatchHandler {
        fn new() -> Self {
            Self {
                forward_calls: Arc::new(AtomicUsize::new(0)),
                make_batch_calls: Arc::new(AtomicUsize::new(0)),
                handle_output_calls: Arc::new(AtomicUsize::new(0)),
            }
        }

        // Helper function to check call counts in tests
        fn get_metrics(&self) -> (usize, usize, usize) {
            (
                self.forward_calls.load(Ordering::SeqCst),
                self.make_batch_calls.load(Ordering::SeqCst),
                self.handle_output_calls.load(Ordering::SeqCst),
            )
        }
    }

    #[async_trait]
    impl BatchHandler for TestBatchHandler {
        type Request = TestRequest;
        type ModelInput = TestModelInput;
        type ModelOutput = TestModelOutput;

        async fn make_batch_input(
            &self,
            model_input: &mut Option<Self::ModelInput>,
            requests: &[Self::Request]
        ) {
            self.make_batch_calls.fetch_add(1, Ordering::SeqCst);

            // Get existing batch or create new one
            let mut batch = model_input.take().unwrap_or_else(||
                TestModelInput { batch: Vec::new() }
            );

            // Add new requests to batch
            for req in requests {
                batch.batch.push((req.id, req.input.clone()));
            }

            // Update model input
            *model_input = Some(batch);
        }

        async fn forward(&self, model_input: &Self::ModelInput) -> Self::ModelOutput {
            self.forward_calls.fetch_add(1, Ordering::SeqCst);

            // Simulate model computation by doubling each input value
            let results = model_input.batch.iter()
                .map(|(id, input)| {
                    let output = input.iter().map(|x| x * 2.0).collect();
                    (*id, output)
                })
                .collect();

            TestModelOutput { results }
        }

        async fn handle_outputs(
            &self,
            batch: &mut Vec<Self::Request>,
            input: &mut Option<Self::ModelInput>,
            output: Self::ModelOutput,
            active_count: Arc<Mutex<usize>>,
        ) {
            self.handle_output_calls.fetch_add(1, Ordering::SeqCst);

            // Map output results to corresponding requests
            let output_map: std::collections::HashMap<_, _> = output.results.into_iter().collect();

            // Process each request
            let mut i = 0;
            let mut completed = 0;

            while i < batch.len() {
                let req = &batch[i];
                if let Some(result) = output_map.get(&req.id) {
                    // Send result back via channel
                    let _ = req.sender.try_send(result.clone());

                    // Mark request as complete
                    req.is_complete.store(true, Ordering::SeqCst);

                    // Remove completed request
                    batch.swap_remove(i);
                    completed += 1;
                } else {
                    // Keep request in batch if not processed
                    i += 1;
                }
            }

            // Update active count by decreasing completed count
            if completed > 0 {
                let mut count = active_count.lock().await;
                *count = count.saturating_sub(completed);
            }

            // Update model input for remaining batch items (if any)
            if !batch.is_empty() {
                let remaining_items = batch.iter()
                    .map(|req| (req.id, req.input.clone()))
                    .collect();

                *input = Some(TestModelInput { batch: remaining_items });
            } else {
                *input = None;
            }
        }
    }

    // Test helper function to create a test request with a response channel
    async fn create_test_request(id: usize, input: Vec<f32>) -> (TestRequest, mpsc::Receiver<Vec<f32>>) {
        let (tx, rx) = mpsc::channel(1);
        let request = TestRequest {
            id,
            input,
            sender: tx,
            is_complete: Arc::new(AtomicBool::new(false)),
        };
        (request, rx)
    }

    #[tokio::test]
    async fn test_empty_queue() {
        const BATCH_SIZE: usize = 4;

        let handler = TestBatchHandler::new();
        let running = Arc::new(AtomicBool::new(true));
        let notifier = Arc::new(Notify::new());
        let waiting_requests = Arc::new(Mutex::new(Vec::<TestRequest>::new()));

        // Spawn inference loop with a timeout
        let running_clone = running.clone();
        let notifier_clone = notifier.clone();
        let waiting_requests_clone = waiting_requests.clone();

        let hc = handler.clone();
        let handle = tokio::spawn(async move {
            // Run the inference loop for a short time
            timeout(
                Duration::from_millis(300),
                batch_inference_loop::<TestBatchHandler, BATCH_SIZE>(
                    &hc,
                    running_clone,
                    notifier_clone,
                    waiting_requests_clone
                )
            ).await.unwrap_or(());
        });

        // Let it run for a bit without requests
        tokio::time::sleep(Duration::from_millis(200)).await;

        // Stop the loop and wait for completion
        running.store(false, Ordering::SeqCst);
        notifier.notify_one();
        handle.await.unwrap();

        // Check that no processing happened
        let (forward_calls, make_batch_calls, handle_output_calls) = handler.get_metrics();
        assert_eq!(forward_calls, 0, "No forward calls should happen with empty queue");
        assert_eq!(make_batch_calls, 0, "No make_batch calls should happen with empty queue");
        assert_eq!(handle_output_calls, 0, "No handle_output calls should happen with empty queue");
    }

    #[tokio::test]
    async fn test_single_request_processing() {
        const BATCH_SIZE: usize = 4;

        let handler = TestBatchHandler::new();
        let running = Arc::new(AtomicBool::new(true));
        let notifier = Arc::new(Notify::new());
        let waiting_requests = Arc::new(Mutex::new(Vec::<TestRequest>::new()));

        // Create a single test request
        let (request, mut result_rx) = create_test_request(1, vec![1.0, 2.0, 3.0]).await;
        let is_complete = request.is_complete.clone();

        // Add request to queue
        {
            let mut requests = waiting_requests.lock().await;
            requests.push(request);
        }

        // Spawn inference loop
        let running_clone = running.clone();
        let notifier_clone = notifier.clone();
        let waiting_requests_clone = waiting_requests.clone();

        let hc = handler.clone();
        let handle = tokio::spawn(async move {
            batch_inference_loop::<TestBatchHandler, BATCH_SIZE>(
                &hc,
                running_clone,
                notifier_clone,
                waiting_requests_clone
            ).await;
        });

        // Notify and wait for processing
        notifier.notify_one();

        // Wait for result with timeout
        let result = timeout(Duration::from_millis(500), result_rx.recv()).await;

        // Stop the loop
        running.store(false, Ordering::SeqCst);
        notifier.notify_one();
        handle.await.unwrap();

        // Verify result
        assert!(result.is_ok(), "Should receive result");
        let result = result.unwrap();
        assert!(result.is_some(), "Result should contain data");
        let output = result.unwrap();

        // Verify output values (should be doubled)
        assert_eq!(output, vec![2.0, 4.0, 6.0], "Output should be doubled input");

        // Verify request is marked as complete
        assert!(is_complete.load(Ordering::SeqCst), "Request should be marked as complete");

        // Verify call counts
        let (forward_calls, make_batch_calls, handle_output_calls) = handler.get_metrics();
        assert_eq!(forward_calls, 1, "Should have one forward call");
        assert_eq!(make_batch_calls, 1, "Should have one make_batch call");
        assert_eq!(handle_output_calls, 1, "Should have one handle_output call");

        // Check that queue is empty
        let queue = waiting_requests.lock().await;
        assert!(queue.is_empty(), "Queue should be empty after processing");
    }

    #[tokio::test]
    async fn test_batch_processing() {
        const BATCH_SIZE: usize = 4;

        let handler = TestBatchHandler::new();
        let running = Arc::new(AtomicBool::new(true));
        let notifier = Arc::new(Notify::new());
        let waiting_requests = Arc::new(Mutex::new(Vec::<TestRequest>::new()));

        // Create multiple test requests
        let (req1, mut rx1) = create_test_request(1, vec![1.0, 2.0]).await;
        let (req2, mut rx2) = create_test_request(2, vec![3.0, 4.0]).await;
        let (req3, mut rx3) = create_test_request(3, vec![5.0, 6.0]).await;

        let is_complete1 = req1.is_complete.clone();
        let is_complete2 = req2.is_complete.clone();
        let is_complete3 = req3.is_complete.clone();

        // Add requests to queue
        {
            let mut requests = waiting_requests.lock().await;
            requests.push(req1);
            requests.push(req2);
            requests.push(req3);
        }

        // Spawn inference loop
        let running_clone = running.clone();
        let notifier_clone = notifier.clone();
        let waiting_requests_clone = waiting_requests.clone();

        let hc = handler.clone();
        let handle = tokio::spawn(async move {
            batch_inference_loop::<TestBatchHandler, BATCH_SIZE>(
                &hc,
                running_clone,
                notifier_clone,
                waiting_requests_clone
            ).await;
        });

        // Notify and wait for processing
        notifier.notify_one();

        // Wait for results with timeout
        let result1 = timeout(Duration::from_millis(500), rx1.recv()).await;
        let result2 = timeout(Duration::from_millis(500), rx2.recv()).await;
        let result3 = timeout(Duration::from_millis(500), rx3.recv()).await;

        // Stop the loop
        running.store(false, Ordering::SeqCst);
        notifier.notify_one();
        handle.await.unwrap();

        // Verify results
        let res1 = result1.unwrap();
        let res2 = result2.unwrap();
        let res3 = result3.unwrap();
        assert!(res1.is_some(), "Request 1 should have result");
        assert!(res2.is_some(), "Request 2 should have result");
        assert!(res3.is_some(), "Request 3 should have result");

        // Verify output values
        assert_eq!(res1.unwrap(), vec![2.0, 4.0], "Output 1 should be doubled input");
        assert_eq!(res2.unwrap(), vec![6.0, 8.0], "Output 2 should be doubled input");
        assert_eq!(res3.unwrap(), vec![10.0, 12.0], "Output 3 should be doubled input");

        // Verify requests are marked as complete
        assert!(is_complete1.load(Ordering::SeqCst), "Request 1 should be complete");
        assert!(is_complete2.load(Ordering::SeqCst), "Request 2 should be complete");
        assert!(is_complete3.load(Ordering::SeqCst), "Request 3 should be complete");

        // Verify call counts - should have processed all in one batch
        let (forward_calls, make_batch_calls, handle_output_calls) = handler.get_metrics();
        assert_eq!(forward_calls, 1, "Should have one forward call for the batch");
        assert_eq!(make_batch_calls, 1, "Should have one make_batch call for the batch");
        assert_eq!(handle_output_calls, 1, "Should have one handle_output call for the batch");
    }

    #[tokio::test]
    async fn test_exceeding_batch_size() {
        const BATCH_SIZE: usize = 2;  // Smaller batch size to test overflow

        let handler = TestBatchHandler::new();
        let running = Arc::new(AtomicBool::new(true));
        let notifier = Arc::new(Notify::new());
        let waiting_requests = Arc::new(Mutex::new(Vec::<TestRequest>::new()));

        // Create 3 test requests (more than batch size)
        let (req1, mut rx1) = create_test_request(1, vec![1.0]).await;
        let (req2, mut rx2) = create_test_request(2, vec![2.0]).await;
        let (req3, mut rx3) = create_test_request(3, vec![3.0]).await;

        // Add requests to queue
        {
            let mut requests = waiting_requests.lock().await;
            requests.push(req1);
            requests.push(req2);
            requests.push(req3);
        }

        // Spawn inference loop
        let running_clone = running.clone();
        let notifier_clone = notifier.clone();
        let waiting_requests_clone = waiting_requests.clone();

        let hc = handler.clone();
        let handle = tokio::spawn(async move {
            batch_inference_loop::<TestBatchHandler, BATCH_SIZE>(
                &hc,
                running_clone,
                notifier_clone,
                waiting_requests_clone
            ).await;
        });

        // Notify and wait for processing
        notifier.notify_one();

        // Wait for first batch results
        let result1 = timeout(Duration::from_millis(500), rx1.recv()).await;
        let result2 = timeout(Duration::from_millis(500), rx2.recv()).await;

        // Wait a bit for the second batch to be processed
        tokio::time::sleep(Duration::from_millis(200)).await;

        // Check third result
        let result3 = timeout(Duration::from_millis(500), rx3.recv()).await;

        // Stop the loop
        running.store(false, Ordering::SeqCst);
        notifier.notify_one();
        handle.await.unwrap();

        // Verify all results received
        assert!(result1.is_ok() && result1.unwrap().is_some(), "Request 1 should have result");
        assert!(result2.is_ok() && result2.unwrap().is_some(), "Request 2 should have result");
        assert!(result3.is_ok() && result3.unwrap().is_some(), "Request 3 should have result");

        // Verify call counts - should have processed in two batches
        let (forward_calls, make_batch_calls, handle_output_calls) = handler.get_metrics();
        assert!(forward_calls >= 2, "Should have at least two forward calls");
        assert!(make_batch_calls >= 2, "Should have at least two make_batch calls");
        assert!(handle_output_calls >= 2, "Should have at least two handle_output calls");
    }

    #[tokio::test]
    async fn test_dynamic_request_addition() {
        const BATCH_SIZE: usize = 4;

        let handler = TestBatchHandler::new();
        let running = Arc::new(AtomicBool::new(true));
        let notifier = Arc::new(Notify::new());
        let waiting_requests = Arc::new(Mutex::new(Vec::<TestRequest>::new()));

        // Add first request
        let (req1, mut rx1) = create_test_request(1, vec![1.0]).await;
        {
            let mut requests = waiting_requests.lock().await;
            requests.push(req1);
        }

        // Spawn inference loop
        let running_clone = running.clone();
        let notifier_clone = notifier.clone();
        let waiting_requests_clone = waiting_requests.clone();

        let hc = handler.clone();
        let handle = tokio::spawn(async move {
            batch_inference_loop::<TestBatchHandler, BATCH_SIZE>(
                &hc,
                running_clone,
                notifier_clone,
                waiting_requests_clone
            ).await;
        });

        // Notify for first request
        notifier.notify_one();

        // Wait for first result
        let result1 = timeout(Duration::from_millis(500), rx1.recv()).await;
        assert!(result1.is_ok() && result1.unwrap().is_some(), "First request should complete");

        // Add second request after first batch is processed
        let (req2, mut rx2) = create_test_request(2, vec![2.0]).await;
        {
            let mut requests = waiting_requests.lock().await;
            requests.push(req2);
        }

        // Notify for second request
        notifier.notify_one();

        // Wait for second result
        let result2 = timeout(Duration::from_millis(500), rx2.recv()).await;
        assert!(result2.is_ok() && result2.unwrap().is_some(), "Second request should complete");

        // Stop the loop
        running.store(false, Ordering::SeqCst);
        notifier.notify_one();
        handle.await.unwrap();

        // Verify call counts
        let (forward_calls, make_batch_calls, handle_output_calls) = handler.get_metrics();
        assert_eq!(forward_calls, 2, "Should have two forward calls");
        assert_eq!(make_batch_calls, 2, "Should have two make_batch calls");
        assert_eq!(handle_output_calls, 2, "Should have two handle_output calls");
    }

    #[tokio::test]
    async fn test_graceful_shutdown() {
        const BATCH_SIZE: usize = 4;

        let handler = TestBatchHandler::new();
        let running = Arc::new(AtomicBool::new(true));
        let notifier = Arc::new(Notify::new());
        let waiting_requests = Arc::new(Mutex::new(Vec::<TestRequest>::new()));

        // Create a few requests
        let (req1, _) = create_test_request(1, vec![1.0]).await;
        let (req2, _) = create_test_request(2, vec![2.0]).await;

        // Add requests to queue
        {
            let mut requests = waiting_requests.lock().await;
            requests.push(req1);
            requests.push(req2);
        }

        // Spawn inference loop
        let running_clone = running.clone();
        let notifier_clone = notifier.clone();
        let waiting_requests_clone = waiting_requests.clone();

        let hc = handler.clone();
        let handle = tokio::spawn(async move {
            batch_inference_loop::<TestBatchHandler, BATCH_SIZE>(
                &hc,
                running_clone,
                notifier_clone,
                waiting_requests_clone
            ).await;
        });

        // Run for a bit
        tokio::time::sleep(Duration::from_millis(50)).await;

        // Signal shutdown before requests are processed
        running.store(false, Ordering::SeqCst);
        notifier.notify_one();

        // Wait for shutdown with timeout
        let shutdown_result = timeout(Duration::from_millis(300), handle).await;

        // Verify loop terminated
        assert!(shutdown_result.is_ok(), "Loop should terminate gracefully");
        assert!(shutdown_result.unwrap().is_ok(), "Loop should complete without error");
    }
}

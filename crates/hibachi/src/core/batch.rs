use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;
use tokio::sync::{Mutex, Notify};
use tokio::time::error::Elapsed;
use super::core_trait::BatchHandler;

pub async fn batching_loop<BH: BatchHandler, const S: usize>(
    handler: &BH,
    running: Arc<AtomicBool>,
    notifier: Arc<Notify>,
    waiting_requests: Arc<Mutex<Vec<BH::Request>>>,
) {
    let active_count = Arc::new(Mutex::new(0));
    let active_tensor: Arc<Mutex<Option<BH::ModelInput>>> = Default::default();
    let mut active_requests: Vec<BH::Request> = vec![];

    loop {
        if !running.load(Ordering::SeqCst) {
            break;
        }
        let should_process = should_process(active_count.clone(), waiting_requests.clone()).await;

        if !should_process {
            // No work to do, wait for notification or check periodically
            let timeout = timeout_await_notifier(&notifier).await;
            if timeout.is_err() {
                // Timeout occurred, loop back and check again
                continue;
            }
        }


        let items = drain_possible_requests(
            S, waiting_requests.clone(), active_count.clone(),
        ).await;

        if !items.is_empty() || {
            let active = active_count.lock().await;
            *active > 0
        } {
            let mut tensor_lock = active_tensor.lock().await;
            handler.make_batch_input(&mut tensor_lock, &items).await;
            active_requests.extend(items);
            let input = tensor_lock.clone();
            match input {
                None => {}
                Some(input) => {
                    let output = handler.infer(&input).await;
                    handler.handle_outputs(&mut active_requests, &mut tensor_lock, output, active_count.clone()).await;
                }
            }
        }
    }
}


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

#[inline]
async fn timeout_await_notifier(notifier: &Notify) -> Result<(), Elapsed> {
    tokio::time::timeout(
        Duration::from_millis(100),
        notifier.notified(),
    ).await
}

async fn drain_possible_requests<T>(
    batch_size: usize,
    waiting_requests: Arc<Mutex<Vec<T>>>,
    active_count: Arc<Mutex<usize>>,
) -> Vec<T> {
    // before we even get to the requests lock
    let mut requests = waiting_requests.lock().await;
    let mut active = active_count.lock().await;
    // Calculate how many items we can process
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

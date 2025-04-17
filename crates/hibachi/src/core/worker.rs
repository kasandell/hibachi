//! Module for handling background batch processing tasks.

use std::sync::{Arc, atomic::{AtomicBool, Ordering}};
use tokio::{task::JoinHandle, sync::Notify};

/// A handle for managing a background worker task that performs batch operations.
///
/// This struct provides a convenient way to spawn, manage, and gracefully shut down
/// long-running background tasks in the Tokio runtime.
///
/// # Example
///
/// ```ignore
/// use hibachi::core::batch::BatchInferenceEngine;
/// use std::sync::{Arc, atomic::{AtomicBool, Ordering}};
/// use tokio::sync::Notify;
/// use std::time::Duration;
///
/// async fn example() {
///     let worker = BatchWorkerHandle::new(|running, notifier| {
///         tokio::spawn(async move {
///             while running.load(Ordering::SeqCst) {
///                 // Process batch of work
///                 println!("Working...");
///
///                 // Wait for next notification or timeout
///                 tokio::select! {
///                     _ = notifier.notified() => println!("Notified!"),
///                     _ = tokio::time::sleep(Duration::from_secs(5)) => println!("Timeout"),
///                 }
///             }
///             println!("Worker stopped");
///         })
///     });
///
///     // Notify the worker to process a batch
///     worker.notify();
///
///     // Worker will be automatically shut down when dropped
/// }
/// ```
pub struct BatchWorkerHandle {
    /// Flag indicating whether the background task should continue running
    running: Arc<AtomicBool>,

    /// Handle to the spawned background task, becomes `None` after shutdown is initiated
    handle: Option<JoinHandle<()>>,

    /// Notification mechanism to wake up the background task
    notifier: Arc<Notify>,
}

impl BatchWorkerHandle {
    /// Creates a new `BatchWorkerHandle` by spawning a background task.
    ///
    /// # Parameters
    ///
    /// * `task` - A function that takes a running flag and a notifier, and returns a `JoinHandle`.
    ///   This function is responsible for creating and spawning the actual background task.
    ///
    /// # Returns
    ///
    /// A new `BatchWorkerHandle` instance with the task running.
    pub fn new<F>(task: F) -> Self
    where
        F: FnOnce(Arc<AtomicBool>, Arc<Notify>) -> JoinHandle<()> + Send + 'static,
    {
        let running = Arc::new(AtomicBool::new(true));
        let notifier = Arc::new(Notify::new());
        let handle = task(running.clone(), notifier.clone());

        Self {
            running,
            handle: Some(handle),
            notifier,
        }
    }

    /// Notifies the background task to wake up and process any pending items.
    ///
    /// This typically triggers the worker to process a batch of items.
    pub fn notify(&self) {
        self.notifier.notify_one();
    }

    #[allow(dead_code)]
    /// Returns a clone of the atomic boolean that indicates whether the task should continue running.
    pub fn running(&self) -> Arc<AtomicBool> {
        self.running.clone()
    }

    #[allow(dead_code)]
    /// Returns a clone of the notifier that can be used to wake up the background task.
    pub fn notifier(&self) -> Arc<Notify> {
        self.notifier.clone()
    }

    /// Initiates a graceful shutdown of the background task.
    ///
    /// This method:
    /// 1. Sets the running flag to `false`
    /// 2. Notifies the task to wake up (so it can observe that it should stop)
    /// 3. Takes ownership of the task handle and spawns a separate task to await its completion
    pub fn shutdown(&mut self) {
        self.running.store(false, Ordering::SeqCst);
        self.notifier.notify_one();

        if let Some(handle) = self.handle.take() {
            tokio::spawn(async move {
                let _ = handle.await;
            });
        }
    }
}

impl Drop for BatchWorkerHandle {
    /// Ensures the background task is properly terminated when the handle is dropped.
    fn drop(&mut self) {
        self.shutdown();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;
    use std::time::Duration;
    use tokio::time;

    #[tokio::test]
    async fn test_worker_starts_running() {
        let worker = BatchWorkerHandle::new(|running, _notifier| {
            tokio::spawn(async move {
                while running.load(Ordering::SeqCst) {
                    time::sleep(Duration::from_millis(10)).await;
                }
            })
        });

        assert!(worker.running().load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn test_worker_notifies() {
        let notification_count = Arc::new(Mutex::new(0));
        let notification_count_clone = notification_count.clone();

        let worker = BatchWorkerHandle::new(|running, notifier| {
            tokio::spawn(async move {
                while running.load(Ordering::SeqCst) {
                    notifier.notified().await;
                    let mut count = notification_count_clone.lock().unwrap();
                    *count += 1;
                }
            })
        });

        // Wait a bit to ensure the task is running
        time::sleep(Duration::from_millis(50)).await;

        // Notify the worker and check that it processed the notification
        worker.notify();
        time::sleep(Duration::from_millis(50)).await;

        assert_eq!(*notification_count.lock().unwrap(), 1);

        // Notify again and check count increases
        worker.notify();
        time::sleep(Duration::from_millis(50)).await;

        assert_eq!(*notification_count.lock().unwrap(), 2);
    }

    #[tokio::test]
    async fn test_worker_shutdown() {
        let is_shutdown = Arc::new(AtomicBool::new(false));
        let is_shutdown_clone = is_shutdown.clone();

        let mut worker = BatchWorkerHandle::new(|running, notifier| {
            tokio::spawn(async move {
                while running.load(Ordering::SeqCst) {
                    notifier.notified().await;
                }
                is_shutdown_clone.store(true, Ordering::SeqCst);
            })
        });

        // Ensure worker starts running
        assert!(worker.running().load(Ordering::SeqCst));

        // Trigger the worker once to ensure it enters the notification wait
        worker.notify();
        time::sleep(Duration::from_millis(50)).await;

        // Shut down the worker
        worker.shutdown();

        // Allow time for shutdown to complete
        time::sleep(Duration::from_millis(100)).await;

        // Check that running flag is set to false
        assert!(!worker.running().load(Ordering::SeqCst));

        // Check that the worker observed shutdown and set our flag
        assert!(is_shutdown.load(Ordering::SeqCst));

        // Check that handle was taken (is None)
        assert!(worker.handle.is_none());
    }

    #[tokio::test]
    async fn test_worker_drop_triggers_shutdown() {
        let is_shutdown = Arc::new(AtomicBool::new(false));
        let is_shutdown_clone = is_shutdown.clone();

        {
            // Create worker in a new scope so it will be dropped
            let worker = BatchWorkerHandle::new(|running, notifier| {
                tokio::spawn(async move {
                    while running.load(Ordering::SeqCst) {
                        notifier.notified().await;
                    }
                    is_shutdown_clone.store(true, Ordering::SeqCst);
                })
            });

            // Trigger the worker once to ensure it enters the notification wait
            worker.notify();
            time::sleep(Duration::from_millis(50)).await;

            // Worker will be dropped here
        }

        // Allow time for drop/shutdown to complete
        time::sleep(Duration::from_millis(100)).await;

        // Check that the worker observed shutdown and set our flag
        assert!(is_shutdown.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn test_worker_can_access_notifier() {
        let notification_received = Arc::new(AtomicBool::new(false));
        let notification_received_clone = notification_received.clone();

        let worker = BatchWorkerHandle::new(|running, _notifier| {
            // We'll use our own notifier fetched from the handle
            let notifier_clone = Arc::clone(&_notifier);

            tokio::spawn(async move {
                while running.load(Ordering::SeqCst) {
                    notifier_clone.notified().await;
                    notification_received_clone.store(true, Ordering::SeqCst);
                }
            })
        });

        // Get the notifier from the handle and use it
        let notifier = worker.notifier();
        notifier.notify_one();

        // Allow time for notification to be processed
        time::sleep(Duration::from_millis(50)).await;

        // Check that the notification was received
        assert!(notification_received.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn test_shutdown_after_handle_already_taken() {
        let mut worker = BatchWorkerHandle::new(|running, _notifier| {
            tokio::spawn(async move {
                while running.load(Ordering::SeqCst) {
                    time::sleep(Duration::from_millis(10)).await;
                }
            })
        });

        // Take the handle first
        let _ = worker.handle.take();

        // This should not panic even though handle is None
        worker.shutdown();

        // Verify running flag is still set to false
        assert!(!worker.running().load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn test_multiple_shutdowns() {
        let mut worker = BatchWorkerHandle::new(|running, _notifier| {
            tokio::spawn(async move {
                while running.load(Ordering::SeqCst) {
                    time::sleep(Duration::from_millis(10)).await;
                }
            })
        });

        // Shut down once
        worker.shutdown();

        // This should not panic
        worker.shutdown();

        // And again, just to be sure
        worker.shutdown();

        assert!(!worker.running().load(Ordering::SeqCst));
    }
}

use std::sync::{Arc, atomic::{AtomicBool, Ordering}};
use tokio::{task::JoinHandle, sync::Notify};

pub struct BatchWorkerHandle {
    running: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
    notifier: Arc<Notify>,
}

impl BatchWorkerHandle {
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

    pub fn notify(&self) {
        self.notifier.notify_one();
    }

    pub fn running(&self) -> Arc<AtomicBool> {
        self.running.clone()
    }

    pub fn notifier(&self) -> Arc<Notify> {
        self.notifier.clone()
    }

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
    fn drop(&mut self) {
        self.shutdown();
    }
}

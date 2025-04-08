use tokio::sync::mpsc;

/// Queue item struct to store both the input tensor and the sender for results
pub struct QueueItem<Q, T> {
    input: Q,
    sender: mpsc::UnboundedSender<T>
}

impl <Q, T> QueueItem<Q, T> {
    pub fn new(input: Q, sender: mpsc::UnboundedSender<T>) -> Self {
        Self {
            input,
            sender
        }
    }

    pub fn input(&self) -> &Q {
        &self.input
    }

    pub fn sender(&self) -> mpsc::UnboundedSender<T> {
        self.sender.clone()
    }
}

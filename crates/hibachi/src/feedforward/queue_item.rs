use tokio::sync::oneshot::Sender;

/// # QueueItem
///
/// A container for associating input data with a channel for sending results.
///
/// `QueueItem` pairs an input value with an unbounded sender channel,
/// creating a complete work item that can be processed asynchronously.
/// The input value (of type `Q`) is what needs to be processed, and the
/// sender (for type `T`) is the channel where results should be sent.
///
/// ## Usage Context
///
/// This struct is typically used in asynchronous processing queues
/// where work items are submitted for processing and results are
/// returned through channels to the original requesters.
///
/// ## Type Parameters
///
/// * `Q` - The type of the input value to be processed
/// * `T` - The type of the results that will be sent back
pub struct QueueItem<Q, T> {
    /// The input value to be processed
    #[allow(dead_code)]
    input: Q,

    /// Channel for sending results back to the requester
    sender: Sender<T>
}

impl<Q, T> QueueItem<Q, T> {
    /// Creates a new `QueueItem` with the specified input and sender.
    ///
    /// # Parameters
    ///
    /// * `input` - The value to be processed
    /// * `sender` - Channel for sending results back to the requester
    ///
    /// # Returns
    ///
    /// A new `QueueItem` instance
    #[allow(dead_code)]
    pub fn new(input: Q, sender: Sender<T>) -> Self {
        Self {
            input,
            sender
        }
    }

    /// Returns a reference to the input value.
    ///
    /// # Returns
    ///
    /// A reference to the input that needs to be processed
    #[allow(dead_code)]
    pub fn input(&self) -> &Q {
        &self.input
    }

    /// Returns the sender channel for this queue item.
    #[allow(dead_code)]
    pub fn sender(self) -> Sender<T> {
        self.sender
    }
}

impl<Q, T> AsRef<Sender<T>> for QueueItem<Q, T> {
    /// Implements the `AsRef` trait to allow borrowing the sender directly.
    ///
    /// This enables the `QueueItem` to be treated as a reference to its
    /// sender in contexts where a reference to the sender is needed.
    ///
    /// # Returns
    ///
    /// A reference to the sender channel
    fn as_ref(&self) -> &Sender<T> {
        &self.sender
    }
}

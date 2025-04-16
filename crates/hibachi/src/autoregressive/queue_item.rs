use tokio::sync::mpsc;
use tokio::sync::mpsc::UnboundedSender;
use uuid::Uuid;

#[derive(Debug, Clone)]
pub struct QueueItem<T> {
    /// Unique identifier for this batch item
    id: Uuid,

    #[allow(dead_code)]
    input: T,

    /// Current length of the sequence associated with this batch item
    #[allow(dead_code)]
    sequence_length: usize,

    /// Channel for sending processed results back to the requester
    sender: mpsc::UnboundedSender<T>
}

impl<T> QueueItem<T> {
    /// Creates a new `BatchItem` with the specified initial sequence length and sender.
    ///
    /// # Parameters
    ///
    /// * `sequence_length` - Initial sequence length, which may be non-zero if
    ///   the sequence already contains tokens
    /// * `sender` - Channel for sending processed results
    ///
    /// # Returns
    ///
    /// A new `BatchItem` instance with a randomly generated UUID
    #[allow(dead_code)]
    pub fn new(input: T, sequence_length: usize, sender: mpsc::UnboundedSender<T>) -> Self {
        Self {
            input,
            id: Uuid::new_v4(),
            sequence_length,
            sender,
        }
    }

    /// Returns a reference to the sender channel for dispatching messages.
    ///
    /// # Returns
    ///
    /// A reference to the unbounded sender channel
    #[allow(dead_code)]
    pub fn sender(&self) -> &mpsc::UnboundedSender<T> {
        &self.sender
    }

    /// Increases the tracked sequence length by the specified amount.
    ///
    /// This is typically called when new tokens are added to the sequence
    /// associated with this batch item.
    ///
    /// # Parameters
    ///
    /// * `amount` - The number of tokens/elements to add to the sequence length
    #[allow(dead_code)]
    pub fn increment_sequence_length(&mut self, amount: usize) {
        self.sequence_length += amount
    }

    /// Returns the current sequence length for this batch item.
    ///
    /// The sequence length corresponds to the number of tokens or elements
    /// in the sequence associated with this batch item.
    ///
    /// # Returns
    ///
    /// The current sequence length
    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.sequence_length
    }

    /// Checks if the sequence length is zero.
    ///
    /// # Returns
    ///
    /// `true` if the sequence length is zero, `false` otherwise
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.sequence_length == 0
    }

    /// Returns the unique identifier for this batch item.
    ///
    /// # Returns
    ///
    /// The UUID of this batch item
    #[allow(dead_code)]
    pub fn id(&self) -> Uuid {
        self.id
    }

    #[allow(dead_code)]
    pub fn input(&self) -> &T {
        &self.input
    }

    /// Finds the maximum sequence length across a collection of batch items.
    ///
    /// This is useful for determining padding requirements in batched processing.
    ///
    /// # Parameters
    ///
    /// * `batch_items` - A slice of `BatchItem` instances to examine
    ///
    /// # Returns
    ///
    /// The maximum sequence length found, or 0 if the collection is empty
    #[allow(dead_code)]
    pub fn max_seq_len_for_batch_items(
        batch_items: &[QueueItem<T>]
    ) -> usize {
        batch_items.iter()
            .map(|item| item.len())
            .max()
            .unwrap_or(0)
    }
}

impl<T> PartialEq for QueueItem<T> {
    /// Compares two `BatchItem` instances for equality based on their unique IDs.
    ///
    /// Note that equality is determined solely by the UUID, not by sequence length
    /// or the sender channels.
    ///
    /// # Parameters
    ///
    /// * `other` - Another `BatchItem` to compare with
    ///
    /// # Returns
    ///
    /// `true` if both batch items have the same ID, `false` otherwise
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl<T> Eq for QueueItem<T> {}

impl<T> AsRef<UnboundedSender<T>> for QueueItem<T> {
    fn as_ref(&self) -> &UnboundedSender<T> {
        &self.sender
    }
}

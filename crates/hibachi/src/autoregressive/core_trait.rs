use async_trait::async_trait;
use crate::backend::{Backend, Unsqueezable};
use super::item_stream::ItemStream;

/// # Autoregressive
///
/// A trait for models that support autoregressive generation with batched processing.
/// ```rust
/// # use std::io;
/// use hibachi::autoregressive::Autoregressive;
/// use candle_core::{DType, Device, Tensor};
/// use futures::StreamExt;
/// use async_trait::async_trait;
///
///
/// pub struct Model {}
///
/// #[async_trait]
/// impl Autoregressive<Tensor> for Model {
///
///     async fn forward(&self, tensor: Tensor) -> Tensor {
///         // Extract the dimensions we need
///         let batch_size = tensor.dims()[0];
///         Tensor::ones(&[batch_size], tensor.dtype(), tensor.device()).unwrap()
///     }
/// }
///
/// # #[tokio::main]
/// # async fn main() -> io::Result<()> {
/// let device = Device::Cpu;
/// let model = Model {};
///
/// let input = Tensor::zeros(&[3], DType::U8, &device).expect("creates start token");
/// let output = model.forward(input).await;
/// # Ok(())
/// # }
/// ```
///
/// This trait defines the core interface for models that generate output tokens
/// sequentially based on previously generated tokens. It is specifically designed
/// for use with the batched inference engine, which handles the mechanics of
/// efficiently processing multiple generation requests concurrently.
///
/// ## Input/Output Dimensions
///
/// The expected input dimensions are `(batch, seq, **tok_dimensions)`, where:
/// - `batch`: The batch size (number of sequences being processed)
/// - `seq`: The sequence length (number of tokens in each sequence)
/// - `**tok_dimensions`: Any additional dimensions describing token representations
///
/// The output dimensions are `(batch, **tok_dimensions)`, representing the next token
/// for each sequence in the batch.
///
/// ## Implementation Notes
///
/// When implementing this trait:
/// - The input tensor includes the full context of previously generated tokens
/// - The model should return logits or representations for the next token only
/// - The batching engine will automatically append generated tokens to the input
///   for subsequent generation steps
/// - The implementation should be compatible with the [`Backend`] and [`Unsqueezable`] constraints
///
///
/// ## Usage Context
///
/// This trait is primarily used by the batching engine to coordinate efficient
/// autoregressive generation across multiple requests. Models implementing this trait
/// can be plugged into the batching system to benefit from optimized resource utilization.
#[async_trait]
pub trait Autoregressive<B> where B: Backend + Unsqueezable
{
    /// Performs a single forward pass for autoregressive generation.
    ///
    /// # Parameters
    ///
    /// * `tensor` - Input tensor with shape `(batch, seq, **tok_dimensions)` containing
    ///   the token sequences for which to generate the next tokens.
    ///
    /// # Returns
    ///
    /// A tensor with shape `(batch, **tok_dimensions)` containing the predicted
    /// next token representations for each sequence in the batch.
    ///
    /// # Async Behavior
    ///
    /// This method is asynchronous to allow for non-blocking execution, which is
    /// particularly important for models that may involve remote API calls or
    /// significant computation that should not block the executor.
    async fn forward(&self, tensor: B::Unsqueezed) -> B;
}

/// # AutoregressiveBatcher
///
/// A trait for components that manage batched processing of autoregressive generation requests.
///
/// The `AutoregressiveBatcher` is responsible for:
/// 1. Collecting multiple generation requests
/// 2. Batching them efficiently
/// 3. Scheduling their execution on the underlying model
/// 4. Streaming results back to the requesters
///
/// ## Type Parameters
///
/// * `T` - The input item type (typically a generation request with parameters)
/// * `Q` - The output item type (typically generated tokens or sequences)
///
/// ## Implementation Notes
///
/// Implementations of this trait should handle:
/// - Dynamic batch construction and management
/// - Efficient scheduling of model invocations
/// - Proper distribution of results to the correct output streams
/// - Resource management (e.g., ensuring that memory usage is bounded)
/// - Error handling and recovery
///
/// ## Usage Context
///
/// This trait represents the primary interface for clients to interact with
/// the batched inference engine. Clients submit generation requests through
/// the `run` method and receive results asynchronously through the returned
/// `ItemStream`.
#[async_trait]
pub trait AutoregressiveBatcher<T, Q> {
    /// Processes a generation request and returns a stream of results.
    ///
    /// # Parameters
    ///
    /// * `item` - The generation request to process, containing the input and any
    ///   generation parameters.
    ///
    /// # Returns
    ///
    /// An `ItemStream` that yields the generated outputs as they become available.
    /// The stream completes when generation is finished for the given request.
    ///
    /// # Async Behavior
    ///
    /// This method returns immediately with a stream, without waiting for generation
    /// to complete. The actual generation happens asynchronously in the background,
    /// with results being pushed to the returned stream as they become available.
    async fn run(&self, item: T) -> ItemStream<Q>;
}

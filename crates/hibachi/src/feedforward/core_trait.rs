use async_trait::async_trait;
use crate::backend::{Backend, Unsqueezable};
use super::item::Item;

/// # Feedforward
///
/// Defines a feed-forward model interface that processes tensors in a single pass.
///
/// ```rust
/// # use std::io;
/// use hibachi::feedforward::Feedforward;
/// use candle_core::{Tensor, DType, Device};
/// use async_trait::async_trait;
///
/// struct MyModel {
///     weights: Tensor,
/// }
///
/// #[async_trait]
/// impl Feedforward<Tensor, Tensor> for MyModel {
///     async fn forward(&self, input: Tensor) -> Tensor {
///         input.matmul(&self.weights).unwrap()
///     }
/// }
///
/// # #[tokio::main]
/// # async fn main() -> io::Result<()> {
/// let device = Device::Cpu;
/// let model = MyModel { weights: Tensor::ones(&[64, 10], DType::F16, &device).unwrap() };
///
/// // Note the extra batch dimension. Normally the batcher handles this dimension
/// let input = Tensor::ones(&[1, 64], DType::F16, &device).expect("creates start token");
/// let output = model.forward(input).await;
///
/// # Ok(())
/// # }
/// ```
///
/// This trait represents models that take an input tensor and produce an output
/// tensor in a single forward pass, without autoregressive or iterative behavior.
/// Typical implementations include classification models, encoders, and transformations.
///
/// # Type Parameters
///
/// * `B` - The input tensor type that implements [`Backend`] and [`Unsqueezable`]
/// * `O` - The output tensor type that implements [`Backend`]
///
/// # Implementation Notes
///
/// Implementations should:
/// * Handle batched inputs with the first dimension as the batch dimension
/// * Preserve the batch structure in outputs
/// * Be thread-safe and non-blocking
///
#[async_trait]
pub trait Feedforward<B, O> where B: Backend + Unsqueezable, O: Backend
{
    /// Processes an input tensor and produces an output tensor.
    ///
    /// This method represents the core computation of the feed-forward model.
    ///
    /// # Parameters
    ///
    /// * `tensor` - The input tensor to process, with an additional outer batch dimension
    ///
    /// # Returns
    ///
    /// The output tensor produced by the model
    ///
    /// # Note
    ///
    /// The input tensor is of type `B::Unsqueezed` because it includes an additional
    /// batch dimension compared to the raw input type `B`.
    async fn forward(&self, tensor: B::Unsqueezed) -> O;
}

#[async_trait]
pub trait FeedforwardBatcher<T, Q> {
    /// Submits an input tensor for asynchronous processing.
    ///
    /// This method queues the input for processing and returns an `Item`
    /// that will resolve to the output when processing is complete.
    ///
    /// # Parameters
    ///
    /// * `item` - The input tensor to process
    ///
    /// # Returns
    ///
    /// An `Item` that will resolve to the output tensor
    ///
    /// # Note
    ///
    /// The returned `Item` is a future-like type that can be awaited
    /// to obtain the final output tensor.
    async fn run(&self, item: T) -> Item<Q>;
}

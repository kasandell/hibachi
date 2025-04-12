use async_trait::async_trait;
use crate::backend::Backend;
use crate::communcation::ItemStream;

/// The forward trait that all models wanting to use the batched interface must implement.
/// for autoregressive generation. Expected input dimensions are (batch, seq, **tok_dimensions) -> (batch, **tok_dimensions)
/// Internally, the batching engine will append the output tensor to the input for autoregressive models
#[async_trait]
pub trait Autoregressive<B> where B: Backend {
    async fn forward(&self, tensor: B) -> B;
}

#[async_trait]
pub trait AutoregressiveBatcher<T, Q> {
    async fn run(&self, item: T) -> ItemStream<Q>;
}

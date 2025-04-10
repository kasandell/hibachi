use async_trait::async_trait;

/// The forward trait that all models wanting to use the batched interface must implement.
/// for autoregressive generation. Expected input dimensions are (batch, seq, **tok_dimensions) -> (batch, **tok_dimensions)
/// Internally, the batching engine will append the output tensor to the input for autoregressive models
#[async_trait]
pub trait Autoregressive {
    type Sequence;
    type Output;

    async fn forward(&self, tensor: Self::Sequence) -> Self::Output;
}

use async_trait::async_trait;
use burn::prelude::{Backend, Tensor};

/// The forward trait that all models wanting to use the batched interface must implement.
/// Requires input of a rank 3 tensor, and guarantees output of a rank 2 tensor
/// for autoregressive generation. Expected dimensions are (batch, seq, tok) -> (batch, tok)
/// Internally, the batching engine will append the output tensor to the input for autoregressive
/// behavior
#[async_trait]
pub trait Forward<B>
where B: Backend {
    fn forward(&self, tensor: Tensor<B, 3>) -> Tensor<B, 2>;
}

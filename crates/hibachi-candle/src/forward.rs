use async_trait::async_trait;
use candle_core::Tensor;

/// The forward trait that all models wanting to use the batched interface must implement.
/// Requires input of a rank 3 tensor, and guarantees output of a rank 2 tensor
/// for autoregressive generation. Expected dimensions are (batch, seq, tok) -> (batch, tok)
/// Internally, the batching engine will append the output tensor to the input for autoregressive
/// behavior
/// TODO: this actually likely needs to be Tensor<B, 2> -> Tensor<B, 3>
/// this should take in [batch_size, seq_len] token ids, tokenize them, and return out
/// [batch_size] token_ids or even potentially we force the selection of the token, so it is really
/// [batch_size, seq_len] -> [batch_size] -> [seq_len]
#[async_trait]
pub trait CandleForward {
    async fn forward(&self, tensor: Tensor) -> Tensor;
}

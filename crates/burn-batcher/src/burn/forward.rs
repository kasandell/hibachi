use async_trait::async_trait;
use burn::prelude::{Backend, Tensor};

#[async_trait]
pub trait Forward<B>//<B, const S_IN: usize, const S_OUT: usize>
where B: Backend {
    fn forward(&self, tensor: Tensor<B, 3>) -> Tensor<B, 2>;
}

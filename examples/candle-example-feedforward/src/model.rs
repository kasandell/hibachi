use hibachi::feedforward::Feedforward;
use async_trait::async_trait;
use candle_core::{Tensor};

pub struct Model {}

impl Model {
    pub fn new() -> Self  {
        Self {
        }
    }
}

#[async_trait]
impl Feedforward<Tensor, Tensor> for Model {

    async fn forward(&self, tensor: Tensor) -> Tensor {
        // Extract the dimensions we need
        let batch_size = tensor.dims()[0];
        let zeros = Tensor::zeros(&[batch_size, 3, 3], tensor.dtype(), tensor.device()).unwrap();
        zeros
    }
}

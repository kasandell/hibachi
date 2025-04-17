use hibachi::feedforward::Feedforward;
use async_trait::async_trait;
use rand::{thread_rng, Rng};
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
        let mut rng = thread_rng();
        let val = rng.gen_range(0..=1);
        let mut zeros = Tensor::zeros(&[batch_size, 3, 3], tensor.dtype(), tensor.device()).unwrap();
        zeros
    }
}

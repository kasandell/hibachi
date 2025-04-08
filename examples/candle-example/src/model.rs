use hibachi::{Autoregressive};
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
impl Autoregressive for Model {
    type Sequence = Tensor;
    type Output = Tensor;

    async fn forward(&self, tensor: Self::Sequence) -> Self::Output {
        // Extract the dimensions we need
        let batch_size = tensor.dims()[0];
        let mut rng = thread_rng();
        let val = rng.gen_range(0..batch_size/10);
        let mut zeros = Tensor::zeros(&[batch_size], tensor.dtype(), &tensor.device()).unwrap();
        let ones = Tensor::ones(&[1], tensor.dtype(), &tensor.device()).unwrap();
        for _ in 0..val {
            let idx = rng.gen_range(0..batch_size);
            zeros = zeros.slice_assign(&[idx..idx+1], &ones).unwrap();
        }
        zeros
    }
}

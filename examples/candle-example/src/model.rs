use hibachi::autoregressive::{Autoregressive};
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
impl Autoregressive<Tensor> for Model {

    async fn forward(&self, tensor: Tensor) -> Tensor {
        // Extract the dimensions we need
        let batch_size = tensor.dims()[0];
        let mut rng = thread_rng();
        let val = rng.gen_range(0..=batch_size);
        let mut zeros = Tensor::zeros(&[batch_size], tensor.dtype(), tensor.device()).unwrap();
        let ones = Tensor::ones(&[1], tensor.dtype(), tensor.device()).unwrap();
        for _ in 0..val {
            let idx = rng.gen_range(0..batch_size);
            zeros = zeros.slice_assign(&[idx..idx+1], &ones).unwrap();
        }
        zeros
        /*
        let mut rng = thread_rng();
        let mut shape = tensor.dims().to_vec();
        shape.remove(1);

        if rng.gen_range(0..1000) < 5 {
            return Tensor::ones(shape, tensor.dtype(), &tensor.device()).unwrap();
        }
        let zeros = Tensor::zeros(shape, tensor.dtype(), &tensor.device()).unwrap();
        zeros

         */
    }
}
